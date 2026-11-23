"""Rigorous distillation column (MESH equations), the DWSIM "Distillation Column".

Model (phase 1, validated against DWSIM 10.2.12.0 on the benzene/toluene tutorial column, see
reference_data/dwsim/rigorous_column_benzene_toluene.json):

* N equilibrium stages numbered from the top as in DWSIM: stage 1 is the condenser and stage N the reboiler, so the
  column has N - 2 trays. Total condenser (no vapor leaves stage 1: the vapor of stage 2 is condensed completely,
  at its bubble point, and split into reflux and distillate) and kettle reboiler (an equilibrium stage whose liquid
  is the bottoms product).
* One feed, of any thermal state, enters a stage as a whole: its liquid and vapor are added to the stage balances
  with the feed enthalpy (this is what DWSIM does, see the profile of the validation case).
* Uniform pressure (no pressure drop, one efficiency of 1: equilibrium stages).
* Specifications: reflux ratio (condenser) and bottoms molar flow (reboiler). Other specifications raise
  NotImplementedError until they are validated.

Equations per stage j (component i): material balance, equilibrium ln(y_i/x_i) = ln K_i with K_i = phi_L,i(x) /
phi_V,i(y) (the property method of the feed stream, see `StageThermo`) and energy balance. The condenser has its
material balance, equilibrium of the incipient vapor and sum y = 1 (bubble point); the reboiler replaces the energy
balance by the bottoms flow specification. The duties are explicit outputs of the energy balances of those two stages.

Solver ``'naphtali_sandholm'``: Newton on all the unknowns at once (ln of the component flows of liquid and vapor and
T of each stage, (2C + 1) N unknowns), analytic structure of a block tridiagonal Jacobian evaluated by finite
differences with stage coloring (3 colors, each one perturbs one variable of every third stage at once), damping on
the step and a backtracking line search. Logs of the flows keep every flow positive and give relative accuracy to
trace components. The converged solution does not depend on the algorithm: the same equations have the same
solution in the DWSIM Wang-Henke, Naphtali-Sandholm and Inside-Out options up to their convergence tolerance
(1e-5 in DWSIM), so the validation compares profiles and not iteration paths.

Signs: ``Qc_kW`` is the heat removed in the condenser and ``Qr_kW`` the heat added in the reboiler, both positive
(DWSIM prints the reboiler duty as a negative number).

Flows: ``L[j]`` is the liquid leaving stage j (for the condenser, the reflux), ``V[j]`` the vapor leaving stage j
(0 for a total condenser), both in mol/s, as the DWSIM profile tables. The feed flow is read from ``Stream.fn``
(kmol/s) and ``bottoms_flow`` is given in mol/s; the product streams are returned in kmol/s like any ``Stream``.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from ..stage_thermo import StageThermo


@dataclass
class ColumnResult:
    names: list
    converged: bool
    iterations: int
    residual: float                        # max |scaled residual| at the solution
    T: np.ndarray                          # [K] per stage
    P: np.ndarray                          # [bar]
    L: np.ndarray                          # [mol/s] liquid leaving the stage (reflux for the condenser)
    V: np.ndarray                          # [mol/s] vapor leaving the stage
    x: np.ndarray                          # liquid mole fractions (N x C)
    y: np.ndarray                          # vapor mole fractions (N x C)
    D: float                               # distillate [mol/s]
    B: float                               # bottoms [mol/s]
    reflux_ratio: float
    Qc_kW: float
    Qr_kW: float
    distillate: object = None              # Stream (saturated liquid at the condenser)
    bottoms: object = None                 # Stream (liquid leaving the reboiler)
    energy_balance_error_kW: float = np.nan
    notes: list = field(default_factory=list)

    def profile(self):
        """pandas DataFrame with the profiles, one row per stage."""
        import pandas as pd
        cols = {'T_K': self.T, 'P_bar': self.P, 'L_mol_s': self.L, 'V_mol_s': self.V}
        for i, n in enumerate(self.names):
            cols[f'x_{n}'] = self.x[:, i]
        for i, n in enumerate(self.names):
            cols[f'y_{n}'] = self.y[:, i]
        return pd.DataFrame(cols, index=np.arange(1, len(self.T) + 1))


class RigorousColumn:
    _SOLVERS = ('naphtali_sandholm',)

    def __init__(self, n_stages: int, feed_stage: int, P_top: float, *, reflux_ratio: float | None = None,
                 bottoms_flow: float | None = None, pressure_drop: float = 0.0, condenser: str = 'total',
                 efficiency: float = 1.0, solver: str = 'naphtali_sandholm', tol: float = 1e-10,
                 max_iter: int = 200, h_ig_correction=None, initial_guess: dict | None = None):
        if solver not in self._SOLVERS:
            raise NotImplementedError(f"solver {solver!r}: available {self._SOLVERS}")
        if condenser != 'total':
            raise NotImplementedError("only the total condenser is validated")
        if pressure_drop != 0.0:
            raise NotImplementedError("pressure drop is not validated yet (uniform pressure only)")
        if efficiency != 1.0:
            raise NotImplementedError("stage efficiencies other than 1 are not validated yet")
        if reflux_ratio is None or bottoms_flow is None:
            raise NotImplementedError("phase 1 needs reflux_ratio and bottoms_flow; other specifications are not "
                                      "implemented yet")
        if n_stages < 4:
            raise ValueError("n_stages must be at least 4 (condenser, two trays, reboiler)")
        if not 2 <= feed_stage <= n_stages - 1:
            raise ValueError("feed_stage must be between 2 and n_stages - 1 (stage 1 is the condenser)")
        if reflux_ratio <= 0.0 or bottoms_flow <= 0.0:
            raise ValueError("reflux_ratio and bottoms_flow must be positive")
        self.N = int(n_stages)
        self.feed_stage = int(feed_stage)
        self.P_top = float(P_top)
        self.R = float(reflux_ratio)
        self.B_spec = float(bottoms_flow)
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.h_ig_correction = h_ig_correction
        self.initial_guess = initial_guess
        self.phi = 1.0 / (1.0 + self.R)          # distillate fraction of the condensate

    # ------------------------------------------------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------------------------------------------------
    def _prepare(self, feed):
        self.th = StageThermo(feed, h_ig_correction=self.h_ig_correction)
        self.C = self.th.n
        self.P = np.full(self.N, self.P_top)
        self.F = float(feed.fn) * 1000.0          # feed flow [mol/s] (Stream.fn is kmol/s)
        if not self.B_spec < self.F:
            raise ValueError("bottoms_flow must be smaller than the feed flow")
        self.f = np.zeros((self.N, self.C))
        self.fH = np.zeros(self.N)               # feed enthalpy flow [W]
        jf = self.feed_stage - 1
        self.jf = jf
        self.f[jf] = self.F * np.asarray(feed.z, float)
        self.fH[jf] = self.F * self._feed_enthalpy(feed)
        self.feed = feed
        self.hs = 3.0e4                          # energy scale [J/mol] (order of a latent heat)

    def _feed_enthalpy(self, feed) -> float:
        """Molar enthalpy of the feed [J/mol] through the same functions as the stages (so that an ideal-gas
        correction applies to the feed too). Equal to ``feed.props.H_total_mix`` without a correction."""
        p, th = feed.props, self.th
        T, P = float(feed.T), float(feed.P)
        z = np.asarray(feed.z, float)
        if str(p.phase) == 'liquid':
            h = th.h_liquid(T, P, z)
        elif str(p.phase) == 'vapor':
            h = th.h_vapor(T, P, z)
        else:
            beta = float(p.V_frac)
            h = beta * th.h_vapor(T, P, np.asarray(p.yi, float)) + (1.0 - beta) * th.h_liquid(
                T, P, np.asarray(p.xi, float))
        if self.h_ig_correction is None and abs(h - float(p.H_total_mix)) > 1e-6 * max(1.0, abs(h)):
            warnings.warn(f"feed enthalpy from the stage functions ({h:.3f}) differs from the stream value "
                          f"({float(p.H_total_mix):.3f}) J/mol", RuntimeWarning)
        return h

    # ------------------------------------------------------------------------------------------------------------
    # unknowns: block j = [ln l_i (C), ln w_i (C), T]; w = vapor flows, or the vapor fractions of the condenser
    # ------------------------------------------------------------------------------------------------------------
    def _split(self, u):
        N, C = self.N, self.C
        U = u.reshape(N, 2 * C + 1)
        return U[:, :C], U[:, C:2 * C], U[:, 2 * C]

    def _stage_thermo(self, j, lnl, lnw, T):
        """(ln K, h_L, h_V) of stage j for the block values."""
        l = np.exp(lnl)
        x = l / l.sum()
        w = np.exp(lnw)
        y = w / w.sum()
        lnK = self.th.ln_K(T, self.P[j], x, y)
        hL = self.th.h_liquid(T, self.P[j], x)
        hV = self.th.h_vapor(T, self.P[j], y) if j > 0 else 0.0
        return lnK, hL, hV

    def _thermo_all(self, u):
        lnl, lnw, T = self._split(u)
        lnK = np.zeros((self.N, self.C))
        hL = np.zeros(self.N)
        hV = np.zeros(self.N)
        for j in range(self.N):
            lnK[j], hL[j], hV[j] = self._stage_thermo(j, lnl[j], lnw[j], T[j])
        return lnK, hL, hV

    def _residual(self, u, th):
        N, C = self.N, self.C
        lnl, lnw, T = self._split(u)
        lnK, hL, hV = th
        l = np.exp(lnl)
        L = l.sum(axis=1)
        w = np.exp(lnw)
        v = w.copy()
        v[0] = 0.0
        V = v.sum(axis=1)
        F = np.zeros((N, 2 * C + 1))
        Fs, Es = self.F, self.F * self.hs
        # condenser (stage 0): all the vapor of stage 1 is condensed; incipient vapor in equilibrium; sum y = 1
        F[0, :C] = (v[1] - l[0]) / Fs
        F[0, C:2 * C] = lnw[0] - (lnl[0] - np.log(L[0])) - lnK[0]
        F[0, 2 * C] = w[0].sum() - 1.0
        for j in range(1, N):
            lin = (1.0 - self.phi) * l[0] if j == 1 else l[j - 1]
            vin = v[j + 1] if j < N - 1 else 0.0
            F[j, :C] = (lin + vin + self.f[j] - l[j] - v[j]) / Fs
            F[j, C:2 * C] = (lnw[j] - np.log(V[j])) - (lnl[j] - np.log(L[j])) - lnK[j]
            if j < N - 1:
                Hin_l = (1.0 - self.phi) * L[0] * hL[0] if j == 1 else L[j - 1] * hL[j - 1]
                Hin_v = V[j + 1] * hV[j + 1]
                F[j, 2 * C] = (Hin_l + Hin_v + self.fH[j] - L[j] * hL[j] - V[j] * hV[j]) / Es
            else:
                F[j, 2 * C] = (L[j] - self.B_spec) / Fs
        return F.ravel()

    def _jacobian(self, u, th, F0):
        """Finite-difference Jacobian with stage coloring: stage j only couples to j-1, j, j+1."""
        N, C = self.N, self.C
        nb = 2 * C + 1
        J = np.zeros((N * nb, N * nb))
        lnK0, hL0, hV0 = th
        for k in range(nb):
            dk = 1e-6 if k < 2 * C else 1e-5
            for color in range(3):
                up = u.copy().reshape(N, nb)
                stages = list(range(color, N, 3))
                for j in stages:
                    up[j, k] += dk
                lnK, hL, hV = lnK0.copy(), hL0.copy(), hV0.copy()
                lnl, lnw, T = self._split(up.ravel())
                for j in stages:
                    lnK[j], hL[j], hV[j] = self._stage_thermo(j, lnl[j], lnw[j], T[j])
                Fp = self._residual(up.ravel(), (lnK, hL, hV))
                dF = (Fp - F0) / dk
                for j in stages:
                    r0, r1 = max(j - 1, 0), min(j + 2, N)
                    J[r0 * nb:r1 * nb, j * nb + k] = dF[r0 * nb:r1 * nb]
        return J

    # ------------------------------------------------------------------------------------------------------------
    # initial estimate
    # ------------------------------------------------------------------------------------------------------------
    def _initial(self):
        N, C, F = self.N, self.C, self.F
        z = self.f.sum(axis=0) / F
        Dm = F - self.B_spec
        # sharp split by volatility at the feed (lightest components to the distillate)
        zs = np.clip(z, 1e-12, None)
        K = np.exp(self.th.ln_K(self.feed.T, self.P_top, zs / zs.sum(), zs / zs.sum()))
        order = np.argsort(-K)
        xD = np.full(C, 1e-3 / max(C - 1, 1))
        left = Dm
        for i in order:
            take = min(F * z[i], left)
            xD[i] = max(take / Dm, 1e-3 / max(C, 1))
            left -= take
        xD /= xD.sum()
        xB = np.clip((F * z - Dm * xD) / self.B_spec, 1e-3 / max(C, 1), None)
        xB /= xB.sum()
        q = 1.0 - float(self.feed.V_frac)
        R = self.R
        Lr, Vr = R * Dm, (R + 1.0) * Dm
        eng = self.th.eng
        lnl = np.zeros((N, C)); lnw = np.zeros((N, C)); T = np.zeros(N)
        xs = np.zeros((N, C)); ys = np.zeros((N, C))
        for j in range(N):
            a = j / (N - 1)
            xj = np.exp((1 - a) * np.log(xD) + a * np.log(xB))
            xs[j] = xj / xj.sum()
        Ls = np.zeros(N); Vs = np.zeros(N)
        for j in range(N):
            Ls[j] = Lr if j < self.jf else Lr + q * F
            Vs[j] = 0.0 if j == 0 else (Vr if j <= self.jf else Vr - (1 - q) * F)
        Ls[0] = Vr                              # total condensate
        Ls[N - 1] = self.B_spec
        for j in range(N):
            sat = eng.bubble_T(self.P[j], xs[j])
            T[j] = float(sat.T)
            ys[j] = np.asarray(sat.w, float) / np.sum(sat.w)
        for j in range(N):
            lnl[j] = np.log(Ls[j] * xs[j])
            lnw[j] = np.log(ys[j]) if j == 0 else np.log(Vs[j] * ys[j])
        # the condensate has the composition of the vapor of stage 1
        x0 = ys[1]
        lnl[0] = np.log(Vs[1] * x0)
        sat0 = eng.bubble_T(self.P[0], x0)
        T[0] = float(sat0.T)
        lnw[0] = np.log(np.asarray(sat0.w, float) / np.sum(sat0.w))
        U = np.zeros((N, 2 * C + 1))
        U[:, :C], U[:, C:2 * C], U[:, 2 * C] = lnl, lnw, T
        return U.ravel()

    # ------------------------------------------------------------------------------------------------------------
    # solve
    # ------------------------------------------------------------------------------------------------------------
    def solve(self, feed) -> ColumnResult:
        self._prepare(feed)
        N, C = self.N, self.C
        u = self._initial() if self.initial_guess is None else np.asarray(self.initial_guess['u'], float).copy()
        nb = 2 * C + 1
        th = self._thermo_all(u)
        F0 = self._residual(u, th)
        it = 0
        converged = False
        for it in range(1, self.max_iter + 1):
            nF = np.max(np.abs(F0))
            if nF < self.tol:
                converged = True
                it -= 1
                break
            J = self._jacobian(u, th, F0)
            try:
                du = np.linalg.solve(J, -F0)
            except np.linalg.LinAlgError:
                du = np.linalg.lstsq(J, -F0, rcond=None)[0]
            D = du.reshape(N, nb)
            # step limits: logs of flows, temperatures
            s = 1.0
            lim_log = np.max(np.abs(D[:, :2 * C]))
            lim_T = np.max(np.abs(D[:, 2 * C]))
            if lim_log > 2.0:
                s = min(s, 2.0 / lim_log)
            if lim_T > 15.0:
                s = min(s, 15.0 / lim_T)
            f0 = float(F0 @ F0)
            ok = False
            for _ in range(30):
                un = u + s * du
                try:
                    thn = self._thermo_all(un)
                    Fn = self._residual(un, thn)
                    fn = float(Fn @ Fn)
                except Exception:
                    fn = np.inf
                if np.isfinite(fn) and fn < f0 * (1.0 - 1e-4 * s) + 1e-300:
                    ok = True
                    break
                s *= 0.5
            if not ok:
                warnings.warn("column line search failed: the Newton step does not reduce the residual",
                              RuntimeWarning)
                break
            u, th, F0 = un, thn, Fn
        else:
            it = self.max_iter
        nF = float(np.max(np.abs(F0)))
        if not converged and nF < self.tol:
            converged = True
        if not converged:
            warnings.warn(f"column did not converge: max scaled residual {nF:.3e} after {it} iterations",
                          RuntimeWarning)
        return self._result(u, th, converged, it, nF)

    # ------------------------------------------------------------------------------------------------------------
    def _result(self, u, th, converged, iterations, nF):
        N, C = self.N, self.C
        lnl, lnw, T = self._split(u)
        lnK, hL, hV = th
        l = np.exp(lnl)
        L = l.sum(axis=1)
        w = np.exp(lnw)
        v = w.copy(); v[0] = 0.0
        V = v.sum(axis=1)
        x = l / L[:, None]
        y = np.zeros((N, C))
        y[0] = x[0]                              # total condenser: the vapor is the condensate
        y[1:] = v[1:] / V[1:, None]
        D = self.phi * L[0]
        Lout = L.copy()
        Lout[0] = (1.0 - self.phi) * L[0]        # reflux
        Qc = (V[1] * hV[1] - L[0] * hL[0]) / 1000.0
        Qr = (L[N - 1] * hL[N - 1] + V[N - 1] * hV[N - 1] - L[N - 2] * hL[N - 2] - self.fH[N - 1]) / 1000.0
        B = L[N - 1]
        Hf = float(self.fH.sum())
        err = (Hf + Qr * 1000.0 - Qc * 1000.0 - D * hL[0] - B * hL[N - 1]) / 1000.0
        names = self.th.names
        dist = bott = None
        try:
            dist = self.feed.at(T=float(T[0]), P=float(self.P[0]), fn=float(D) / 1000.0,
                                comp_mol={n: float(x[0, i]) for i, n in enumerate(names)}, description='distillate')
            bott = self.feed.at(T=float(T[N - 1]), P=float(self.P[N - 1]), fn=float(B) / 1000.0,
                                comp_mol={n: float(x[N - 1, i]) for i, n in enumerate(names)}, description='bottoms')
        except Exception as exc:                  # pragma: no cover
            warnings.warn(f"product streams could not be built: {exc}", RuntimeWarning)
        return ColumnResult(names=names, converged=converged, iterations=iterations, residual=nF, T=T.copy(),
                            P=self.P.copy(), L=Lout, V=V, x=x, y=y, D=float(D), B=float(B), reflux_ratio=self.R,
                            Qc_kW=float(Qc), Qr_kW=float(Qr), distillate=dist, bottoms=bott,
                            energy_balance_error_kW=float(err))