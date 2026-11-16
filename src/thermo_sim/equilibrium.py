"""Phase-equilibrium engine for cubic EOS.

The liquid phase model is pluggable,like the property methods of process simulators:
  * no liquid model: phi-phi with the EOS in both phases ("cubic")
  * with a liquid model: vapor = EOS, liquid = model (gamma-phi). The model only
    has to implement lnphi_liquid(kernel, T, P, x) -> (ln phi_i, Z, n_roots).

How it works
  1. Pure components: Psat(T) from the EOS spinodals; Tsat(P) by inverting Psat.
  2. Mixtures: bubble/dew points by incipient-phase successive substitution, with
     trivial-solution detection. If there is no crossing the result is NaN and
     the status says why.
  3. The phase is not decided by comparing T with Tbub/Tdew. It comes from a
     Michelsen TPD stability test followed by a TP flash.
  4. Components with z = 0 are dropped: the engine works on the active subset and
     expands the results afterwards.
  5. Vectorized: matrix products for the mixing rules, closed-form cubic roots,
     a_ij(T) cached per temperature.

Notes
  * DewPointSolver / BubblePointSolver keep `.solve(P)` and `._ki_wilson` for the Stream.
  * SolverOptions still accepts the fields of the previous solver (they are ignored).
  * `tol` is capped at 1e-10 for the successive substitutions.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq

from .eos import cubic_real_roots, mixing_aij, physical_roots


def _lnphi(Z, A, B, sx, am, bm, b, d1, d2):
    """ln phi_i of a cubic EOS written with delta1, delta2. Single definition: used by the
    kernel (mixtures) and by the pure-component saturation (vectors of length 1).
    sx = a_ij x, am = x a_ij x, bm = x b, b = b_i."""
    bb = b / bm
    return (bb * (Z - 1.0) - np.log(Z - B)
            - A / ((d1 - d2) * B) * (2.0 * sx / am - bb) * np.log((Z + d1 * B) / (Z + d2 * B)))


class _SubEOS:
    """Same interface as CubicEOS, restricted to the components in `mask` (z > 0)."""

    def __init__(self, eos, mask: np.ndarray):
        self._eos = eos
        self.mask = np.asarray(mask, bool)
        m = self.mask
        self.n = int(m.sum())
        self.R = float(eos.R)
        self.bi = np.asarray(eos.bi, float)[m]
        self.kij = np.asarray(eos.kij, float)[np.ix_(m, m)]
        self.Tc = np.asarray(eos.Tc, float)[m]
        self.Pc = np.asarray(eos.Pc, float)[m]
        self.omega = np.asarray(eos.omega, float)[m]
        self.zi = np.asarray(eos.zi, float)[m]
        self.delta1, self.delta2 = float(eos.delta1), float(eos.delta2)
        self.Omega_a, self.Omega_b = float(eos.Omega_a), float(eos.Omega_b)
        self.label = getattr(eos, 'label', None)

    def ai(self, T):
        return np.asarray(self._eos.ai(T), float)[self.mask]

    def sqrt_ai(self, T):
        return np.asarray(self._eos.sqrt_ai(T), float)[self.mask]


class _Kernel:
    """Vectorized EOS calculations (fugacity coefficients, roots)."""

    def __init__(self, eos):
        self.eos = eos
        self.n = int(eos.n)
        self.R = float(eos.R)
        self.b = np.asarray(eos.bi, float)
        self.kij = np.asarray(eos.kij, float)
        assert self.kij.shape == (self.n, self.n), f"kij {self.kij.shape} != n={self.n}"
        self.d1, self.d2 = float(eos.delta1), float(eos.delta2)
        self.Tc = np.asarray(eos.Tc, float)
        self.Pc = np.asarray(eos.Pc, float)
        self.w = np.asarray(eos.omega, float)
        # Critical V/b of the EOS (PR: 3.951, SRK: 3.847). Used to tell liquid from gas.
        Zc = float(np.mean(cubic_real_roots(
            (self.d1 + self.d2 - 1) * eos.Omega_b - 1,
            eos.Omega_a + (self.d1 * self.d2 - self.d1 - self.d2) * eos.Omega_b ** 2
            - (self.d1 + self.d2) * eos.Omega_b,
            -(eos.Omega_a * eos.Omega_b + self.d1 * self.d2 * eos.Omega_b ** 2 * (1 + eos.Omega_b)))))
        self.vc_over_b = Zc / float(eos.Omega_b)
        self._T = None
        self._ai = None
        self._aij = None
        self.liq = None  # liquid phase model (gamma-phi, Henry...). None -> EOS in both phases

    def prep(self, T: float):
        if T != self._T:
            self._ai = np.asarray(self.eos.ai(T), float)
            self._aij = mixing_aij(np.asarray(self.eos.sqrt_ai(T), float), self.kij)
            self._T = T
        return self._aij

    def AB(self, T, P_bar, x):
        aij = self.prep(T)
        sx = aij @ x
        am = float(x @ sx)
        bm = float(x @ self.b)
        RT = self.R * T
        P = P_bar * 1e5
        return am * P / RT ** 2, bm * P / RT, sx, am, bm

    def roots(self, A, B):
        return physical_roots(A, B, self.d1, self.d2)

    def lnphi_Z(self, Z, A, B, sx, am, bm):
        return _lnphi(Z, A, B, sx, am, bm, self.b, self.d1, self.d2)

    def phi_eos(self, T, P_bar, x, phase):
        """(ln phi_i, Z, n_roots) from the EOS only. 'v' -> largest root, 'l' -> smallest."""
        A, B, sx, am, bm = self.AB(T, P_bar, x)
        r = self.roots(A, B)
        Z = r[-1] if phase == "v" else r[0]
        return self.lnphi_Z(Z, A, B, sx, am, bm), float(Z), len(r)

    def phi(self, T, P_bar, x, phase):
        """(ln phi_i, Z, n_roots). The liquid phase uses the plugged-in model if there is one."""
        if phase == "l" and self.liq is not None:
            return self.liq.lnphi_liquid(self, T, P_bar, x)
        return self.phi_eos(T, P_bar, x, phase)

    def stable_root(self, T, P_bar, x):
        """Phase with the lowest Gibbs energy: (ln phi, Z, 'l' | 'v' | None, n_roots)."""
        if self.liq is not None:
            lpl, Zl, nl = self.phi(T, P_bar, x, "l")
            lpv, Zv, nv = self.phi_eos(T, P_bar, x, "v")
            if float(x @ lpl) < float(x @ lpv):
                return lpl, Zl, "l", max(nl, nv, 2)
            return lpv, Zv, "v", max(nl, nv, 2)
        A, B, sx, am, bm = self.AB(T, P_bar, x)
        r = self.roots(A, B)
        cand = [r[0]] if len(r) == 1 else [r[0], r[-1]]
        best = None
        for Z in cand:
            lp = self.lnphi_Z(Z, A, B, sx, am, bm)
            g = float(x @ lp)
            if best is None or g < best[0]:
                best = (g, lp, Z)
        _, lp, Z = best
        tag = "l" if (len(r) > 1 and Z == r[0]) else ("v" if len(r) > 1 else None)
        return lp, float(Z), tag, len(r)


@dataclass
class SatResult:
    T: float = np.nan
    # 'ok' | 'P>=Pc' | 'no_crossing' | 'near_critical' | 'approx_bracket'
    status: str = "not_computed"
    w: np.ndarray = field(default_factory=lambda: np.array([]))   # composition of the incipient phase
    iters: int = 0
    K: np.ndarray = field(default_factory=lambda: np.array([]))   # K_i = y_i/x_i at the saturation point


@dataclass
class FlashResult:
    phase: str                         # 'vapor' | 'liquid' | 'two_phase'
    beta: float                        # vapor mole fraction
    x: np.ndarray
    y: np.ndarray
    K: np.ndarray
    Zl: float = np.nan                 # Z of the liquid phase (if present)
    Zv: float = np.nan                 # Z of the vapor phase (if present)
    converged: bool = True
    iters: int = 0
    above_criticals: bool = False      # T and P above the critical T and P of every component


@dataclass
class EquilibriumResult:               # kept for the Stream
    T: float
    x: np.ndarray
    Ki: np.ndarray
    iters: int
    method: str
    kind: str                          # 'dew' | 'bubble'
    Ki_detail: dict = field(default_factory=dict, repr=False)

    def __repr__(self) -> str:
        return (f"EquilibriumResult(kind='{self.kind}', T={self.T:.4f} K, "
                f"method='{self.method}', iters={self.iters})")


@dataclass(frozen=True)
class StabilityOutcome:
    status: str                        # 'unstable' | 'stable' | 'inconclusive'
    S: float = float("nan")            # sum of trial-phase mole numbers (S > 1: unstable)
    w: np.ndarray | None = None        # composition of the trial phase
    trial: str | None = None           # 'v' | 'l'


@dataclass
class SolverOptions:
    tol: float = 1e-11                 # successive-substitution tolerance (capped at 1e-10)
    n_scan: int = 48                   # T points scanned to locate the bubble/dew crossing
    max_ss: int = 400                  # max successive-substitution iterations
    ztol: float = 1e-14                # z <= ztol counts as an inactive component
    stability_extra_starts: bool = True    # also try each pure component as trial phase (Michelsen)
    # Fields of the previous solver: accepted and ignored.
    max_newton: int = 60
    n_inner: int = 12
    tol_inner: float = 1e-9
    dT_max: float = 20
    margin: float = 20
    max_jump: float = 3


class EquilibriumEngine:
    def __init__(self, eos, options: SolverOptions | None = None, liquid_model=None):
        self.eos = eos
        self.k = _Kernel(eos)
        self.opt = options or SolverOptions()
        self._tol = min(float(self.opt.tol), 1e-10)
        self.liquid_model = liquid_model
        if liquid_model is not None:
            self.k.liq = liquid_model
            if hasattr(liquid_model, "bind"):
                liquid_model.bind(self.k)
        self._sub: dict = {}

    def _z(self, z):
        z = np.asarray(self.eos.zi if z is None else z, float)
        return z / z.sum()

    def _wilson(self, T, P):
        k = self.k
        return np.clip(k.Pc / P * 10.0 ** ((7.0 / 3.0) * (1.0 + k.w) * (1.0 - k.Tc / T)), 1e-12, 1e12)

    def _active(self, z):
        """(engine, active z, mask | None). If some z <= ztol, builds (and caches) a sub-engine."""
        z = self._z(z)
        act = z > self.opt.ztol
        if act.all():
            return self, z, None
        key = act.tobytes()
        sub = self._sub.get(key)
        if sub is None:
            lm = self.liquid_model.subset(act) if self.liquid_model is not None else None
            sub = EquilibriumEngine(_SubEOS(self.eos, act), self.opt, liquid_model=lm)
            self._sub[key] = sub
        zs = z[act]
        return sub, zs / zs.sum(), act

    @staticmethod
    def _expand(v, act, fill=0.0):
        if act is None:
            return v
        out = np.full(act.size, fill, float)
        out[act] = v
        return out

    # pure component

    def _spinodal(self, T, i):
        """(P_min, P_max, u_liq, u_vap) of the isotherm, with u = V/b. None if T >= Tc."""
        k = self.k
        k.prep(T)
        a, b, R = k._ai[i], k.b[i], k.R
        s, p = k.d1 + k.d2, k.d1 * k.d2
        kap = a / (R * T * b)
        poly = [-1.0, 2 * kap - 2 * s, kap * (s - 4) - (s * s + 2 * p),
                kap * (2 - 2 * s) - 2 * s * p, kap * s - p * p]
        r = np.roots(poly)
        u = np.sort(r[(np.abs(r.imag) < 1e-9 * (1 + np.abs(r.real))) & (r.real > 1.0 + 1e-9)].real)
        if u.size < 2:
            return None
        ul, uv = u[0], u[1] if u.size == 2 else u[-1]
        P = lambda uu: R * T / (b * (uu - 1.0)) - a / (b * b * (uu * uu + s * uu + p))
        Pmin, Pmax = P(ul) / 1e5, P(uv) / 1e5
        if not (Pmax > Pmin):
            return None
        return Pmin, Pmax, ul, uv

    def pure_psat(self, T, i=0):
        """EOS saturation pressure [bar] for T < Tc, solving phiL = phiV."""
        k = self.k
        if T >= k.Tc[i] * (1 - 1e-12):
            return np.nan
        sp = self._spinodal(T, i)
        if sp is None:
            return np.nan
        Pmin, Pmax, ul, uv = sp
        a, b, R = k._ai[i], k.b[i], k.R
        RT = R * T
        umid = 0.5 * (ul + uv)
        d1, d2 = k.d1, k.d2

        ai, bi = np.array([a]), np.array([b])

        def lnphi(Z, A, B):
            return float(_lnphi(Z, A, B, ai, a, b, bi, d1, d2)[0])

        def f(lp):
            P = np.exp(lp)
            A, B = a * P * 1e5 / RT ** 2, b * P * 1e5 / RT
            r = k.roots(A, B)
            if len(r) >= 3:
                return lnphi(r[0], A, B) - lnphi(r[-1], A, B)
            return -1.0 if r[0] / B < umid else 1.0   # liquid only -> phiL < phiV; vapor only -> phiL > phiV

        hi = np.log(Pmax * (1 - 1e-10))
        lo_P = max(Pmin * (1 + 1e-10), Pmax * 1e-7) if Pmin > 0 else Pmax * 1e-7
        lo = np.log(lo_P)
        n = 0
        while f(lo) <= 0 and n < 6:                   # lower P until the vapor is stable
            lo -= 2.0 * np.log(10.0)
            n += 1
        if f(lo) <= 0 or f(hi) >= 0:
            return np.nan
        return float(np.exp(brentq(f, lo, hi, xtol=1e-13, rtol=1e-13, maxiter=200)))

    def pure_Tsat(self, P_bar, i=0) -> SatResult:
        k = self.k
        Tc, Pc, w = k.Tc[i], k.Pc[i], k.w[i]
        one = np.array([1.0])
        if P_bar >= Pc * (1 - 1e-9):
            return SatResult(np.nan, "P>=Pc", K=one)          # no saturation
        den = 1.0 - (3.0 / 7.0) * np.log(P_bar / Pc) / (1.0 + w)
        Tw = Tc / den if den > 0 else 0.6 * Tc
        Tw = float(np.clip(Tw, 0.25 * Tc, 0.999 * Tc))
        Tmax = Tc * (1 - 1e-8)

        def g(T):
            ps = self.pure_psat(T, i)
            return np.log(ps / P_bar) if np.isfinite(ps) and ps > 0 else np.nan

        T_near = Tc * (1 - 1e-4)            # above this the spinodal quartic loses precision
        ps_near = self.pure_psat(T_near, i)
        if np.isfinite(ps_near) and P_bar > ps_near:           # near-critical stretch: interpolate in ln P
            fr = np.log(P_bar / ps_near) / np.log(Pc / ps_near)
            return SatResult(float(T_near + (Tc - T_near) * fr), "near_critical", one, K=one)
        Tmax = T_near
        lo, hi = Tw * 0.93, min(Tw * 1.07, Tmax)
        for _ in range(40):
            gl = g(lo)
            if np.isfinite(gl) and gl < 0:
                break
            lo *= 0.93
        for _ in range(40):
            gh = g(hi)
            if np.isfinite(gh) and gh > 0:
                break
            if hi >= Tmax:
                break
            hi = min(hi * 1.03, Tmax)
        gl, gh = g(lo), g(hi)
        if not (np.isfinite(gl) and np.isfinite(gh) and gl < 0 < gh):
            return SatResult(np.nan, "no_crossing", K=one)
        T = brentq(g, lo, hi, xtol=1e-10, rtol=1e-13, maxiter=200)
        return SatResult(float(T), "ok", one, K=one)

    def phase_pure(self, T, P_bar, i=0) -> str:
        k = self.k
        Tc, Pc = k.Tc[i], k.Pc[i]
        if T >= Tc:
            return "vapor"                             # no liquid above Tc (see above_component_criticals)
        if P_bar >= Pc:
            return "liquid"                            # compressed liquid
        ps = self.pure_psat(T, i)
        return "vapor" if P_bar < ps else "liquid"

    # mixtures: incipient phase (bubble / dew). z holds the active components only.

    def _incipient(self, T, P, z, kind, w0=None):
        k = self.k
        lnb = k.phi(T, P, z, "l" if kind == "bubble" else "v")[0]
        if w0 is None:
            K = self._wilson(T, P)
            W = z * K if kind == "bubble" else z / K
        else:
            W = w0.copy()
        w = np.maximum(W / W.sum(), 1e-300)
        err, S, lnK = np.inf, 1.0, np.zeros_like(z)
        for it in range(self.opt.max_ss):
            if kind == "bubble":
                lnt = k.phi(T, P, w, "v")[0]
                lnK = lnb - lnt
                Wn = z * np.exp(lnK)
            else:
                lnt = k.phi(T, P, w, "l")[0]
                lnK = lnt - lnb
                Wn = z * np.exp(-lnK)
            S = float(Wn.sum())
            wn = np.maximum(Wn / S, 1e-300)
            err = float(np.max(np.abs(np.log(wn / w))))
            w = wn
            if err < self._tol:
                break
        trivial = bool(np.max(np.abs(np.log(w / np.maximum(z, 1e-300))[z > 1e-12])) < 1e-4)
        return S, w, err < 1e-8, trivial, it + 1

    def _sat_T(self, P, z, kind) -> SatResult:
        k = self.k
        Tmax = float(k.Tc[z > 0].max())
        Tmin = float(k.Tc[z > 0].min())
        Ts = np.geomspace(0.20 * Tmin, 1.15 * Tmax, max(self.opt.n_scan, 64))
        g = np.full(Ts.size, np.nan)
        W = {}
        w_prev = None
        for j, T in enumerate(Ts):
            S, w, conv, triv, _ = self._incipient(T, P, z, kind, w0=w_prev)
            if conv and not triv and S > 0:
                g[j] = np.log(S)
                w_prev = w
                W[j] = w
            else:
                w_prev = None
        sgn = 1.0 if kind == "bubble" else -1.0       # bubble: g grows with T; dew: g decreases
        cr = [j for j in range(Ts.size - 1)
              if np.isfinite(g[j]) and np.isfinite(g[j + 1]) and sgn * g[j] < 0 <= sgn * g[j + 1]]
        if not cr:
            return SatResult(np.nan, "no_crossing")
        j = cr[0] if kind == "bubble" else cr[-1]     # bubble: coldest crossing; dew: hottest
        cache = {"w": W.get(j)}

        def h(T):
            S, w, conv, triv, _ = self._incipient(T, P, z, kind, w0=cache["w"])
            if not conv or triv:
                raise ValueError("trivial or non-converged solution")
            cache["w"] = w
            return np.log(S)

        try:
            T = brentq(h, Ts[j], Ts[j + 1], xtol=1e-9, rtol=1e-13, maxiter=100)
            S, w, conv, triv, it = self._incipient(T, P, z, kind, w0=cache["w"])
            K = w / z if kind == "bubble" else z / w
            return SatResult(float(T), "ok", w, it, K)
        except ValueError:
            return SatResult(float(0.5 * (Ts[j] + Ts[j + 1])), "approx_bracket", W.get(j, np.array([])))

    def _sat(self, P, z, kind) -> SatResult:
        if self.k.n == 1:
            if self.liquid_model is not None:
                r = self._pure_Tsat_liquid_model(P)
                if r is not None:
                    return r
            return self.pure_Tsat(P, 0)
        return self._sat_T(P, z, kind)

    def _pure_Tsat_liquid_model(self, P):
        """Saturation temperature of a pure component with the gamma-phi liquid model: ln(phiL/phiV) = 0,
        where phiL is the effective liquid fugacity coefficient (Psat correlation) and phiV the EOS vapor one.
        Returns None if no crossing is found (the caller falls back to the EOS saturation)."""
        k = self.k
        one = np.array([1.0])

        def g(T):
            try:
                v = float(k.phi(T, P, one, "l")[0][0] - k.phi(T, P, one, "v")[0][0])
            except Exception:
                return np.nan
            return v if np.isfinite(v) else np.nan

        Tc = float(k.Tc[0])
        Ts = np.linspace(0.35 * Tc, 0.98 * Tc, 24)
        gs = [g(T) for T in Ts]
        for j in range(len(Ts) - 1):
            if np.isfinite(gs[j]) and np.isfinite(gs[j + 1]) and gs[j] < 0 <= gs[j + 1]:
                T = brentq(g, Ts[j], Ts[j + 1], xtol=1e-10, rtol=1e-13, maxiter=200)
                return SatResult(float(T), "ok", one, K=one)
        return None

    def _sat_pub(self, P, z, kind) -> SatResult:
        eng, zs, act = self._active(z)
        r = eng._sat(P, zs, kind)
        if act is None:
            return r
        w = self._expand(r.w, act) if r.w.size else r.w
        K = self._expand(r.K, act, fill=1.0) if r.K.size else r.K
        return SatResult(r.T, r.status, w, r.iters, K)

    def bubble_T(self, P, z=None) -> SatResult:
        return self._sat_pub(P, z, "bubble")

    def dew_T(self, P, z=None) -> SatResult:
        return self._sat_pub(P, z, "dew")

    # TP flash with Michelsen stability test

    @staticmethod
    def _rr(z, K):
        Kmax, Kmin = float(K.max()), float(K.min())
        if Kmax <= 1.0 or Kmin >= 1.0:
            return None
        lo, hi = 1.0 / (1.0 - Kmax), 1.0 / (1.0 - Kmin)
        eps = 1e-12 * (hi - lo)
        f = lambda b: float(np.sum(z * (K - 1.0) / (1.0 + b * (K - 1.0))))
        return float(brentq(f, lo + eps, hi - eps, xtol=1e-14, rtol=1e-13, maxiter=200))

    def _above_criticals(self, T, P) -> bool:
        """T and P above the critical T and P of every (active) component. This is NOT the
        critical point of the mixture, which can lie elsewhere."""
        return bool(T > self.k.Tc.max() and P > self.k.Pc.max())

    def above_component_criticals(self, T, P, z=None) -> bool:
        eng, zs, act = self._active(z)
        return eng._above_criticals(T, P)

    def _classify_single(self, T, P, z):
        """(label, Z) of a single phase. With a liquid model it goes by Gibbs energy, with the EOS by volume."""
        k = self.k
        if self._above_criticals(T, P):
            lp, Z, tag, nr = k.stable_root(T, P, z)
            return "vapor", Z
        lp, Z, tag, nr = k.stable_root(T, P, z)
        if k.liq is not None:
            return ("liquid" if tag == "l" else "vapor"), Z
        _, B, _, _, _ = k.AB(T, P, z)
        return ("liquid" if (Z / B) < k.vc_over_b else "vapor"), Z

    def classify_single(self, T, P, z=None) -> str:
        eng, zs, act = self._active(z)
        return eng._classify_single(T, P, zs)[0]

    def _stability(self, T, P, z) -> StabilityOutcome:
        """Michelsen TPD test. status: 'unstable' | 'stable' | 'inconclusive'.
        'inconclusive': no converged unstable minimum was found and at least one trial did not
        converge, so stability cannot be claimed."""
        k = self.k
        lnz, Zref, tag, nr = k.stable_root(T, P, z)
        d = np.log(z) + lnz
        Kw = self._wilson(T, P)
        starts = [("v", z * Kw, False), ("l", z / Kw, False)]
        if self.opt.stability_extra_starts:
            for i in range(z.size):
                e = np.full(z.size, 1e-10)
                e[i] = 1.0
                starts += [("v", e, True), ("l", e.copy(), True)]
        best, unconverged = None, False
        for trial, W0, extra in starts:
            W = W0.copy()
            converged = False
            for _ in range(self.opt.max_ss):
                w = W / W.sum()
                lnt = k.phi(T, P, w, trial)[0]
                lnW = d - lnt
                err = float(np.max(np.abs(lnW - np.log(W))))
                W = np.exp(lnW)
                if err < self._tol:
                    converged = True
                    break
            if not converged:
                if not extra:          # pure-component starts are best effort: a slow drift to
                    unconverged = True  # the trivial solution is not a reason to doubt stability
                continue
            S = float(W.sum())
            w = W / S
            triv = float(np.max(np.abs(np.log(w / z)))) < 1e-3
            if (not triv) and S > 1.0 + 1e-8 and (best is None or S > best[0]):
                best = (S, w, trial)
        if best is not None:
            S, w, trial = best
            if k.liq is None:
                # The label of the trial phase is only the starting guess: a pure-component start
                # can converge to the same stationary point with either label. Orient by density:
                # a trial phase denser than the feed (smaller Z) is the liquid-like one.
                Zw = k.phi(T, P, w, trial)[1]
                trial = "l" if Zw < Zref else "v"
            return StabilityOutcome("unstable", S, w, trial)
        return StabilityOutcome("inconclusive" if unconverged else "stable")

    def _two_phase(self, T, P, z, K):
        k = self.k
        beta, x, y = 0.5, z, z
        for it in range(500):
            beta = self._rr(z, K)
            if beta is None:
                return None
            x = z / (1.0 + beta * (K - 1.0))
            y = K * x
            x, y = x / x.sum(), y / y.sum()
            lnl, Zl, _ = k.phi(T, P, x, "l")
            lnv, Zv, _ = k.phi(T, P, y, "v")
            lnKn = lnl - lnv
            err = float(np.max(np.abs(lnKn - np.log(K))))
            K = np.exp(lnKn)
            if err < self._tol:
                break
        if k.liq is None and Zl > Zv * (1.0 + 1e-6):
            # Mirrored labelling: an inverted starting K converges to the same split with the
            # phases swapped (the "liquid" is the less dense one). A phase with a single EOS root
            # returns that root for both 'l' and 'v', so the fugacity equations cannot tell them
            # apart. Relabel: liquid = denser phase (smaller Z at the same T and P).
            beta, x, y, K, Zl, Zv = 1.0 - beta, y, x, 1.0 / K, Zv, Zl
        trivial = float(np.max(np.abs(np.log(K)))) < 1e-5
        return beta, x, y, K, Zl, Zv, err < 1e-8 and not trivial, it + 1

    def _flash_TP(self, T, P, z) -> FlashResult:
        k = self.k
        n = k.n
        zeros = np.zeros(n)
        if n == 1:
            ph = self.phase_pure(T, P, 0)
            b = 0.0 if ph == "liquid" else 1.0
            Z = k.phi_eos(T, P, z, "l" if b == 0.0 else "v")[1]
            return FlashResult(ph, b, z.copy() if b == 0 else zeros, z.copy() if b == 1 else zeros,
                               np.ones(n), Zl=Z if b == 0 else np.nan, Zv=Z if b == 1 else np.nan)
        st = self._stability(T, P, z)
        K0 = None
        if st.status == "unstable":
            K0 = st.w / z if st.trial == "v" else z / st.w
        elif st.status == "inconclusive":
            K0 = self._wilson(T, P)           # stability not proven: let the verified flash decide
        if K0 is not None:
            tp = self._two_phase(T, P, z, np.clip(K0, 1e-12, 1e12))
            if tp is not None:
                beta, x, y, K, Zl, Zv, conv, it = tp
                if 0.0 < beta < 1.0 and (conv or st.status == "unstable"):
                    return FlashResult("two_phase", beta, x, y, K, Zl, Zv, conv, it)
        if st.status == "inconclusive":
            warnings.warn(f"Stability test did not converge at T={T:.2f} K, P={P:.3f} bar; "
                          "the single-phase result is not guaranteed", RuntimeWarning, stacklevel=2)
        ph, Z = self._classify_single(T, P, z)
        if ph == "liquid":
            return FlashResult(ph, 0.0, z.copy(), zeros, np.ones(n), Zl=Z)
        return FlashResult(ph, 1.0, zeros, z.copy(), np.ones(n), Zv=Z)

    def _expand_flash(self, fl: FlashResult, T, P, act) -> FlashResult:
        if act is None or fl is None:
            return fl
        K = self._wilson(T, P)
        K[act] = fl.K
        return FlashResult(fl.phase, fl.beta, self._expand(fl.x, act), self._expand(fl.y, act), K,
                           fl.Zl, fl.Zv, fl.converged, fl.iters)

    def flash_TP(self, T, P, z=None) -> FlashResult:
        eng, zs, act = self._active(z)
        fl = self._expand_flash(eng._flash_TP(T, P, zs), T, P, act)
        fl.above_criticals = eng._above_criticals(T, P)
        return fl

    # VP flash: find T such that the vapor fraction is VF at pressure P

    def _flash_VP(self, P, VF, z):
        k = self.k
        n = k.n
        zeros = np.zeros(n)
        ones = np.ones(n)
        if n == 1:
            r = self.pure_Tsat(P, 0)
            if not np.isfinite(r.T):
                return np.nan, None
            if VF >= 1.0:
                return r.T, FlashResult("vapor", 1.0, zeros, z.copy(), ones)
            if VF <= 0.0:
                return r.T, FlashResult("liquid", 0.0, z.copy(), zeros, ones)
            return r.T, FlashResult("two_phase", VF, z.copy(), z.copy(), ones)
        td = self._sat(P, z, "dew")
        tb = self._sat(P, z, "bubble")
        if VF >= 1.0:
            return td.T, (FlashResult("vapor", 1.0, zeros, z.copy(), ones) if np.isfinite(td.T) else None)
        if VF <= 0.0:
            return tb.T, (FlashResult("liquid", 0.0, z.copy(), zeros, ones) if np.isfinite(tb.T) else None)
        if not np.isfinite(td.T):
            return np.nan, None

        def beta(T):
            try:
                return self._flash_TP(T, P, z).beta
            except Exception:
                return np.nan

        hi = td.T
        lo = tb.T if np.isfinite(tb.T) else hi * 0.9
        for _ in range(60):
            b = beta(lo)
            if np.isfinite(b) and b < VF:
                break
            lo *= 0.95
        else:
            return np.nan, None                      # beta never drops below VF (e.g. nearly non-condensable mixture)
        h = lambda T: beta(T) - VF
        try:
            T = float(brentq(h, lo, hi, xtol=1e-9, rtol=1e-13, maxiter=100))
        except ValueError:
            return np.nan, None
        return T, self._flash_TP(T, P, z)

    def flash_VP(self, P, VF, z=None):
        """T such that the vapor fraction is VF at pressure P. Returns (T, FlashResult | None)."""
        eng, zs, act = self._active(z)
        T, fl = eng._flash_VP(P, VF, zs)
        return T, (self._expand_flash(fl, T, P, act) if np.isfinite(T) else None)


class _CompatSolver:
    """Legacy solver interface (`solve`, `_ki_wilson`) on top of EquilibriumEngine."""
    _kind = "dew"
    _OK = ("ok", "near_critical")

    def __init__(self, eos, *, k_calc=None, options=None, engine: EquilibriumEngine | None = None):
        """k_calc: liquid phase model (GammaPhiModel), or None for phi-phi with the EOS."""
        self.eos = eos
        self.k_calc = k_calc
        self.engine = engine if engine is not None else EquilibriumEngine(eos, options, liquid_model=k_calc)

    def _ki_wilson(self, T, P_bar):
        return self.engine._wilson(T, P_bar)

    def solve(self, P_bar):
        e = self.engine
        r = e.dew_T(P_bar) if self._kind == "dew" else e.bubble_T(P_bar)
        if not np.isfinite(r.T) or r.status not in self._OK:
            raise ValueError(f"No {self._kind} temperature exists at P={P_bar} bar ({r.status}).")
        Ki = r.K if r.K.size else np.ones(self.eos.n)
        return EquilibriumResult(T=r.T, x=r.w, Ki=Ki, iters=r.iters,
                                 method="incipient", kind=self._kind,
                                 Ki_detail={"status": r.status})


class DewPointSolver(_CompatSolver):
    _kind = "dew"


class BubblePointSolver(_CompatSolver):
    _kind = "bubble"