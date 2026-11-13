"""Phase envelope (P-T) of a mixture of fixed composition, traced with Michelsen's continuation method.

Unknowns X = [ln K_1 ... ln K_n, ln T, ln P]. Along the whole envelope the incipient phase w = K z
(w_i = K_i z_i) is in equilibrium with the main phase z:

    F_i = ln K_i - ln phi_i(T, P, z; main root) + ln phi_i(T, P, w/sum(w); incipient root) = 0
    F_{n+1} = sum_i z_i (K_i - 1) = 0

On the bubble branch the main phase is the liquid root and the incipient phase the vapor root; at the
critical point K_i = 1 and the curve continues smoothly into the dew branch, where the main phase is
the vapor root and the incipient phase the liquid root. One continuous curve therefore gives the
bubble branch, the critical point and the dew branch, including the retrograde region, the
cricondentherm (maximum T) and the cricondenbar (maximum P), where P(T) or T(P) are multivalued and
step-by-step methods in T or in P fail.

Each step is a predictor (tangent) plus Newton corrector with the most rapidly varying variable of X as
specification (arclength-like continuation, Michelsen 1980).

The curve also gives, by refinement on the same equations: the critical point, the cricondentherm and
the cricondenbar, the saturation pressures at a given T and the saturation temperatures at a given P.

`phase_identification_curve` is the curve that separates liquid-like from vapor-like single-phase fluid
for the fixed composition: the vapor-liquid equilibrium line of the mixture treated as one pseudo
component (equal Gibbs energy of the two roots of the cubic) below the pseudo-critical point and the
line PIP = 1 (Venkatarathnam and Oellrich, 2011) above it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq

from .equilibrium import EquilibriumEngine


class PhaseEnvelopeError(RuntimeError):
    """The continuation could not start or advance."""


@dataclass
class EnvelopePoint:
    T: float
    P: float
    branch: str                     # 'bubble' | 'dew' | 'critical'
    lnK: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    R_max: float = float('nan')         # max_i |ln x_i + ln phi_i(main) - ln y_i - ln phi_i(incipient)|
    sum_error: float = float('nan')     # |sum_i y_i - 1| of the incipient phase
    dZ: float = float('nan')            # |Z_vapor - Z_liquid| of the two phases (0 at the critical point)


@dataclass
class PhaseEnvelopeResult:
    T_bubble: np.ndarray
    P_bubble: np.ndarray
    T_dew: np.ndarray
    P_dew: np.ndarray
    critical_point: tuple[float, float] | None
    cricondentherm: tuple[float, float] | None
    cricondenbar: tuple[float, float] | None
    path: list[EnvelopePoint] = field(repr=False, default_factory=list)   # in the order traced
    n_points: int = 0
    status: str = 'ok'

    def report(self) -> str:
        sep = '-' * 60
        f = lambda p: 'n/a' if p is None else f"T = {p[0]:.3f} K, P = {p[1]:.4f} bar"
        return '\n'.join([sep, 'Phase envelope (P-T)', sep,
                          f"bubble points   : {len(self.T_bubble)}",
                          f"dew points      : {len(self.T_dew)}",
                          f"critical point  : {f(self.critical_point)}",
                          f"cricondentherm  : {f(self.cricondentherm)}",
                          f"cricondenbar    : {f(self.cricondenbar)}",
                          f"max residual    : {self.residual_max():.2e} (|R_i|), {self.sum_error_max():.2e} (|sum y - 1|)",
                          f"status          : {self.status}", sep])

    def residual_max(self, k_min: float = 0.0) -> float:
        """Largest equilibrium residual over the traced points with ||ln K|| > k_min."""
        v = [p.R_max for p in self.path if np.linalg.norm(p.lnK) > k_min]
        return float(max(v)) if v else float('nan')

    def sum_error_max(self, k_min: float = 0.0) -> float:
        v = [p.sum_error for p in self.path if np.linalg.norm(p.lnK) > k_min]
        return float(max(v)) if v else float('nan')


class PhaseEnvelope:
    """P-T phase envelope of the composition `z` for a cubic EOS (see the module docstring).

    `eos` is a CubicEOS; `z` defaults to the EOS composition (`eos.zi`). `PhaseEnvelope.from_stream`
    takes both from a Stream. Components with z = 0 are dropped."""

    def __init__(self, eos, z=None, *, engine: EquilibriumEngine | None = None):
        eng = engine or EquilibriumEngine(eos)
        zz = eng._z(z)
        self.engine, self.z, self.active = eng._active(zz)[0], eng._active(zz)[1], eng._active(zz)[2]
        self.eos_full = eos
        self.k = self.engine.k
        self.n = int(self.k.n)
        if self.n < 2:
            raise ValueError("A phase envelope needs at least two components with z > 0.")

    @classmethod
    def from_stream(cls, stream):
        eos = stream._eos
        return cls(eos, np.asarray(eos.zi, float))

    # ------------------------------------------------------------------ equations
    def _F(self, X, flags):
        n = self.n
        K = np.exp(X[:n])
        T, P = float(np.exp(X[n])), float(np.exp(X[n + 1]))
        w = self.z * K
        S = float(w.sum())
        lm = self.k.phi_eos(T, P, self.z, flags[0])[0]
        li = self.k.phi_eos(T, P, w / S, flags[1])[0]
        F = np.empty(n + 1)
        F[:n] = X[:n] - lm + li
        F[n] = S - 1.0
        return F

    def _FJ(self, X, flags, h=2e-6):
        """F and its Jacobian (central differences: near the critical point the Jacobian is nearly
        singular and a one-sided difference is not accurate enough for Newton to converge)."""
        F0 = self._F(X, flags)
        J = np.empty((F0.size, X.size))
        for j in range(X.size):
            Xp, Xm = X.copy(), X.copy()
            Xp[j] += h
            Xm[j] -= h
            J[:, j] = (self._F(Xp, flags) - self._F(Xm, flags)) / (2.0 * h)
        return F0, J

    def _correct(self, X0, ispec, val, flags, tol=5e-9, maxit=30):
        X = X0.copy()
        X[ispec] = val
        n2 = X.size
        for it in range(1, maxit + 1):
            F, J = self._FJ(X, flags)
            A = np.vstack([J, np.eye(n2)[ispec]])
            rhs = np.concatenate([-F, [0.0]])
            try:
                dX = np.linalg.solve(A, rhs)
            except np.linalg.LinAlgError:
                return X, False, it
            m = float(np.max(np.abs(dX)))
            if m > 0.7:                               # damping: the variables are logarithms
                dX *= 0.7 / m
            X += dX
            if not np.all(np.isfinite(X)):
                return X, False, it
            if float(np.max(np.abs(dX))) < tol:
                return X, True, it
        return X, False, maxit

    def _tangent(self, X, flags, ispec, sign_ref=None):
        n2 = X.size
        F, J = self._FJ(X, flags)
        for _ in range(2):
            A = np.vstack([J, np.eye(n2)[ispec]])
            rhs = np.zeros(n2)
            rhs[-1] = 1.0
            t = np.linalg.solve(A, rhs)
            t /= np.linalg.norm(t)
            j = int(np.argmax(np.abs(t)))
            if j == ispec:
                break
            ispec = j
        if sign_ref is not None and float(t @ sign_ref) < 0.0:
            t = -t
        return t, ispec

    # ------------------------------------------------------------------ start
    def _start(self, P0, start):
        n = self.n
        sat = self.engine.bubble_T(P0, self.z) if start == 'bubble' else self.engine.dew_T(P0, self.z)
        if sat.status != 'ok' or not np.isfinite(sat.T):
            raise PhaseEnvelopeError(
                f"No {start} point at P = {P0} bar (status '{sat.status}'). Try another starting pressure.")
        K = np.asarray(sat.K, float)
        lnK = np.log(K) if start == 'bubble' else -np.log(K)       # incipient phase over main phase
        X = np.concatenate([lnK, [np.log(sat.T), np.log(P0)]])
        flags = ('l', 'v') if start == 'bubble' else ('v', 'l')
        X, ok, _ = self._correct(X, n + 1, np.log(P0), flags)
        if not ok:
            raise PhaseEnvelopeError("The starting point did not converge.")
        return X, flags

    # ------------------------------------------------------------------ trace
    def _trace_branch(self, P_start, start, max_points, ds_max, T_max, P_max, k_stop):
        """One branch from its starting point at P_start upward until ln K is nearly zero (critical
        point), or until the limits. Returns (list of (X, flags)), status."""
        n = self.n
        X, flags = self._start(P_start, start)
        path = [X.copy()]
        t, ispec = self._tangent(X, flags, n + 1)
        if t[n + 1] < 0:                                   # leave the starting pressure upward
            t = -t
        ds, status, guard = 0.05, 'ok', 0
        while len(path) < max_points:
            guard += 1
            if guard > 20 * max_points:
                status = 'step_limit'
                break
            ispec = int(np.argmax(np.abs(t)))
            Xp = X + ds * t
            Xn, ok, it = self._correct(Xp, ispec, Xp[ispec], flags)
            if (not ok) or it > 12 or np.max(np.abs(Xn - X)) > 1.5:
                ds *= 0.5
                if ds < 1e-8:
                    status = 'step_collapsed'
                    break
                continue
            t, _ = self._tangent(Xn, flags, ispec, sign_ref=t)
            X = Xn
            path.append(X.copy())
            if float(np.linalg.norm(X[:n])) < k_stop:     # close to the critical point
                status = 'critical'
                break
            if np.exp(X[n]) > T_max or np.exp(X[n + 1]) > P_max:
                status = 'limit'
                break
            ds = float(np.clip(ds * 5.0 / max(it, 1), 1e-6, ds_max))
        else:
            status = 'max_points'
        return path, flags, status

    def trace(self, P_start=1.01325, *, max_points=300, ds_max=0.25, T_max=1000.0, P_max=5000.0,
              k_stop=0.04) -> PhaseEnvelopeResult:
        """Trace the bubble branch and the dew branch from their points at `P_start` [bar] upward, each by
        continuation, until ln K is smaller than `k_stop` in norm (close to the critical point); the
        critical point (ln K = 0) is then interpolated from the last points of both branches and closes both
        branches. `max_points` limits each branch."""
        n = self.n
        res, flags_of = {}, {}
        for kind in ('bubble', 'dew'):
            res[kind] = self._trace_branch(P_start, kind, max_points, ds_max, T_max, P_max, k_stop)
        status = 'ok' if all(r[2] == 'critical' for r in res.values()) else \
            ';'.join(f"{k}:{v[2]}" for k, v in res.items() if v[2] != 'critical')
        last = {k: res[k][0][-1] for k in res}
        crit = None
        if all(r[2] == 'critical' for r in res.values()):
            crit = self._critical_from_branches(res['bubble'][0], res['dew'][0])
        pts = []
        for kind in ('bubble', 'dew'):
            for X in res[kind][0]:
                pt = EnvelopePoint(float(np.exp(X[n])), float(np.exp(X[n + 1])), kind, X[:n].copy())
                pt.R_max, pt.sum_error, pt.dZ = self.check_point(pt)
                pts.append(pt)
        T_b = [p.T for p in pts if p.branch == 'bubble']
        P_b = [p.P for p in pts if p.branch == 'bubble']
        T_d = [p.T for p in pts if p.branch == 'dew']
        P_d = [p.P for p in pts if p.branch == 'dew']
        if crit is not None:
            T_b.append(crit[0]); P_b.append(crit[1])
            T_d.append(crit[0]); P_d.append(crit[1])
        cth, cbar = self._extrema(res)
        return PhaseEnvelopeResult(np.array(T_b), np.array(P_b), np.array(T_d), np.array(P_d), crit, cth,
                                   cbar, pts, len(pts), status)

    # ------------------------------------------------------------------ checks
    def check_point(self, p: EnvelopePoint):
        """Independent check of a traced point: the equilibrium residuals are recomputed from ln K with
        the roots asked for by name (main phase: liquid on the bubble branch, vapor on the dew branch;
        incipient phase: the other one), not through the corrector's own equations.

        Returns (R_max, sum_error, dZ): R_i = ln x_i + ln phi_i(x) - ln y_i - ln phi_i(y) with x = z the
        main phase and y = K z / sum(K z) the incipient phase; |sum_i K_i z_i - 1|; and |Z_v - Z_l|,
        which tells the two phases apart (a trivial solution has dZ = 0 and ln K = 0)."""
        w = self.z * np.exp(p.lnK)
        S = float(w.sum())
        y = w / S
        main, inc = ('l', 'v') if p.branch == 'bubble' else ('v', 'l')
        lm, Zm, _ = self.k.phi_eos(p.T, p.P, self.z, main)
        li, Zi, _ = self.k.phi_eos(p.T, p.P, y, inc)
        R = np.log(self.z) + lm - np.log(y) - li
        return float(np.max(np.abs(R))), abs(S - 1.0), abs(Zm - Zi)

    # ------------------------------------------------------------------ critical point
    def _critical_from_branches(self, bubble, dew, m=6, deg=4):
        """Critical point from the last `m` points of each branch. ln K goes through zero at the critical
        point and the curve is smooth there, so ln T and ln P are fitted (polynomial of degree `deg`) as
        functions of the projection u of ln K on the direction of the last bubble point and evaluated at
        u = 0 (the projection changes sign from one branch to the other)."""
        n = self.n
        b, d = np.asarray(bubble)[-m:], np.asarray(dew)[-m:]
        v = b[-1, :n] / np.linalg.norm(b[-1, :n])
        u = np.concatenate([b[:, :n] @ v, d[:, :n] @ v])
        deg = min(deg, u.size - 1)
        out = []
        for j in (n, n + 1):
            y = np.concatenate([b[:, j], d[:, j]])
            out.append(float(np.exp(np.polyval(np.polyfit(u, y, deg), 0.0))))
        return out[0], out[1]

    # ------------------------------------------------------------------ extrema
    def _extrema(self, res):
        n = self.n
        best = {n: None, n + 1: None}
        for kind, (Xs, flags, _) in res.items():
            for i in range(1, len(Xs) - 1):
                for comp in (n, n + 1):
                    d1 = Xs[i][comp] - Xs[i - 1][comp]
                    d2 = Xs[i + 1][comp] - Xs[i][comp]
                    if d1 > 0 > d2:
                        dX = Xs[i + 1] - Xs[i - 1]
                        t, _ = self._tangent(Xs[i - 1], flags, int(np.argmax(np.abs(Xs[i] - Xs[i - 1]))),
                                             sign_ref=Xs[i] - Xs[i - 1])
                        cand = self._refine_extremum(Xs[i - 1], t, flags, float(np.linalg.norm(dX)), comp)
                        if best[comp] is None or cand[comp - n] > best[comp][comp - n]:
                            best[comp] = cand
        return best[n], best[n + 1]

    def _refine_extremum(self, Xa, ta, flags, ds, comp):
        """Zero of the tangent component `comp` (n: ln T, n+1: ln P) between Xa and Xa + ds ta."""
        lo, hi = 0.0, ds
        ispec = int(np.argmax(np.abs(ta)))
        Xbest = Xa
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            Xp = Xa + mid * ta
            Xm, ok, _ = self._correct(Xp, ispec, Xp[ispec], flags)
            if not ok:
                break
            tm, _ = self._tangent(Xm, flags, ispec, sign_ref=ta)
            if tm[comp] * ta[comp] > 0.0:
                lo, Xbest = mid, Xm
            else:
                hi = mid
            if hi - lo < 1e-9:
                break
        return float(np.exp(Xbest[self.n])), float(np.exp(Xbest[self.n + 1]))

    # ------------------------------------------------------------------ phase identification curve
    def _am_bm(self, T):
        aij = self.k.prep(T)
        return float(self.z @ aij @ self.z), float(self.z @ self.k.b)

    def _pressure(self, V, T):
        """Pressure [Pa] of the pseudo-pure fluid (am, bm of the mixture) at molar volume V [m3/mol]."""
        a, b = self._am_bm(T)
        return self.k.R * T / (V - b) - a / ((V + self.k.d1 * b) * (V + self.k.d2 * b))

    def _PIP(self, T, V):
        """Phase identification parameter Pi = V (P_VT / P_T - P_VV / P_V) (Venkatarathnam and Oellrich)."""
        h, hT = V * 1e-5, T * 1e-6
        P = self._pressure
        PV = (P(V + h, T) - P(V - h, T)) / (2 * h)
        PVV = (P(V + h, T) - 2 * P(V, T) + P(V - h, T)) / h ** 2
        PT = (P(V, T + hT) - P(V, T - hT)) / (2 * hT)
        PVT = (P(V + h, T + hT) - P(V - h, T + hT) - P(V + h, T - hT) + P(V - h, T - hT)) / (4 * h * hT)
        return V * (PVT / PT - PVV / PV)

    def phase_identification_point(self, T, *, P_min=1e-3, P_max=1000.0, n_scan=4000):
        """Pressure [bar] of the phase identification curve at T [K]; NaN if there is none.

        Below the pseudo-critical temperature (the cubic has three roots in a pressure window) it is the
        pressure at which the liquid and vapor roots of the mixture, as a pseudo component, have the same
        Gibbs energy; above it, the pressure where Pi = 1."""
        k, z = self.k, self.z
        Ps = np.geomspace(P_min, P_max, n_scan)
        nroots = np.array([k.phi_eos(T, p, z, 'v')[2] for p in Ps])
        idx = np.flatnonzero(nroots == 3)
        if idx.size:
            def dG(p):
                return float(z @ (k.phi_eos(T, p, z, 'l')[0] - k.phi_eos(T, p, z, 'v')[0]))
            grid = np.geomspace(Ps[max(idx[0] - 1, 0)], Ps[min(idx[-1] + 1, n_scan - 1)], 600)
            g = np.array([dG(p) for p in grid])
            ok3 = np.array([k.phi_eos(T, p, z, 'v')[2] == 3 for p in grid])
            s = np.flatnonzero((g[:-1] * g[1:] < 0) & ok3[:-1] & ok3[1:])
            if s.size:
                return float(brentq(dG, grid[s[0]], grid[s[0] + 1], xtol=1e-12, rtol=1e-13))

        def f(p):
            Zv = k.phi_eos(T, p, z, 'v')[1]
            return self._PIP(T, Zv * k.R * T / (p * 1e5)) - 1.0
        grid = np.geomspace(max(P_min, 1.0), P_max, 400)
        g = np.array([f(p) for p in grid])
        s = np.flatnonzero(g[:-1] * g[1:] < 0)
        return float(brentq(f, grid[s[0]], grid[s[0] + 1], xtol=1e-12, rtol=1e-13)) if s.size else float('nan')

    def phase_identification_curve(self, T):
        """Pressures [bar] of the phase identification curve for the temperatures `T` (array, K)."""
        return np.array([self.phase_identification_point(float(t)) for t in np.atleast_1d(T)])