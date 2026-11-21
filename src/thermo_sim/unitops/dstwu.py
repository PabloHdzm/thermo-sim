"""Shortcut distillation column (Fenske-Underwood-Gilliland), DWSIM "Shortcut Column" / DSTWU.

Given the feed, the separation of two key components and either the reflux ratio or the number of stages, the
column returns the minimum reflux ratio, the minimum and actual number of stages, the feed stage, the flows inside
the column, the two product streams and the condenser and reboiler duties.

Algorithm (validated against DWSIM 10.2.12.0 on benzene/toluene, reference_data/dwsim/dstwu_benzene_toluene.json):

1. Product split from the key specification. Components lighter than the light key go entirely to the distillate and
   components heavier than the heavy key entirely to the bottoms (assumption: the keys must be adjacent in
   volatility). The keys are fixed either by ``x_LK_bottoms`` and ``x_HK_distillate`` (mole fractions, DWSIM inputs)
   or by ``rec_LK`` and ``rec_HK`` (fractions of the feed light and heavy key that go to the distillate).
2. Volatilities alpha_i = K_i / K_HK. DWSIM evaluates them once, at the state of the feed stream, and uses them in
   both Fenske and Underwood (``alpha_mode='feed'``, reproduces Nmin and Rmin of DWSIM to 5 digits). For a
   single-phase feed K = phi_L / phi_V is taken with the incipient phase of the stability trial (see ``_K_feed``). The alternatives
   are the arithmetic or geometric mean of the values at the top and bottom product bubble points (the Fenske
   convention of the first thermo-sim version); in that case the same mean is used in Underwood.
3. Minimum stages: Fenske (``nmin_method='fenske'``) or Winn (``'winn'``, uses K at top and bottom).
4. Minimum reflux: Underwood, root theta between alpha_HK = 1 and alpha_LK of sum(alpha z / (alpha - theta)) = 1 - q,
   Rmin + 1 = sum(alpha x_D / (alpha - theta)). q = 1 - vapor fraction of the feed (DWSIM, ``q_mode='dwsim'``) or
   the thermal condition from enthalpies (``q_mode='enthalpy'``).
5. Gilliland: X = (R - Rmin) / (R + 1), Y = (N - Nmin) / (N + 1). DWSIM matches the Eduljee form
   Y = 0.75 (1 - X^0.5668) (``gilliland='eduljee'``); ``'molokanov'`` is also available. With the reflux ratio given
   the number of stages is computed, with the number of stages given the reflux ratio is computed.
6. Feed stage (default ``'fenske'``): the stages are split in proportion to the minimum stages of each section with one
   alpha, n_R = ln[(x_D,LK / x_D,HK)(z_HK / z_LK)] above the feed and n_S = ln[(z_LK / z_HK)(x_B,HK / x_B,LK)] below
   it. DWSIM's "optimum feed stage" equals N n_S / (n_R + n_S), that is N_stripping (the stage counted from the
   bottom). It reproduces three feed compositions (50/50, 70/30, 90/10 benzene/toluene); with a feed of 50/50 and a
   symmetric split it is N/2, which is why one case could not tell it from Kirkbride. ``feed_stage`` of the result is
   that number; ``N_rectifying`` is the count of stages above the feed.
7. Internal flows: L = R D, V = (R + 1) D, L' = L + q F, V' = V - (1 - q) F.
8. Duties: condenser Qc = V (H_dew - H_bubble) of the distillate composition (total condenser, DWSIM), reboiler
   from the overall energy balance.

Alternative methods (each selectable on its own): ``alpha_mode='average'|'geometric'`` (mean of the top and bottom
alphas instead of the feed alpha), ``nmin_method='winn'``, ``q_mode='enthalpy'`` (thermal condition from enthalpies) or
``'hybrid'`` (1 - vapor fraction for a two-phase feed, thermal condition for a single-phase one),
``feed_stage_method='kirkbride'`` (N_R / N_S = [(z_HK / z_LK)(B / D)(x_B,LK / x_D,HK)^2]^0.206, not what DWSIM
does) or ``'winn_sections'`` (feed stage from the Winn minimum stages of the rectifying and stripping
sections, using the liquid phase of the feed: N_above / N = N_R,min / (N_R,min + N_S,min); the Winn parameter is
ill-conditioned when K of the heavy key is nearly equal at both ends of a section, a warning is issued). Only the
defaults have been compared with DWSIM.

Limitation of the Gilliland correlations: the Eduljee form does not diverge at R = Rmin. N tends to
(0.75 + Nmin) / 0.25 = 4 Nmin + 3, so a number of stages above that value cannot be reached with ``n_stages``
(an error is raised) and the stage count near Rmin is underestimated. The Molokanov form does diverge
(Y -> 1 and N -> infinity as X -> 0).

Assumptions and limits of the Fenske-Underwood-Gilliland method: constant relative volatility (the value at the feed
can differ a lot from the mean of the column, see ``alpha_mode``), constant molar overflow, components other than the
keys do not distribute (lighter ones leave in the distillate, heavier ones in the bottoms). For strongly non-ideal
mixtures or distributing components the results are a starting point for a rigorous column, not a design. The numbers
of stages are real (not rounded up) to match DWSIM.

q: ``q_mode='dwsim'`` takes q = 1 - vapor fraction, so a subcooled liquid gives q = 1 and a superheated vapor q = 0
(DWSIM does the same and it is the validated default). The thermal condition (q > 1 or q < 0) is ``q_mode='enthalpy'``.

Not implemented: partial condenser with vapor and liquid distillate (needs the distillate vapor fraction as an
input and a reference to compare with), column diameter. The height is an estimate with one DWSIM data point (see
`ShortcutResult.height`). Partial condenser with a vapor distillate (``condenser='partial'``): the distillate leaves
as saturated vapor at its dew point, Qc = L (H_dew - H_bubble) of the distillate composition (only the reflux
condenses) and Qr from the overall balance; validated against DWSIM (benzene/toluene, R = 3).

Units: T [K], P [bar], flows [kmol/s] (mole flows in the report in mol/s), duties [kW], height [m].
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq

_CONDENSERS = ('total', 'partial')
_NMIN = ('fenske', 'winn')
_ALPHA = ('feed', 'average', 'geometric')
_GILLILAND = ('eduljee', 'molokanov')
_Q_MODES = ('dwsim', 'enthalpy', 'hybrid')
_FEED_STAGE = ('fenske', 'kirkbride', 'winn_sections')
TRACE = 1e-10                      # mole fraction given to absent components when their K is evaluated
KIRKBRIDE_EXPONENT = 0.206
EDULJEE_EXPONENT = 0.5668


# --------------------------------------------------------------------------- Gilliland correlations

def gilliland_Y(X: float, correlation: str = 'eduljee') -> float:
    """Y = (N - Nmin) / (N + 1) as a function of X = (R - Rmin) / (R + 1), 0 < X < 1."""
    if not 0.0 < X < 1.0:
        raise ValueError(f"X must be in (0, 1), got {X}.")
    if correlation == 'eduljee':
        return 0.75 * (1.0 - X ** EDULJEE_EXPONENT)
    if correlation == 'molokanov':
        return 1.0 - math.exp((1.0 + 54.4 * X) / (11.0 + 117.2 * X) * (X - 1.0) / math.sqrt(X))
    raise ValueError(f"gilliland must be one of {_GILLILAND}, got '{correlation}'.")


def gilliland_X(Y: float, correlation: str = 'eduljee') -> float:
    """Inverse of `gilliland_Y` (0 < Y < Y(0+))."""
    if correlation == 'eduljee':
        if not 0.0 < Y < 0.75:
            raise ValueError(f"With Eduljee, Y must be in (0, 0.75), got {Y}. Y >= 0.75 means N >= 4 Nmin + 3, "
                             "which the Eduljee correlation cannot reach (it does not diverge at Rmin).")
        return (1.0 - Y / 0.75) ** (1.0 / EDULJEE_EXPONENT)
    if correlation == 'molokanov':
        if not 0.0 < Y < 1.0:
            raise ValueError(f"Y must be in (0, 1), got {Y}.")
        return brentq(lambda X: gilliland_Y(X, 'molokanov') - Y, 1e-12, 1.0 - 1e-12, xtol=1e-14)
    raise ValueError(f"gilliland must be one of {_GILLILAND}, got '{correlation}'.")


# --------------------------------------------------------------------------- result

@dataclass
class ShortcutResult:
    components: list
    light_key: str
    heavy_key: str
    condenser: str
    alpha_mode: str
    nmin_method: str
    gilliland: str
    q_mode: str
    feed_stage_method: str
    # separation
    distillate: object
    bottoms: object
    D: float                       # [kmol/s]
    B: float
    x_D: dict
    x_B: dict
    rec_LK: float                  # fraction of the feed light key in the distillate
    rec_HK: float                  # fraction of the feed heavy key in the distillate
    # volatilities
    alpha_feed: dict
    alpha_top: dict
    alpha_bottom: dict
    alpha_used: dict               # the ones used in Fenske and Underwood
    q: float
    theta: float
    # design
    R_min: float
    R: float
    N_min: float
    N: float                       # equilibrium stages (DWSIM "actual number of stages")
    N_rectifying: float            # stages above the feed
    N_stripping: float             # stages below the feed
    feed_stage: float              # DWSIM "optimum feed stage" = N_stripping (counted from the bottom)
    # internal flows [kmol/s]
    L_rect: float
    V_rect: float
    L_strip: float
    V_strip: float
    # duties [kW], both positive (heat removed in the condenser, heat added in the reboiler)
    Qc_kW: float
    Qr_kW: float
    T_top: float
    T_bottom: float
    P_top: float
    P_bottom: float
    tray_height: float

    @property
    def height(self) -> float:
        """Estimated column height [m] = tray_height * (N + 2). HYPOTHESIS: one DWSIM data point (6.74513 m with
        N = 11.4903 and tray height 0.5 m); tray_height * N + 1 m fits it equally well."""
        return self.tray_height * (self.N + 2.0)

    def report(self) -> str:
        sep = '-' * 64
        L = [sep,
             f"Shortcut column ({self.condenser} condenser): LK {self.light_key}, HK {self.heavy_key}",
             f"alpha: {self.alpha_mode}   Nmin: {self.nmin_method}   Gilliland: {self.gilliland}   q: {self.q_mode}"
             f"   feed stage: {self.feed_stage_method}",
             sep,
             f"Minimum reflux ratio    : {self.R_min:.5f}",
             f"Reflux ratio            : {self.R:.5f}",
             f"Minimum stages          : {self.N_min:.5f}",
             f"Actual stages           : {self.N:.5f}",
             f"Feed stage (DWSIM)      : {self.feed_stage:.5f}   (above: {self.N_rectifying:.5f}, below: {self.N_stripping:.5f})",
             f"Rectifying L / V        : {self.L_rect * 1000:.4f} / {self.V_rect * 1000:.4f} mol/s",
             f"Stripping  L / V        : {self.L_strip * 1000:.4f} / {self.V_strip * 1000:.4f} mol/s",
             f"Condenser duty          : {self.Qc_kW:.3f} kW   (T {self.T_top:.3f} K, {self.P_top:.4f} bar)",
             f"Reboiler duty           : {self.Qr_kW:.3f} kW   (T {self.T_bottom:.3f} K, {self.P_bottom:.4f} bar)",
             f"Distillate / bottoms    : {self.D * 1000:.4f} / {self.B * 1000:.4f} mol/s",
             f"Recovery LK / HK in D   : {self.rec_LK:.5f} / {self.rec_HK:.5f}",
             f"Estimated height        : {self.height:.3f} m (hypothesis, see ShortcutResult.height)",
             sep]
        return '\n'.join(L)


# --------------------------------------------------------------------------- column

class ShortcutColumn:
    """Shortcut column. Key specification: give ``x_LK_bottoms`` and ``x_HK_distillate`` (DWSIM) or ``rec_LK`` and
    ``rec_HK`` (fractions of the feed LK / HK that go to the distillate). Hydraulic specification: give exactly one of
    ``reflux_ratio``, ``reflux_to_min`` (R = reflux_to_min * Rmin) or ``n_stages`` (equilibrium stages, same
    definition as the DWSIM "actual number of stages")."""

    def __init__(self, light_key: str, heavy_key: str, P_top: float, P_bottom: float | None = None, *,
                 x_LK_bottoms: float | None = None, x_HK_distillate: float | None = None,
                 rec_LK: float | None = None, rec_HK: float | None = None,
                 reflux_ratio: float | None = None, reflux_to_min: float | None = None,
                 n_stages: float | None = None,
                 condenser: str = 'total', alpha_mode: str = 'feed', nmin_method: str = 'fenske',
                 gilliland: str = 'eduljee', q_mode: str = 'dwsim', feed_stage_method: str = 'fenske',
                 tray_height: float = 0.5):
        by_x = (x_LK_bottoms is not None, x_HK_distillate is not None)
        by_rec = (rec_LK is not None, rec_HK is not None)
        if any(by_x) and any(by_rec):
            raise ValueError("Give the key specification either as (x_LK_bottoms, x_HK_distillate) or as "
                             "(rec_LK, rec_HK), not both.")
        if not (all(by_x) or all(by_rec)):
            raise ValueError("Give both x_LK_bottoms and x_HK_distillate, or both rec_LK and rec_HK.")
        for name, v in (('x_LK_bottoms', x_LK_bottoms), ('x_HK_distillate', x_HK_distillate),
                        ('rec_LK', rec_LK), ('rec_HK', rec_HK)):
            if v is not None and not 0.0 < v < 1.0:
                raise ValueError(f"{name} must be in (0, 1), got {v}.")
        if sum(v is not None for v in (reflux_ratio, reflux_to_min, n_stages)) != 1:
            raise ValueError("Give exactly one of reflux_ratio, reflux_to_min, n_stages.")
        if reflux_ratio is not None and reflux_ratio <= 0:
            raise ValueError("reflux_ratio must be positive (use reflux_to_min for a multiple of the minimum).")
        if reflux_to_min is not None and reflux_to_min <= 1.0:
            raise ValueError("reflux_to_min must be above 1.")
        if n_stages is not None and n_stages <= 0:
            raise ValueError("n_stages must be positive.")
        if condenser == 'partial_vl':
            raise NotImplementedError("Partial condenser with vapor and liquid distillate is not implemented yet "
                                      "(it needs the distillate vapor fraction and a reference to compare with).")
        for val, allowed, label in ((condenser, _CONDENSERS, 'condenser'), (alpha_mode, _ALPHA, 'alpha_mode'),
                                    (feed_stage_method, _FEED_STAGE, 'feed_stage_method'),
                                    (nmin_method, _NMIN, 'nmin_method'), (gilliland, _GILLILAND, 'gilliland'),
                                    (q_mode, _Q_MODES, 'q_mode')):
            if val not in allowed:
                raise ValueError(f"{label} must be one of {allowed}, got '{val}'.")
        P_bottom = P_top if P_bottom is None else P_bottom
        if P_top <= 0 or P_bottom < P_top:
            raise ValueError("Pressures must satisfy 0 < P_top <= P_bottom.")
        if tray_height <= 0:
            raise ValueError("tray_height must be positive.")
        self.LK, self.HK = str(light_key).lower(), str(heavy_key).lower()
        if self.LK == self.HK:
            raise ValueError("The light and heavy keys must be different components.")
        self.P_top, self.P_bottom = float(P_top), float(P_bottom)
        self.x_LK_bottoms, self.x_HK_distillate = x_LK_bottoms, x_HK_distillate
        self.rec_LK, self.rec_HK = rec_LK, rec_HK
        self.reflux_ratio, self.reflux_to_min, self.n_stages = reflux_ratio, reflux_to_min, n_stages
        self.condenser, self.alpha_mode, self.nmin_method = condenser, alpha_mode, nmin_method
        self.gilliland, self.q_mode, self.tray_height = gilliland, q_mode, float(tray_height)
        self.feed_stage_method = feed_stage_method

    # ---- helpers

    @staticmethod
    def _names(feed) -> list:
        return feed.database['Component'].astype(str).str.lower().tolist()

    def _distillate_flows(self, feed, z: dict, order: list):
        F = feed.fn
        f = {n: F * z[n] for n in order}
        pos = {n: i for i, n in enumerate(order)}
        i_lk, i_hk = pos[self.LK], pos[self.HK]
        light = [n for n in order if pos[n] < i_lk]
        heavy = [n for n in order if pos[n] > i_hk]
        d = {n: (f[n] if n in light else 0.0) for n in order}
        Dl = sum(d.values())
        if self.rec_LK is not None:
            d[self.LK], d[self.HK] = self.rec_LK * f[self.LK], self.rec_HK * f[self.HK]
        else:
            xB, xD = self.x_LK_bottoms, self.x_HK_distillate
            # f_LK - u = xB (F - Dl - u - v) ;  v (1 - xD) = xD (Dl + u)
            A = np.array([[-1.0 + xB, xB], [-xD, 1.0 - xD]])
            rhs = np.array([xB * (F - Dl) - f[self.LK], xD * Dl])
            u, v = np.linalg.solve(A, rhs)
            d[self.LK], d[self.HK] = float(u), float(v)
        for n in heavy:
            d[n] = 0.0
        for n in (self.LK, self.HK):
            if not 0.0 < d[n] < f[n]:
                raise ValueError(f"The key specification is infeasible: '{n}' would have {d[n] / f[n]:.4f} of its "
                                 "feed flow in the distillate (must be strictly between 0 and 1).")
        if d[self.LK] / f[self.LK] <= d[self.HK] / f[self.HK]:
            raise ValueError("The key specification is infeasible: the light key recovery in the distillate "
                             f"({d[self.LK] / f[self.LK]:.4f}) must exceed that of the heavy key "
                             f"({d[self.HK] / f[self.HK]:.4f}).")
        D = sum(d.values())
        B = F - D
        if not 0.0 < D < F:
            raise ValueError("The key specification gives no distillate or no bottoms.")
        b = {n: f[n] - d[n] for n in order}
        return d, b, D, B

    @staticmethod
    def _comp(flows: dict, total: float) -> dict:
        return {n: v / total for n, v in flows.items() if v > 0.0}

    def _K(self, stream, order, names) -> dict:
        Ki = np.asarray(stream.props.Ki, dtype=float)
        active = [n for n in names if stream.comp_mol.get(n, 0.0) > 0]      # Ki holds the active components only
        if len(active) != len(Ki):
            raise RuntimeError("Unexpected size of the K vector of the stream.")
        K = {n: float(Ki[active.index(n)]) for n in order}
        if not all(np.isfinite(v) and v > 0 for v in K.values()):
            raise RuntimeError(f"K values are not usable at T={stream.T:.2f} K, P={stream.P:.3f} bar: {K}")
        return K

    def _K_feed(self, feed, order, names) -> dict:
        """K values at the state of the feed. Two phases: the K of the flash. A single-phase feed has no second phase,
        so K = phi_L / phi_V is evaluated with the incipient phase of the stability trial (the composition x of the
        trial liquid for a vapor feed, y of the trial vapor for a liquid one, found by the fixed point
        x = z / K normalized). DWSIM does this: its 'liquid phase' of a vapor feed has exactly that composition."""
        if feed.props.phase == 'two_phase':
            return self._K(feed, order, names)
        try:
            eng = feed._engine
            zfull = feed.component_data['frac_mol'].to_numpy(dtype=float)
            sub, zs, act = eng._active(zfull)
            k = sub.k
            T, P = float(feed.T), float(feed.P)
            vapor = feed.props.phase == 'vapor'
            lz = lambda x: k.phi(T, P, x, 'l')[0]
            vz = lambda x: k.phi_eos(T, P, x, 'v')[0]
            ref = vz(zs) if vapor else lz(zs)                       # the feed phase; the trial phase is the other one
            K_w = np.asarray(sub._wilson(T, P), float)

            def trial(w):
                for _ in range(300):
                    K = np.exp((lz(w) - ref) if vapor else (ref - vz(w)))
                    wn = zs / K if vapor else zs * K
                    wn = wn / wn.sum()
                    done = float(np.max(np.abs(wn - w))) < 1e-13
                    w = wn
                    if done:
                        break
                lnphi_w = lz(w) if vapor else vz(w)
                K = np.exp((lnphi_w - ref) if vapor else (ref - lnphi_w))
                tpd = float(np.sum(w * (np.log(w) + lnphi_w - np.log(zs) - ref)))     # tangent plane distance
                return K, tpd

            starts = [zs.copy()]                                    # the feed itself and a Wilson-based estimate
            w0 = zs / K_w if vapor else zs * K_w
            starts.append(w0 / w0.sum())
            best = None
            for w0 in starts:
                K, tpd = trial(w0)
                if np.all(np.isfinite(K)) and float(np.max(np.abs(np.log(K)))) > 1e-9:      # not the trivial solution
                    if best is None or tpd < best[1]:
                        best = (K, tpd)
        except (AttributeError, KeyError, IndexError, ValueError, FloatingPointError, np.linalg.LinAlgError) as e:
            import warnings
            warnings.warn(f"K of the single-phase feed from the stability trial failed ({type(e).__name__}: {e}); "
                          "using the K of the stream instead (alpha can differ from DWSIM by a few tenths of a percent).",
                          RuntimeWarning, stacklevel=3)
            return self._K(feed, order, names)
        if best is None:                                            # only the trivial solution: no second root
            return self._K(feed, order, names)
        K = best[0]
        active = [n for n in names if feed.comp_mol.get(n, 0.0) > 0]
        return {n: float(K[active.index(n)]) for n in order}

    def _winn_section(self, K1: dict, K2: dict, x1: dict, x2: dict) -> float:
        """Winn minimum number of stages between two ends of a section (K at both ends, liquid compositions)."""
        LK, HK = self.LK, self.HK
        lr = math.log(K1[HK] / K2[HK])
        if abs(lr) < 0.05:
            import warnings
            warnings.warn("K of the heavy key is almost equal at both ends of a section (feed and a product at "
                          "similar temperature): the Winn parameter is ill-conditioned and the feed stage is "
                          "uncertain.", stacklevel=3)
        if lr == 0.0:
            raise ValueError("K of the heavy key is identical at both ends of a section: Winn is undefined.")
        theta = math.log(K1[LK] / K2[LK]) / lr
        beta = math.exp(math.log(K1[LK]) - theta * math.log(K1[HK]))
        return math.log((x1[LK] / x2[LK]) / (x1[HK] / x2[HK]) ** theta) / math.log(beta)

    def _q(self, feed) -> float:
        if self.q_mode == 'dwsim' or (self.q_mode == 'hybrid' and feed.props.phase == 'two_phase'):
            return 1.0 - float(feed.props.V_frac)
        Hf = feed.props.H_total_mix
        zdict = feed.comp_mol
        dew = feed.at(V_frac=1.0, comp_mol=zdict, T_guess=feed.T)
        bub = feed.at(V_frac=0.0, comp_mol=zdict, T_guess=feed.T)
        Hv, Hl = dew.props.H_total_mix, bub.props.H_total_mix
        return (Hv - Hf) / (Hv - Hl)

    # ---- run

    def run(self, feed) -> ShortcutResult:
        names = self._names(feed)
        z = {n: float(v) for n, v in feed.comp_mol.items() if v > 0}
        s = sum(z.values())
        z = {n: v / s for n, v in z.items()}
        for k in (self.LK, self.HK):
            if k not in z:
                raise ValueError(f"Key component '{k}' is not in the feed (components: {list(z)}).")
        F = float(feed.fn)

        # volatility order from the K values of the feed
        Kf = self._K_feed(feed, list(z), names)
        order = sorted(z, key=lambda n: -Kf[n])
        pos = {n: i for i, n in enumerate(order)}
        if pos[self.LK] > pos[self.HK]:
            raise ValueError(f"'{self.LK}' is less volatile than '{self.HK}' at the feed: swap the keys. "
                             f"Order at the feed: {order}.")
        if pos[self.LK] != pos[self.HK] - 1:
            raise ValueError(f"The keys must be adjacent in volatility. Order at the feed: {order}.")
        if Kf[self.LK] <= Kf[self.HK]:
            raise ValueError("The light key is not more volatile than the heavy key.")

        d, b, D, B = self._distillate_flows(feed, z, order)
        xD, xB = self._comp(d, D), self._comp(b, B)

        # products and K at top and bottom
        top_V = 1.0 if self.condenser == 'partial' else 0.0
        dist = feed.at(P=self.P_top, fn=D, comp_mol=xD, V_frac=top_V, T_guess=feed.T)
        bott = feed.at(P=self.P_bottom, fn=B, comp_mol=xB, V_frac=0.0, T_guess=feed.T)
        # K at the product bubble (or dew) points; components absent from a product get a trace amount so that
        # their K is defined (the product streams themselves keep the exact compositions)
        def trace(x):
            return {n: max(x.get(n, 0.0), TRACE) for n in order}
        K_top = self._K(feed.at(P=self.P_top, fn=D, comp_mol=trace(xD), V_frac=top_V, T_guess=feed.T), order, names)
        K_bot = self._K(feed.at(P=self.P_bottom, fn=B, comp_mol=trace(xB), V_frac=0.0, T_guess=feed.T), order, names)

        def alphas(K):
            return {n: K[n] / K[self.HK] for n in order}
        a_feed, a_top, a_bot = alphas(Kf), alphas(K_top), alphas(K_bot)
        if self.alpha_mode == 'feed':
            a = a_feed
        elif self.alpha_mode == 'average':
            a = {n: 0.5 * (a_top[n] + a_bot[n]) for n in order}
        else:
            a = {n: math.sqrt(a_top[n] * a_bot[n]) for n in order}
        if a[self.LK] <= 1.0:
            raise ValueError(f"alpha(LK/HK) = {a[self.LK]:.4f} is not above 1: the keys cannot be separated.")
        between = [n for n in order if 1.0 < a[n] < a[self.LK]]
        if between:
            raise ValueError(f"Components between the keys in volatility: {between}.")

        # minimum stages
        ratio = (xD[self.LK] / xD[self.HK]) * (xB[self.HK] / xB[self.LK])
        if self.nmin_method == 'fenske':
            N_min = math.log(ratio) / math.log(a[self.LK])
        else:
            lnK = lambda K, n: math.log(K[n])
            dHK = lnK(K_top, self.HK) - lnK(K_bot, self.HK)
            if abs(dHK) < 1e-12:
                raise ValueError("Winn minimum stages: K of the heavy key is the same at the top and at the bottom "
                                 "(isothermal column), the parameter theta is undefined. Use nmin_method='fenske'.")
            theta_w = (lnK(K_top, self.LK) - lnK(K_bot, self.LK)) / dHK
            beta = math.exp(lnK(K_top, self.LK) - theta_w * lnK(K_top, self.HK))
            if not beta > 1.0:
                raise ValueError(f"Winn minimum stages: beta = {beta:.4g} must exceed 1 (the light key is not more "
                                 "volatile than the heavy key at the column conditions).")
            N_min = math.log((xD[self.LK] / xB[self.LK]) * (xB[self.HK] / xD[self.HK]) ** theta_w) / math.log(beta)

        # Underwood
        q = self._q(feed)
        g = lambda th: sum(a[n] * z[n] / (a[n] - th) for n in order) - (1.0 - q)
        theta = brentq(g, 1.0 + 1e-9, a[self.LK] - 1e-9, xtol=1e-14)
        R_min = sum(a[n] * xD.get(n, 0.0) / (a[n] - theta) for n in order) - 1.0

        # Gilliland
        if self.n_stages is not None:
            if self.n_stages <= N_min:
                raise ValueError(f"n_stages ({self.n_stages}) must exceed the minimum number of stages ({N_min:.4f}).")
            Y = (self.n_stages - N_min) / (self.n_stages + 1.0)
            X = gilliland_X(Y, self.gilliland)
            R = (X + R_min) / (1.0 - X)
            N = float(self.n_stages)
        else:
            R = self.reflux_ratio if self.reflux_ratio is not None else self.reflux_to_min * R_min
            if R <= R_min:
                raise ValueError(f"The reflux ratio ({R:.4f}) must exceed the minimum ({R_min:.4f}).")
            X = (R - R_min) / (R + 1.0)
            Y = gilliland_Y(X, self.gilliland)
            N = (Y + N_min) / (1.0 - Y)

        # feed stage
        if self.feed_stage_method == 'fenske':
            # minimum stages of each section with one alpha: ln[(xD_LK/xD_HK)(z_HK/z_LK)] above the feed,
            # ln[(z_LK/z_HK)(xB_HK/xB_LK)] below it
            n_r = math.log((xD[self.LK] / xD[self.HK]) * (z[self.HK] / z[self.LK]))
            n_s = math.log((z[self.LK] / z[self.HK]) * (xB[self.HK] / xB[self.LK]))
            if n_r <= 0.0 or n_s <= 0.0:
                raise ValueError("The feed composition is not between the product compositions "
                                 f"(minimum stages of the sections: {n_r:.3f}, {n_s:.3f}).")
            N_R = N * n_r / (n_r + n_s)
        elif self.feed_stage_method == 'kirkbride':
            r = ((z[self.HK] / z[self.LK]) * (B / D) * (xB[self.LK] / xD[self.HK]) ** 2) ** KIRKBRIDE_EXPONENT
            N_R = N * r / (1.0 + r)
        else:
            # 'winn_sections': Winn minimum stages of each section, with the liquid of the feed:
            # N_above / N = N_R,min / (N_R,min + N_S,min)
            if feed.props.phase == 'two_phase':
                active = [n for n in names if feed.comp_mol.get(n, 0.0) > 0]
                x_feed = dict(zip(active, map(float, feed.props.xi)))
            else:
                x_feed = z
            n_r = self._winn_section(K_top, Kf, xD, x_feed)
            n_s = self._winn_section(Kf, K_bot, x_feed, xB)
            if n_r <= 0.0 or n_s <= 0.0:
                raise ValueError("The feed composition is not between the product compositions "
                                 f"(minimum stages of the sections: {n_r:.3f}, {n_s:.3f}).")
            N_R = N * n_r / (n_r + n_s)

        # internal flows
        L_r, V_r = R * D, (R + 1.0) * D
        L_s, V_s = L_r + q * F, V_r - (1.0 - q) * F

        # duties
        if self.condenser == 'total':
            dew = feed.at(P=self.P_top, fn=D, comp_mol=xD, V_frac=1.0, T_guess=feed.T)
            lam = dew.props.H_total_mix - dist.props.H_total_mix
            Qc = V_r * lam
        else:                                           # vapor distillate: only the reflux condenses (DWSIM)
            bub = feed.at(P=self.P_top, fn=D, comp_mol=xD, V_frac=0.0, T_guess=feed.T)
            lam = dist.props.H_total_mix - bub.props.H_total_mix
            Qc = L_r * lam
        E_f = F * feed.props.H_total_mix
        E_d, E_b = D * dist.props.H_total_mix, B * bott.props.H_total_mix
        Qr = Qc + E_d + E_b - E_f

        return ShortcutResult(
            components=order, light_key=self.LK, heavy_key=self.HK, condenser=self.condenser,
            alpha_mode=self.alpha_mode, nmin_method=self.nmin_method, gilliland=self.gilliland, q_mode=self.q_mode,
            feed_stage_method=self.feed_stage_method,
            distillate=dist, bottoms=bott, D=D, B=B, x_D=xD, x_B=xB,
            rec_LK=d[self.LK] / (F * z[self.LK]), rec_HK=d[self.HK] / (F * z[self.HK]),
            alpha_feed=a_feed, alpha_top=a_top, alpha_bottom=a_bot, alpha_used=a, q=q, theta=theta,
            R_min=R_min, R=R, N_min=N_min, N=N, N_rectifying=N_R, N_stripping=N - N_R, feed_stage=N - N_R,
            L_rect=L_r, V_rect=V_r, L_strip=L_s, V_strip=V_s, Qc_kW=Qc, Qr_kW=Qr,
            T_top=float(dist.T), T_bottom=float(bott.T), P_top=self.P_top, P_bottom=self.P_bottom,
            tray_height=self.tray_height)


DSTWU = ShortcutColumn