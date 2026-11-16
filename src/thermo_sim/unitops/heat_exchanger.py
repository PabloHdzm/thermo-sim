"""Two-stream heat exchanger built on `Stream` (counter-current or co-current).

Energy balance and design equation:

    Q = fn_hot (H_hot_in - H_hot_out) = fn_cold (H_cold_out - H_cold_in)      [kW]
    Q = U A F LMTD = UA F LMTD

with the real enthalpies of the streams (`inlet.at(T=..., P=...).props.H_total_mix`), so liquids,
vapors and mixtures are all handled by whatever thermodynamic method the streams carry. The outlet
pressures are P_in - dP of each side.

One thermal specification (`mode`):

    'ua'          UA (or U and A) given; Q and both outlet temperatures are solved
                  (DWSIM: Calculate Outlet Temperatures (UA))
    'hot_out'     T_hot_out given; Q, the cold outlet and the UA required are calculated
    'cold_out'    T_cold_out given; same
    'duty'        Q given
    'efficiency'  Q = efficiency * Q_max (thermal efficiency, DWSIM: Specify heat transfer efficiency)

When U is also given, the required area A = UA / U is reported (DWSIM: Calculate Area).

Units: T [K], P [bar], H [kJ/kmol], S [kJ/(kmol K)], flows [kmol/s], Q [kW], UA [kW/K],
U [kW/(m2 K)], A [m2], dP [bar].

LMTD assumes constant heat capacities along the exchanger, as the design equation of DWSIM does.
With a phase change on either side the report warns and gives `UA_profile`, the integral of dQ / dT
along the T-Q profile, which does not rely on that assumption (and `pinch`, its smallest dT).

`hot` and `cold` must provide `.at(T=..., P=...)` (same stream at another state), `.T`, `.P`, `.fn`
and `.props` with `H_total_mix`, `S_total_mix`, `Cp_total_mix` and `phase`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from scipy.optimize import brentq

_FLOWS = ('counter', 'parallel')
_MODES = ('ua', 'hot_out', 'cold_out', 'duty', 'efficiency')


def lmtd(dT1: float, dT2: float) -> float:
    """Logarithmic mean of the two terminal temperature differences (0 if either is not positive)."""
    if dT1 <= 0.0 or dT2 <= 0.0:
        return 0.0
    if abs(dT1 - dT2) < 1e-9 * max(dT1, dT2):
        return 0.5 * (dT1 + dT2)
    return (dT1 - dT2) / math.log(dT1 / dT2)


def lmtd_correction_factor(T_hot_in: float, T_hot_out: float, T_cold_in: float, T_cold_out: float) -> float:
    """F of a 1-2 shell-and-tube exchanger (one shell pass, an even number of tube passes; Bowman,
    Mueller and Nagle, 1940): LMTD_effective = F * LMTD_counter-current. F = 1 for R = 0 (isothermal side)."""
    R = (T_hot_in - T_hot_out) / (T_cold_out - T_cold_in)
    P = (T_cold_out - T_cold_in) / (T_hot_in - T_cold_in)
    if R < 1e-12 or P < 1e-12:
        return 1.0
    S = math.sqrt(R * R + 1.0)
    if abs(R - 1.0) < 1e-9:
        num = S * P / (1.0 - P)
        den = math.log((2.0 - P * (2.0 - S)) / (2.0 - P * (2.0 + S)))
    else:
        num = S / (R - 1.0) * math.log((1.0 - P) / (1.0 - R * P))
        den = math.log((2.0 - P * (R + 1.0 - S)) / (2.0 - P * (R + 1.0 + S)))
    return num / den


@dataclass
class ProfilePoint:
    fraction: float            # Q / Q_total from the hot inlet end
    Q: float                   # heat transferred so far [kW]
    T_hot: float
    T_cold: float
    dT: float


@dataclass
class HeatExchangerResult:
    flow: str
    mode: str
    Q_kW: float                # heat transferred from hot to cold (positive)
    Q_hot_kW: float            # fn (H_in - H_out) of the hot side
    Q_cold_kW: float           # fn (H_out - H_in) of the cold side
    balance_error: float       # |Q_hot - Q_cold| / Q_hot
    Q_max_kW: float            # largest duty possible with these inlets (and pressures)
    efficiency: float          # Q / Q_max
    T_hot_in: float
    T_hot_out: float
    T_cold_in: float
    T_cold_out: float
    P_hot_out: float
    P_cold_out: float
    LMTD: float                # counter-current / co-current LMTD (uncorrected) [K]
    F: float
    UA: float                  # Q / (F LMTD) [kW/K]
    U: float | None
    A: float | None            # UA / U [m2] (None without U)
    C_hot: float               # fn dH / dT of each side [kW/K]
    C_cold: float
    NTU: float                 # UA / C_min
    S_gen_kWK: float           # entropy generation [kW/K]
    exergy_destroyed_kW: float
    pinch: float | None        # smallest dT along the profile [K]
    UA_profile: float | None   # integral dQ / dT along the profile [kW/K]
    profile: list = field(default_factory=list)
    hot_out: object = None     # Stream
    cold_out: object = None    # Stream
    warnings: list = field(default_factory=list)

    def report(self) -> str:
        sep = '-' * 64
        lines = [sep, f"Heat exchanger ({self.flow}, mode '{self.mode}')", sep,
                 f"Duty                    : {self.Q_kW:.3f} kW   (hot {self.Q_hot_kW:.3f}, cold {self.Q_cold_kW:.3f}, "
                 f"balance error {100 * self.balance_error:.4f} %)",
                 f"Maximum duty            : {self.Q_max_kW:.3f} kW   thermal efficiency {100 * self.efficiency:.3f} %",
                 f"Hot  T in -> out        : {self.T_hot_in:.3f} -> {self.T_hot_out:.3f} K   (P out {self.P_hot_out:.4f} bar)",
                 f"Cold T in -> out        : {self.T_cold_in:.3f} -> {self.T_cold_out:.3f} K   (P out {self.P_cold_out:.4f} bar)",
                 f"LMTD                    : {self.LMTD:.4f} K   F = {self.F:.4f}",
                 f"UA                      : {self.UA:.4f} kW/K   NTU = {self.NTU:.4f}"]
        if self.A is not None:
            lines.append(f"U / area                : {1000 * self.U:.2f} W/(m2 K) / {self.A:.4f} m2")
        if self.UA_profile is not None:
            lines.append(f"UA from the profile     : {self.UA_profile:.4f} kW/K   pinch dT = {self.pinch:.3f} K")
        lines += [f"Entropy generation      : {self.S_gen_kWK:.6f} kW/K   exergy destroyed {self.exergy_destroyed_kW:.3f} kW"]
        if self.profile:
            lines += ['', '  Q frac     Q [kW]   T hot [K]  T cold [K]   dT [K]']
            lines += [f'  {p.fraction:6.3f} {p.Q:10.3f} {p.T_hot:10.3f} {p.T_cold:10.3f} {p.dT:9.3f}' for p in self.profile]
        for w in self.warnings:
            lines.append(f'WARNING: {w}')
        lines.append(sep)
        return '\n'.join(lines)


class HeatExchanger:
    def __init__(self, mode: str = 'ua', *, flow: str = 'counter', UA: float | None = None,
                 U: float | None = None, A: float | None = None, T_hot_out: float | None = None,
                 T_cold_out: float | None = None, duty: float | None = None,
                 efficiency: float | None = None, dP_hot: float = 0.0, dP_cold: float = 0.0,
                 F: float = 1.0, n_profile: int = 20, T_ref: float = 298.15):
        if flow not in _FLOWS:
            raise ValueError(f"flow must be one of {_FLOWS}, got '{flow}'.")
        if mode not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}, got '{mode}'.")
        if dP_hot < 0 or dP_cold < 0:
            raise ValueError('dP_hot and dP_cold must be >= 0.')
        if not 0.0 < F <= 1.0:
            raise ValueError('F must be in (0, 1].')
        if mode == 'ua':
            if UA is None and (U is None or A is None):
                raise ValueError("mode 'ua' needs UA, or U and A.")
            if UA is None:
                UA = U * A
        need = {'hot_out': T_hot_out, 'cold_out': T_cold_out, 'duty': duty, 'efficiency': efficiency}
        if mode in need and need[mode] is None:
            raise ValueError(f"mode '{mode}' needs {'T_' + mode if mode.endswith('out') else mode}.")
        if mode == 'efficiency' and not 0.0 < efficiency <= 1.0:
            raise ValueError('efficiency must be in (0, 1].')
        if mode == 'duty' and duty <= 0.0:
            raise ValueError('duty must be > 0.')
        if mode != 'ua' and UA is not None:
            raise ValueError(f"UA is an input of mode 'ua' only; in mode '{mode}' it is a result (give U to get the area).")
        if U is not None and U <= 0.0:
            raise ValueError('U must be > 0.')
        self.mode, self.flow = mode, flow
        self.UA, self.U, self.A = UA, U, (A if mode == 'ua' else None)
        self.T_hot_out, self.T_cold_out, self.duty, self.efficiency = T_hot_out, T_cold_out, duty, efficiency
        self.dP_hot, self.dP_cold, self.F = float(dP_hot), float(dP_cold), float(F)
        self.n_profile, self.T_ref = int(n_profile), float(T_ref)

    # thermodynamic helpers

    @staticmethod
    def _H(inlet, T: float, P: float) -> float:
        return inlet.at(T=T, P=P).props.H_total_mix

    @staticmethod
    def _T_at_H(inlet, P: float, H: float, T_lo: float, T_hi: float) -> float:
        """T in [T_lo, T_hi] with H(T, P) = H [kJ/kmol] (H increases with T)."""
        f = lambda T: inlet.at(T=T, P=P).props.H_total_mix - H
        flo, fhi = f(T_lo), f(T_hi)
        # The bracket ends are isothermal states at the outlet pressure, so a target that
        # sits at an end of the range (Q = 0 or Q = Qmax) can fall just outside it because
        # of the pressure drop. Clamp to the end in that case.
        if flo >= 0.0:
            return T_lo
        if fhi <= 0.0:
            return T_hi
        return float(brentq(f, T_lo, T_hi, xtol=1e-10, rtol=1e-14, maxiter=200))

    def _lmtd(self, T_hi, T_ho, T_ci, T_co) -> float:
        if self.flow == 'counter':
            return lmtd(T_hi - T_co, T_ho - T_ci)
        return lmtd(T_hi - T_ci, T_ho - T_co)

    # model

    def run(self, hot, cold) -> HeatExchangerResult:
        T_hi, T_ci = float(hot.T), float(cold.T)
        P_ho, P_co = float(hot.P) - self.dP_hot, float(cold.P) - self.dP_cold
        if P_ho <= 0.0 or P_co <= 0.0:
            raise ValueError('Pressure drop exceeds the inlet pressure.')
        if T_hi <= T_ci:
            raise ValueError(f'The hot inlet ({T_hi:.2f} K) must be hotter than the cold inlet ({T_ci:.2f} K).')
        fh, fc = float(hot.fn), float(cold.fn)
        H_hi, H_ci = hot.props.H_total_mix, cold.props.H_total_mix
        warn: list = []

        # largest duty: the hot side cooled to T_ci or the cold side heated to T_hi
        Q_hot_max = fh * (H_hi - self._H(hot, T_ci, P_ho))
        Q_cold_max = fc * (self._H(cold, T_hi, P_co) - H_ci)
        Q_max = min(Q_hot_max, Q_cold_max)

        def hot_out_T(Q):
            return self._T_at_H(hot, P_ho, H_hi - Q / fh, T_ci, T_hi)

        def cold_out_T(Q):
            return self._T_at_H(cold, P_co, H_ci + Q / fc, T_ci, T_hi)

        if self.mode == 'ua':
            UA = self.UA

            def f(Q):
                return Q - UA * self.F * self._lmtd(T_hi, hot_out_T(Q), T_ci, cold_out_T(Q))
            Q = float(brentq(f, 0.0, Q_max * (1.0 - 1e-12), xtol=1e-12, rtol=1e-13, maxiter=200))
        elif self.mode == 'hot_out':
            T = float(self.T_hot_out)
            if not T_ci <= T < T_hi:
                raise ValueError(f'T_hot_out must be in [{T_ci:.2f}, {T_hi:.2f}) K.')
            Q = fh * (H_hi - self._H(hot, T, P_ho))
        elif self.mode == 'cold_out':
            T = float(self.T_cold_out)
            if not T_ci < T <= T_hi:
                raise ValueError(f'T_cold_out must be in ({T_ci:.2f}, {T_hi:.2f}] K.')
            Q = fc * (self._H(cold, T, P_co) - H_ci)
        elif self.mode == 'duty':
            Q = float(self.duty)
        else:
            Q = float(self.efficiency) * Q_max
        if Q > Q_max * (1.0 + 1e-9):
            raise ValueError(f'Duty {Q:.3f} kW exceeds the maximum {Q_max:.3f} kW possible with these inlet '
                             'temperatures.')
        Q = min(Q, Q_max)

        T_ho = float(self.T_hot_out) if self.mode == 'hot_out' else hot_out_T(Q)
        T_co = float(self.T_cold_out) if self.mode == 'cold_out' else cold_out_T(Q)
        st_ho, st_co = hot.at(T=T_ho, P=P_ho), cold.at(T=T_co, P=P_co)
        Q_hot = fh * (H_hi - st_ho.props.H_total_mix)
        Q_cold = fc * (st_co.props.H_total_mix - H_ci)

        L = self._lmtd(T_hi, T_ho, T_ci, T_co)
        if L <= 0.0:
            warn.append('LMTD is not positive: the temperatures cross (a co-current exchanger cannot do this duty).')
        UA = Q / (self.F * L) if L > 0.0 else math.inf
        U, A = self.U, (UA / self.U if self.U is not None else None)
        if self.mode == 'ua':
            A = self.A

        def C(inlet, st, H_in, f_):
            dT = st.T - inlet.T
            if abs(dT) < 1e-6:
                return f_ * st.props.Cp_total_mix
            return f_ * abs(st.props.H_total_mix - H_in) / abs(dT)
        C_h, C_c = C(hot, st_ho, H_hi, fh), C(cold, st_co, H_ci, fc)
        NTU = UA / min(C_h, C_c)

        S_gen = (fh * (st_ho.props.S_total_mix - hot.props.S_total_mix)
                 + fc * (st_co.props.S_total_mix - cold.props.S_total_mix))
        if S_gen < -1e-9 * max(1.0, abs(Q)):
            warn.append(f'Entropy generation {S_gen:.3e} kW/K is negative: check the properties.')

        phases = {hot.props.phase, st_ho.props.phase, cold.props.phase, st_co.props.phase}
        if len({hot.props.phase, st_ho.props.phase}) > 1 or len({cold.props.phase, st_co.props.phase}) > 1 \
                or 'two_phase' in phases:
            warn.append('A stream changes phase: LMTD assumes constant heat capacities; use UA_profile.')

        profile, UA_prof, pinch = self._profile(hot, cold, P_ho, P_co, Q, H_hi, H_ci, fh, fc, T_hi, T_ci, T_ho, T_co)
        if pinch is not None and pinch < 0.0:
            warn.append('The T-Q profile crosses (negative dT inside the exchanger).')

        return HeatExchangerResult(
            flow=self.flow, mode=self.mode, Q_kW=Q, Q_hot_kW=Q_hot, Q_cold_kW=Q_cold,
            balance_error=abs(Q_hot - Q_cold) / max(abs(Q_hot), 1e-300), Q_max_kW=Q_max, efficiency=Q / Q_max,
            T_hot_in=T_hi, T_hot_out=T_ho, T_cold_in=T_ci, T_cold_out=T_co, P_hot_out=P_ho, P_cold_out=P_co,
            LMTD=L, F=self.F, UA=UA, U=U, A=A, C_hot=C_h, C_cold=C_c, NTU=NTU, S_gen_kWK=S_gen,
            exergy_destroyed_kW=self.T_ref * S_gen, pinch=pinch, UA_profile=UA_prof, profile=profile,
            hot_out=st_ho, cold_out=st_co, warnings=warn)

    def _profile(self, hot, cold, P_ho, P_co, Q, H_hi, H_ci, fh, fc, T_hi, T_ci, T_ho, T_co):
        """T-Q profile in n segments of equal duty, with the pressures varying linearly along each side."""
        n = self.n_profile
        if n < 2:
            return [], None, None
        pts = []
        for k in range(n + 1):
            x = k / n                                   # fraction of the duty from the hot inlet end
            Qk = x * Q
            Ph = hot.P + (P_ho - hot.P) * x
            Th = T_hi if k == 0 else (T_ho if k == n else self._T_at_H(hot, Ph, H_hi - Qk / fh, T_ci, T_hi))
            if self.flow == 'counter':                  # cold enters at the hot-outlet end
                Qc = Q - Qk                             # duty still to be received by the cold side upstream
                Pc = cold.P + (P_co - cold.P) * (1.0 - x)
            else:
                Qc, Pc = Qk, cold.P + (P_co - cold.P) * x
            if Qc <= 0.0:
                Tc = T_ci
            elif Qc >= Q:
                Tc = T_co
            else:
                Tc = self._T_at_H(cold, Pc, H_ci + Qc / fc, T_ci, T_hi)
            pts.append(ProfilePoint(x, Qk, Th, Tc, Th - Tc))
        pinch = min(p.dT for p in pts)
        UA_prof = None
        if pinch > 0.0:
            UA_prof = sum((pts[i + 1].Q - pts[i].Q) * 0.5 * (1.0 / pts[i].dT + 1.0 / pts[i + 1].dT)
                          for i in range(n))
        return pts, UA_prof, pinch