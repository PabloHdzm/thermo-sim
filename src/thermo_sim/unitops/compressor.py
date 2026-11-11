"""Single-stage compressor for real gases, built on `Stream`.

Two efficiency definitions, both applied to the real-gas enthalpy and entropy of the
stream (no ideal-gas shortcuts):

* ``mode='isentropic'``: the ideal outlet is found at constant entropy (PS flash by
  Newton iteration), and the real enthalpy rise is ΔH_is / η.
* ``mode='polytropic'``: the compression path is integrated in small pressure steps with
  dH = V dP / η_p (Heun predictor-corrector in ln P), using the real molar volume at each
  step. For an ideal gas with constant Cp this gives T2 = T1 (P2/P1)^(R / (Cp η_p)).

Units: T [K], P [bar], H [kJ/kmol], S [kJ/(kmol K)], flows [kmol/s], power [kW].
The compressor only handles vapor feeds; it raises if the stream (or the
isentropic outlet) is liquid or two-phase.

`inlet` must provide `inlet.at(T=..., P=...)`, which returns the same stream (flows,
composition, thermodynamic method) at another state.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

BAR_M3_TO_KJ = 100.0                  # 1 bar * 1 m3 = 100 kJ
_VAPOR = ('vapor')
_MODES = ('isentropic', 'polytropic')


@dataclass
class CompressorResult:
    mode: str
    P_in: float
    P_out: float
    T_in: float
    T_out: float
    T_out_is: float            # isentropic outlet temperature
    H_in: float
    H_out: float
    H_out_is: float
    S_in: float
    S_out: float
    dH_is: float               # isentropic enthalpy rise [kJ/kmol]
    dH: float                  # real enthalpy rise [kJ/kmol]
    eta: float                 # efficiency specified (isentropic or polytropic)
    eta_mec: float
    eta_is_equiv: float        # dH_is / dH (equals eta in isentropic mode)
    W_ind_kW: float            # indicated power = fluid enthalpy rise
    W_brake_kW: float          # brake power = indicated / eta_mec
    W_loss_kW: float
    outlet: object             # Stream at (T_out, P_out)
    n_steps: int | None = None

    def report(self) -> str:
        sep = '-' * 60
        eff = 'polytropic' if self.mode == 'polytropic' else 'isentropic'
        return '\n'.join([
            sep,
            f"Compressor ({self.mode}, {eff} efficiency {self.eta:.3f}, "
            f"mechanical {self.eta_mec:.3f})",
            sep,
            f"P in / out              : {self.P_in:.3f} / {self.P_out:.3f} bar",
            f"T in                    : {self.T_in:.3f} K",
            f"T out (isentropic)      : {self.T_out_is:.3f} K",
            f"T out (real)            : {self.T_out:.3f} K",
            f"dH isentropic           : {self.dH_is:.2f} kJ/kmol",
            f"dH real                 : {self.dH:.2f} kJ/kmol",
            f"equivalent eta_is       : {self.eta_is_equiv:.4f}",
            f"Indicated power         : {self.W_ind_kW:.2f} kW",
            f"Brake power             : {self.W_brake_kW:.2f} kW",
            sep,
        ])


class Compressor:
    def __init__(self, P_out: float, eta: float = 0.80, eta_mec: float = 1.0,
                 mode: str = 'isentropic', n_steps: int = 20,
                 tol_T: float = 1e-8, max_iter: int = 40):
        if mode not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}, got '{mode}'.")
        if not 0.0 < eta <= 1.0 or not 0.0 < eta_mec <= 1.0:
            raise ValueError("eta and eta_mec must be in (0, 1].")
        if n_steps < 1:
            raise ValueError("n_steps must be >= 1.")
        self.P_out, self.eta, self.eta_mec = float(P_out), float(eta), float(eta_mec)
        self.mode, self.n_steps = mode, int(n_steps)
        self.tol_T, self.max_iter = tol_T, max_iter

    # thermodynamic helpers

    @staticmethod
    def _state(inlet, T: float, P: float, where: str = ''):
        st = inlet.at(T=T, P=P)
        if st.props.phase not in _VAPOR:
            raise ValueError(
                f"Compressor needs vapor; got phase "
                f"'{st.props.phase}' at T={T:.2f} K, P={P:.2f} bar {where}.")
        return st

    def _solve_T(self, inlet, P: float, prop: str, target: float, T0: float):
        """T such that H(T, P) = target (prop='H') or S(T, P) = target (prop='S').
        Newton with the real Cp: dH/dT = Cp, dS/dT = Cp/T (both increasing in T)."""
        T = float(T0)
        for _ in range(self.max_iter):
            st = self._state(inlet, T, P, f'while solving {prop}')
            p = st.props
            if prop == 'H':
                f, slope = p.H_total_mix - target, p.Cp_total_mix
            else:
                f, slope = p.S_total_mix - target, p.Cp_total_mix / T
            step = max(-100.0, min(100.0, -f / slope))
            T += step
            if abs(step) < self.tol_T:
                return T, st
        raise RuntimeError(f"Compressor: {prop} solve did not converge at P={P:.3f} bar "
                           f"(last T={T:.2f} K).")

    @staticmethod
    def _v_molar(st) -> float:
        """Molar volume [m3/kmol] from the stream's mass density."""
        return (st.fm / st.fn) / st.props.dens_mix

    # model

    def run(self, inlet) -> CompressorResult:
        P1, P2 = float(inlet.P), self.P_out
        if P2 <= P1:
            raise ValueError(f"P_out ({P2} bar) must exceed the inlet pressure ({P1} bar).")
        st1 = self._state(inlet, inlet.T, P1, '(suction)')
        H1, S1 = st1.props.H_total_mix, st1.props.S_total_mix

        # isentropic outlet (always computed: it is reported in both modes)
        T0 = inlet.T * (P2 / P1) ** 0.25
        T2s, st2s = self._solve_T(inlet, P2, 'S', S1, T0)
        H2s = st2s.props.H_total_mix
        dH_is = H2s - H1

        if self.mode == 'isentropic':
            dH = dH_is / self.eta
            T2, st2 = self._solve_T(inlet, P2, 'H', H1 + dH, T2s)
            n_used = None
        else:
            T2, st2 = self._integrate_polytropic(inlet, st1, P2)
            dH = st2.props.H_total_mix - H1
            n_used = self.n_steps

        outlet = inlet.at(T=T2, P=P2)
        W_ind = inlet.fn * dH
        W_brake = W_ind / self.eta_mec
        return CompressorResult(
            mode=self.mode, P_in=P1, P_out=P2, T_in=float(inlet.T), T_out=T2, T_out_is=T2s,
            H_in=H1, H_out=H1 + dH, H_out_is=H2s, S_in=S1, S_out=st2.props.S_total_mix,
            dH_is=dH_is, dH=dH, eta=self.eta, eta_mec=self.eta_mec,
            eta_is_equiv=dH_is / dH, W_ind_kW=W_ind, W_brake_kW=W_brake,
            W_loss_kW=W_brake - W_ind, outlet=outlet, n_steps=n_used)

    def _integrate_polytropic(self, inlet, st1, P2: float):
        """dH = V dP / eta_p along the path, Heun predictor-corrector in ln P."""
        P1 = float(inlet.P)
        h = math.log(P2 / P1) / self.n_steps
        k = BAR_M3_TO_KJ / self.eta
        P_k, T_k, st_k = P1, float(inlet.T), st1
        H_k = st1.props.H_total_mix
        for i in range(self.n_steps):
            P_n = P2 if i == self.n_steps - 1 else P1 * math.exp((i + 1) * h)
            g_k = self._v_molar(st_k) * P_k * k                    # dH/dlnP at the start
            T_p, st_p = self._solve_T(inlet, P_n, 'H', H_k + g_k * h, T_k)
            g_p = self._v_molar(st_p) * P_n * k                    # ... at the predictor
            H_n = H_k + 0.5 * h * (g_k + g_p)
            T_k, st_k = self._solve_T(inlet, P_n, 'H', H_n, T_p)
            P_k, H_k = P_n, H_n
        return T_k, st_k
