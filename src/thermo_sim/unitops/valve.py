"""Isenthalpic expansion valve: outlet at a given pressure with the inlet enthalpy."""
from __future__ import annotations

from dataclasses import dataclass

from scipy.optimize import brentq

H_TOL = 1e-6          # kJ/kmol, accepted |H_out - H_in| of the PH flash


@dataclass
class ValveResult:
    P_in: float
    P_out: float
    T_in: float
    T_out: float
    H_in: float               # [kJ/kmol], formation enthalpy included
    H_out: float
    vapor_fraction_out: float
    outlet: object            # Stream at (P_out, H_in)

    @property
    def dT(self) -> float:
        return self.T_out - self.T_in

    def report(self) -> str:
        sep = '-' * 60
        return '\n'.join([
            sep,
            'Valve (isenthalpic)',
            sep,
            f"P in / out              : {self.P_in:.3f} / {self.P_out:.3f} bar",
            f"T in / out              : {self.T_in:.3f} / {self.T_out:.3f} K",
            f"H in / out              : {self.H_in:.2f} / {self.H_out:.2f} kJ/kmol",
            f"Vapor fraction (out)    : {self.vapor_fraction_out:.5f}",
            sep,
        ])


class Valve:
    """Adiabatic throttling to `P_out` [bar] without work: H_out = H_in (PH flash).
    Works across the saturation dome: a liquid inlet flashes into a two-phase outlet.
    `inlet` must provide `inlet.at(P=..., H_flow=...)` (H_flow in kW) and `inlet.at(P=..., V_frac=...)`.

    The PH flash solves T from H(T, P), which is continuous across the dome for mixtures. For a pure
    component T is fixed at the saturation temperature inside the dome and H(T) jumps, so that
    result does not reproduce H_in; the valve then solves the vapor fraction instead, with the
    VF-P flash (H is linear in the vapor fraction at saturation)."""

    def __init__(self, P_out: float):
        if P_out <= 0:
            raise ValueError("P_out must be positive.")
        self.P_out = float(P_out)

    def run(self, inlet) -> ValveResult:
        P_in = float(inlet.P)
        if self.P_out >= P_in:
            raise ValueError(f"P_out ({self.P_out} bar) must be below the inlet pressure ({P_in} bar).")
        H_in = inlet.props.H_total_mix
        out = self._flash_PH(inlet, H_in)
        return ValveResult(P_in=P_in, P_out=self.P_out, T_in=float(inlet.T), T_out=float(out.T),
                           H_in=H_in, H_out=out.props.H_total_mix,
                           vapor_fraction_out=float(out.props.V_frac), outlet=out)

    def _flash_PH(self, inlet, H_in: float):
        try:
            out = inlet.at(P=self.P_out, H_flow=inlet.fn * H_in, T_guess=float(inlet.T))
            if abs(out.props.H_total_mix - H_in) <= max(H_TOL, 1e-9 * abs(H_in)):
                return out
        except (ValueError, RuntimeError):
            pass
        # inside the dome of a pure component (or the T-root did not balance): solve the vapor fraction
        def f(beta: float) -> float:
            return inlet.at(P=self.P_out, V_frac=beta).props.H_total_mix - H_in

        f0, f1 = f(0.0), f(1.0)
        if f0 > 0.0 or f1 < 0.0:
            raise ValueError(f"The enthalpy {H_in:.2f} kJ/kmol is outside the saturation range "
                             f"[{f0 + H_in:.2f}, {f1 + H_in:.2f}] at P={self.P_out} bar.")
        beta = brentq(f, 0.0, 1.0, xtol=1e-12, rtol=1e-12, maxiter=100)
        return inlet.at(P=self.P_out, V_frac=beta)
