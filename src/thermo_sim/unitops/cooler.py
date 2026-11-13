"""Isobaric cooler/heater for vapor streams: outlet at a given temperature."""
from __future__ import annotations

from dataclasses import dataclass

_VAPOR = ('vapor')


@dataclass
class CoolerResult:
    T_in: float
    T_out: float
    P_in: float
    P_out: float
    dH: float                 # H_out - H_in [kJ/kmol] (negative when cooling)
    duty_kW: float            # heat added [kW]: negative when cooling
    outlet: object            # Stream at (T_out, P_out)


class Cooler:
    """Cools (or heats) a stream to `T_out` with an optional pressure drop `dP` [bar].
    Instead of `T_out`, `V_frac` (0 to 1) takes the outlet to that vapor fraction at P_out, with T solved
    (saturated vapor with V_frac=1, saturated liquid with V_frac=0); it implies `allow_condensation`.
    duty = fn * (H_out - H_in): heat added to the stream, negative when cooling. By default the outlet must stay
    vapor (compressor intercoolers); pass `allow_condensation=True` to accept liquid."""

    def __init__(self, T_out: float | None = None, dP: float = 0.0, allow_condensation: bool = False,
                 V_frac: float | None = None):
        if dP < 0:
            raise ValueError("dP must be >= 0.")
        if (T_out is None) == (V_frac is None):
            raise ValueError("Give exactly one of T_out or V_frac.")
        if V_frac is not None and not 0.0 <= V_frac <= 1.0:
            raise ValueError("V_frac must be in [0, 1].")
        self.T_out = None if T_out is None else float(T_out)
        self.V_frac = None if V_frac is None else float(V_frac)
        self.dP, self.allow_condensation = float(dP), allow_condensation or V_frac is not None

    def run(self, inlet) -> CoolerResult:
        P_out = float(inlet.P) - self.dP
        if P_out <= 0:
            raise ValueError("Pressure drop exceeds the inlet pressure.")
        out = inlet.at(T=self.T_out, P=P_out) if self.V_frac is None else inlet.at(V_frac=self.V_frac, P=P_out)
        if not self.allow_condensation and out.props.phase not in _VAPOR:
            raise ValueError(f"Cooler outlet is '{out.props.phase}' at T={out.T:.2f} K, "
                             f"P={P_out:.2f} bar; pass allow_condensation=True to accept it.")
        dH = out.props.H_total_mix - inlet.props.H_total_mix
        return CoolerResult(T_in=float(inlet.T), T_out=float(out.T), P_in=float(inlet.P),
                            P_out=P_out, dH=dH, duty_kW=inlet.fn * dH, outlet=out)