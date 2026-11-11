"""Two-phase separator (flash drum) built on `Stream`.

One thermal specification, with the drum pressure from `P` or from the feed pressure minus `dP`:

* ``T``        isothermal flash (TP)
* ``duty_kW``  heat added to the stream [kW] (positive heats, negative cools);
               the temperature is solved by a PH flash
* ``V_frac``   molar vapor fraction; the temperature is solved by a VF-P flash

The vapor and liquid outlets are saturated streams built with V_frac = 1 and 0, so each has 
its own saturation temperatures. Their temperature is solved again; if it differs 
from the drum temperature by more than `T_tol` the flash may have landed on another
root of the phase envelope, and the drum raises instead of returning inconsistent streams.
If the feed is a single phase at drum conditions, the empty outlet is `None`.

Units: T [K], P [bar], flows [kmol/s], duty [kW].
"""
from __future__ import annotations

from dataclasses import dataclass

_VAPOR = ('vapor')


@dataclass
class DrumResult:
    T: float
    P: float
    V_frac: float
    duty_kW: float                  # heat added: fn * (H_drum - H_feed)
    vapor: object | None
    liquid: object | None
    drum: object                    # the stream at drum conditions (two-phase if applicable)
    mass_error: float               # |sum of outlet mass flows - feed| / feed
    energy_error_kW: float          # outlets' enthalpy flow - (feed enthalpy flow + duty)

    def report(self) -> str:
        sep = '-' * 60
        L = [sep, f"Flash drum: T {self.T:.3f} K, P {self.P:.3f} bar, "
                  f"vapor fraction {self.V_frac:.5f}, duty {self.duty_kW:.3f} kW", sep]
        for name, s in (('vapor', self.vapor), ('liquid', self.liquid)):
            if s is None:
                L.append(f"{name:7s}: none")
            else:
                L.append(f"{name:7s}: fn {s.fn:.5f} kmol/s   H {s.props.H_total_mix:.2f} kJ/kmol"
                         f"   S {s.props.S_total_mix:.4f} kJ/(kmol K)")
        L += [f"mass balance error  : {self.mass_error:.2e}",
              f"energy balance error: {self.energy_error_kW:.3e} kW", sep]
        return '\n'.join(L)


class FlashDrum:
    def __init__(self, T: float | None = None, duty_kW: float | None = None,
                 V_frac: float | None = None, P: float | None = None, dP: float = 0.0,
                 T_tol: float = 1e-3):
        if sum(x is not None for x in (T, duty_kW, V_frac)) != 1:
            raise ValueError("Give exactly one of T, duty_kW, V_frac.")
        if V_frac is not None and not 0.0 <= V_frac <= 1.0:
            raise ValueError("V_frac must be in [0, 1].")
        if P is not None and dP:
            raise ValueError("Give P or dP, not both.")
        if dP < 0:
            raise ValueError("dP must be >= 0.")
        self.T, self.duty_kW, self.V_frac = T, duty_kW, V_frac
        self.P, self.dP, self.T_tol = P, float(dP), T_tol

    def _drum_stream(self, feed, P: float):
        if self.T is not None:
            return feed.at(T=self.T, P=P)
        if self.duty_kW is not None:
            H_flow = feed.fn * feed.props.H_total_mix + self.duty_kW
            return feed.at(P=P, H_flow=H_flow, T_guess=feed.T)
        return feed.at(P=P, V_frac=self.V_frac, T_guess=feed.T)

    def _outlet(self, drum, comp: list, w, fn: float, V_frac: float, label: str):
        comp_mol = dict(zip(comp, (float(x) for x in w)))
        out = drum.at(fn=fn, comp_mol=comp_mol, V_frac=V_frac, T_guess=drum.T)
        if abs(out.T - drum.T) > self.T_tol:
            raise RuntimeError(
                f"Flash drum: the {label} outlet saturates at {out.T:.4f} K, not at the drum "
                f"temperature {drum.T:.4f} K (tolerance {self.T_tol} K); the saturation "
                "solver found another root of the phase envelope.")
        return out

    def run(self, feed) -> DrumResult:
        P = (feed.P - self.dP) if self.P is None else float(self.P)
        if P <= 0:
            raise ValueError("Drum pressure must be positive.")
        drum = self._drum_stream(feed, P)
        p = drum.props
        H_feed = feed.fn * feed.props.H_total_mix
        duty = drum.fn * p.H_total_mix - H_feed

        if p.phase in _VAPOR:
            V, vapor, liquid = 1.0, drum, None
        elif p.phase == 'liquid':
            V, vapor, liquid = 0.0, None, drum
        else:
            V = float(p.V_frac)
            comp = drum.component_data['Component'].tolist()
            vapor = self._outlet(drum, comp, p.yi, V * drum.fn, 1.0, 'vapor')
            liquid = self._outlet(drum, comp, p.xi, (1.0 - V) * drum.fn, 0.0, 'liquid')

        outs = [s for s in (vapor, liquid) if s is not None]
        fm_out = sum(s.fm for s in outs)
        H_out = sum(s.fn * s.props.H_total_mix for s in outs)
        return DrumResult(T=drum.T, P=P, V_frac=V, duty_kW=duty, vapor=vapor, liquid=liquid,
                          drum=drum, mass_error=abs(fm_out - feed.fm) / feed.fm,
                          energy_error_kW=H_out - (H_feed + duty))
