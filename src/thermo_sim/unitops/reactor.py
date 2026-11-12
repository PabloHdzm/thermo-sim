"""Stoichiometric (fixed conversion or fixed extent) reactor built on `Stream`.

The outlet composition is set by the reaction stoichiometry and either the conversion of a key
component or the extent of reaction; no chemical equilibrium is evaluated. The outlet is at
`T_out` (isothermal at the inlet temperature if omitted) and at P_in - dP. The duty is the heat
added to the stream, fn_out * H_out - fn_in * H_in [kW], so formation enthalpies are part of H
and the heat of reaction appears in the duty (negative for an exothermic reaction).

Units: T [K], P [bar], flows [kmol/s], H [kJ/kmol], duty [kW].
`inlet` must provide `.at(T=..., P=..., fn=..., comp_mol=...)`, `.fn`, `.comp_mol` (active
components) and `.database` (all components, so products absent from the feed can be created).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ReactorResult:
    extent: float                  # extent of reaction [kmol/s]
    conversion: float              # conversion of the key component
    T_in: float
    T_out: float
    P_in: float
    P_out: float
    n_in: dict                     # component -> kmol/s
    n_out: dict
    dH_flow_kW: float              # fn_out * H_out - fn_in * H_in
    duty_kW: float                 # heat added (negative = heat removed)
    outlet: object

    def report(self) -> str:
        sep = '-' * 60
        L = [sep, f"Stoichiometric reactor: T {self.T_out:.2f} K, P {self.P_out:.3f} bar, "
                  f"conversion {self.conversion:.5f}", sep,
             f"extent               : {self.extent:.6e} kmol/s"]
        for c in self.n_in:
            if self.n_in[c] == 0.0 and self.n_out[c] == 0.0:
                continue
            L.append(f"{c:12s} in/out : {self.n_in[c]:.6e} / {self.n_out[c]:.6e} kmol/s")
        L += [f"duty                 : {self.duty_kW:.4f} kW", sep]
        return '\n'.join(L)


class StoichReactor:
    """stoich: {component: coefficient}, negative for reactants, positive for products.
    Give exactly one of `conversion` (of `key`) or `extent` [kmol/s]."""

    def __init__(self, stoich: dict, conversion: float | None = None, key: str | None = None,
                 extent: float | None = None, T_out: float | None = None, dP: float = 0.0):
        if (conversion is None) == (extent is None):
            raise ValueError("Give exactly one of conversion or extent.")
        if conversion is not None:
            if key is None or stoich.get(key, 0.0) >= 0.0:
                raise ValueError("conversion needs `key`, a reactant of the stoichiometry.")
            if not 0.0 <= conversion <= 1.0:
                raise ValueError("conversion must be in [0, 1].")
        if dP < 0:
            raise ValueError("dP must be >= 0.")
        self.stoich, self.conversion, self.key = dict(stoich), conversion, key
        self.extent, self.T_out, self.dP = extent, T_out, float(dP)

    def run(self, inlet) -> ReactorResult:
        comp = inlet.database['Component'].astype(str).str.lower().tolist()
        stoich = {str(c).lower(): v for c, v in self.stoich.items()}
        key = None if self.key is None else str(self.key).lower()
        unknown = set(stoich) - set(comp)
        if unknown:
            raise ValueError(f"Components not in the database: {sorted(unknown)}.")
        n_in = {c: float(inlet.fn) * float(inlet.comp_mol.get(c, 0.0)) for c in comp}
        if self.conversion is not None:
            xi = self.conversion * n_in[key] / -stoich[key]
        else:
            xi = float(self.extent)
        n_out = {c: n_in[c] + stoich.get(c, 0.0) * xi for c in comp}
        if min(n_out.values()) < -1e-15:
            raise ValueError("Extent consumes more than the available reactant.")
        n_out = {c: max(v, 0.0) for c, v in n_out.items()}
        fn_out = sum(n_out.values())
        T_out = float(inlet.T) if self.T_out is None else float(self.T_out)
        P_out = float(inlet.P) - self.dP
        if P_out <= 0:
            raise ValueError("Pressure drop exceeds the inlet pressure.")
        out = inlet.at(T=T_out, P=P_out, fn=fn_out,
                       comp_mol={c: v / fn_out for c, v in n_out.items()})
        dH_flow = fn_out * out.props.H_total_mix - float(inlet.fn) * inlet.props.H_total_mix
        conv = (xi * -stoich[key] / n_in[key]) if key else float('nan')
        return ReactorResult(extent=xi, conversion=conv, T_in=float(inlet.T), T_out=T_out,
                             P_in=float(inlet.P), P_out=P_out, n_in=n_in, n_out=n_out,
                             dH_flow_kW=dH_flow, duty_kW=dH_flow, outlet=out)