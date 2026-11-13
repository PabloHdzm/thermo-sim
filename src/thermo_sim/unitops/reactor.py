"""Reactors built on `Stream`: stoichiometric (`StoichReactor`) and chemical equilibrium
(`EquilibriumReactor`).

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

import numpy as np
from scipy.optimize import brentq

from ..stream import _cp_gas_coeffs, _cp_gas_limits, _int_ext, _int_over_T_ext

T_STD = 298.15            # K, reference temperature of the formation data
P_STD = 1.0               # bar, standard state of the formation data (ideal gas)


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
    label: str = 'Stoichiometric reactor'

    def report(self) -> str:
        sep = '-' * 60
        L = [sep, f"{self.label}: T {self.T_out:.2f} K, P {self.P_out:.3f} bar, "
                  f"conversion {self.conversion:.5f}", sep,
             f"extent               : {self.extent:.6e} kmol/s"]
        for c in self.n_in:
            if self.n_in[c] == 0.0 and self.n_out[c] == 0.0:
                continue
            L.append(f"{c:12s} in/out : {self.n_in[c]:.6e} / {self.n_out[c]:.6e} kmol/s")
        L += self._extra_lines()
        L += [f"duty                 : {self.duty_kW:.4f} kW", sep]
        return '\n'.join(L)

    def _extra_lines(self) -> list:
        return []


@dataclass
class EquilibriumResult(ReactorResult):
    label: str = 'Equilibrium reactor'
    dG0_J_mol: float = float('nan')        # standard Gibbs energy of reaction at T_out [J/mol]
    dH0_J_mol: float = float('nan')        # standard enthalpy of reaction at T_out [J/mol]
    ln_K: float = float('nan')             # ln of the thermodynamic equilibrium constant
    ln_K_phi: float = float('nan')         # sum(nu_i ln phi_i) at the solution
    ln_K_y: float = float('nan')           # sum(nu_i ln y_i) + sum(nu_i) ln(P/P0) at the solution
    iterations: int = 0

    def _extra_lines(self) -> list:
        return [f"ln K (T)             : {self.ln_K:.6f}   (dG0 {self.dG0_J_mol / 1000:.4f} kJ/mol, "
                f"dH0 {self.dH0_J_mol / 1000:.4f} kJ/mol)",
                f"ln K_phi / ln K_y    : {self.ln_K_phi:.6f} / {self.ln_K_y:.6f}",
                f"iterations           : {self.iterations}"]


def _names(inlet) -> list:
    return inlet.database['Component'].astype(str).str.lower().tolist()


def _inlet_moles(inlet, comp) -> dict:
    return {c: float(inlet.fn) * float(inlet.comp_mol.get(c, 0.0)) for c in comp}


def _check_stoich(stoich: dict, comp: list) -> dict:
    stoich = {str(c).lower(): float(v) for c, v in stoich.items()}
    unknown = set(stoich) - set(comp)
    if unknown:
        raise ValueError(f"Components not in the database: {sorted(unknown)}.")
    return stoich


def _outlet(inlet, comp, stoich, xi, n_in, T_out, P_out):
    """Outlet stream and moles for extent xi [kmol/s]."""
    n_out = {c: n_in[c] + stoich.get(c, 0.0) * xi for c in comp}
    if min(n_out.values()) < -1e-15:
        raise ValueError("Extent consumes more than the available reactant.")
    n_out = {c: max(v, 0.0) for c, v in n_out.items()}
    fn_out = sum(n_out.values())
    out = inlet.at(T=T_out, P=P_out, fn=fn_out, comp_mol={c: v / fn_out for c, v in n_out.items()})
    return out, n_out, fn_out


def _out_pressure(inlet, dP) -> float:
    P_out = float(inlet.P) - dP
    if P_out <= 0:
        raise ValueError("Pressure drop exceeds the inlet pressure.")
    return P_out


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
        comp = _names(inlet)
        stoich = _check_stoich(self.stoich, comp)
        key = None if self.key is None else str(self.key).lower()
        n_in = _inlet_moles(inlet, comp)
        if self.conversion is not None:
            xi = self.conversion * n_in[key] / -stoich[key]
        else:
            xi = float(self.extent)
        T_out = float(inlet.T) if self.T_out is None else float(self.T_out)
        P_out = _out_pressure(inlet, self.dP)
        out, n_out, fn_out = _outlet(inlet, comp, stoich, xi, n_in, T_out, P_out)
        dH_flow = fn_out * out.props.H_total_mix - float(inlet.fn) * inlet.props.H_total_mix
        conv = (xi * -stoich[key] / n_in[key]) if key else float('nan')
        return ReactorResult(extent=xi, conversion=conv, T_in=float(inlet.T), T_out=T_out,
                             P_in=float(inlet.P), P_out=P_out, n_in=n_in, n_out=n_out,
                             dH_flow_kW=dH_flow, duty_kW=dH_flow, outlet=out)


def reaction_thermo(database, stoich: dict, T: float, R: float, Tref: float = T_STD) -> dict:
    """Standard-state (ideal gas, 1 bar) reaction properties at T from the formation data
    (hf_298_k, gf_298_k in kJ/mol) and the ideal-gas Cp polynomials of the database:

        dH0(T) = dH0(298.15) + sum nu_i int Cp_i dT
        dS0(T) = (dH0 - dG0)(298.15) / 298.15 + sum nu_i int Cp_i / T dT
        dG0(T) = dH0(T) - T dS0(T),        ln K = -dG0 / (R T)

    Returns dict(dH0, dS0, dG0 [J/mol], ln_K). The same Cp (with its linear extrapolation
    outside the fitted range) is used as in Stream, so K(T) is consistent with the stream H and S."""
    cd = database.reset_index(drop=True)
    names = cd['Component'].astype(str).str.lower().to_numpy()
    nu = np.array([stoich.get(n, 0.0) for n in names], float)
    cd, nu = cd[nu != 0.0].reset_index(drop=True), nu[nu != 0.0]     # other rows may hold NaN
    C, (lo, hi) = _cp_gas_coeffs(cd), _cp_gas_limits(cd)
    hf = cd['hf_298_k'].to_numpy(float) * 1000.0
    gf = cd['gf_298_k'].to_numpy(float) * 1000.0
    dH298, dG298 = float(nu @ hf), float(nu @ gf)
    dH = dH298 + float(nu @ _int_ext(C, lo, hi, Tref, T))
    dS = (dH298 - dG298) / Tref + float(nu @ _int_over_T_ext(C, lo, hi, Tref, T))
    dG = dH - T * dS
    return dict(dH0=dH, dS0=dS, dG0=dG, ln_K=-dG / (R * T))


class EquilibriumReactor:
    """Chemical equilibrium of one reaction in a single vapor phase.

    stoich: {component: coefficient}, negative for reactants, positive for products. The
    extent solves  prod_i (phi_i y_i P / P0)^nu_i = K(T)  with P0 = 1 bar, K from the formation
    Gibbs energies and the ideal-gas Cp (`reaction_thermo`), and phi_i from the stream's EOS at
    the outlet composition (successive substitution on phi). The reaction can run backwards if
    the inlet contains products. The outlet is at `T_out` (inlet T if omitted) and P_in - dP.
    The duty is fn_out H_out - fn_in H_in, as in `StoichReactor`.

    Options (to reproduce other simulators or user-supplied equilibrium data):
      lnK       callable T[K] -> ln K, replaces the K(T) computed from the formation data.
      P_ref     standard-state pressure P0 [bar] in prod (f_i / P0)^nu_i = K (default 1.0; the
                formation data of the database are tabulated at 1 bar, 1 atm is 1.01325).
      fugacity  True (default): f_i = phi_i y_i P with phi from the EOS; False: f_i = y_i P
                (partial pressures, ideal gas).

    Limitations: one reaction, one vapor phase (the outlet must be a vapor), no solids.
    """

    def __init__(self, stoich: dict, T_out: float | None = None, dP: float = 0.0,
                 tol: float = 1e-12, max_iter: int = 100, lnK=None, P_ref: float = P_STD,
                 fugacity: bool = True):
        if P_ref <= 0:
            raise ValueError("P_ref must be positive.")
        self.lnK, self.P_ref, self.fugacity = lnK, float(P_ref), bool(fugacity)
        if dP < 0:
            raise ValueError("dP must be >= 0.")
        if not any(v < 0 for v in stoich.values()) or not any(v > 0 for v in stoich.values()):
            raise ValueError("The stoichiometry needs reactants (negative) and products (positive).")
        self.stoich, self.T_out, self.dP = dict(stoich), T_out, float(dP)
        self.tol, self.max_iter = tol, int(max_iter)

    @staticmethod
    def _bracket(stoich, n_in):
        up = [n_in[c] / -v for c, v in stoich.items() if v < 0]
        dn = [n_in[c] / v for c, v in stoich.items() if v > 0]
        hi, lo = min(up), -min(dn)
        if not hi - lo > 0:
            raise ValueError("No room for reaction: a reactant and a product are both absent "
                             "or exhausted.")
        return lo, hi

    def _solve_extent(self, stoich, n_in, fn_in, lnK, lnphi_term, lnPratio, lo, hi):
        """Extent for fixed phi: root of  sum nu ln n_i - dnu ln n_tot + sum nu ln phi
        + dnu ln(P/P0) - ln K = 0."""
        nu = {c: v for c, v in stoich.items()}
        dnu = sum(nu.values())

        def h(xi):
            ln_n = sum(v * np.log(max(n_in[c] + v * xi, 1e-300)) for c, v in nu.items())
            return ln_n - dnu * np.log(fn_in + dnu * xi) + lnphi_term + dnu * lnPratio - lnK

        d = 1e-14 * (hi - lo)
        return brentq(h, lo + d, hi - d, xtol=1e-16 * (hi - lo), rtol=4 * np.finfo(float).eps,
                      maxiter=500)

    def run(self, inlet) -> EquilibriumResult:
        comp = _names(inlet)
        stoich = _check_stoich(self.stoich, comp)
        stoich = {c: v for c, v in stoich.items() if v != 0.0}
        n_in = _inlet_moles(inlet, comp)
        fn_in = float(inlet.fn)
        T_out = float(inlet.T) if self.T_out is None else float(self.T_out)
        P_out = _out_pressure(inlet, self.dP)
        R = float(inlet._eos.R)

        if self.lnK is None:
            th = reaction_thermo(inlet.database, stoich, T_out, R)
        else:
            th = dict(dH0=float('nan'), dS0=float('nan'), dG0=float('nan'),
                      ln_K=float(self.lnK(T_out)))
        lnK = th['ln_K']
        lnPratio = float(np.log(P_out / self.P_ref))
        lo, hi = self._bracket(stoich, n_in)

        lnphi = {c: 0.0 for c in stoich}                  # ideal-gas start
        xi_old, out, it = None, None, 0
        for it in range(1, self.max_iter + 1):
            term = sum(v * lnphi[c] for c, v in stoich.items())
            xi = self._solve_extent(stoich, n_in, fn_in, lnK, term, lnPratio, lo, hi)
            out, n_out, fn_out = _outlet(inlet, comp, stoich, xi, n_in, T_out, P_out)
            if out.props.phase != 'vapor':
                raise ValueError(f"The equilibrium outlet is not a single vapor phase "
                                 f"(phase = '{out.props.phase}' at {T_out:.2f} K, {P_out:.2f} bar).")
            names = out.component_data['Component'].astype(str).str.lower().tolist()
            phi = dict(zip(names, np.asarray(out.props.phi_v, float)))
            if not self.fugacity:                        # partial pressures: phi = 1, one pass
                lnphi = {c: 0.0 for c in stoich}
                break
            lnphi = {c: float(np.log(phi[c])) for c in stoich}
            if xi_old is not None and abs(xi - xi_old) <= self.tol * (hi - lo):
                break
            xi_old = xi
        else:
            raise RuntimeError(f"Reaction equilibrium did not converge in {self.max_iter} iterations.")

        y = {c: n_out[c] / fn_out for c in stoich}
        ln_Ky = sum(v * np.log(y[c]) for c, v in stoich.items())
        ln_Kphi = sum(v * lnphi[c] for c, v in stoich.items())
        dnu = sum(stoich.values())
        dH_flow = fn_out * out.props.H_total_mix - fn_in * inlet.props.H_total_mix
        key = min(stoich, key=lambda c: stoich[c])        # most negative coefficient: reported conversion
        conv = (xi * -stoich[key] / n_in[key]) if n_in[key] > 0 else float('nan')
        return EquilibriumResult(
            extent=xi, conversion=conv, T_in=float(inlet.T), T_out=T_out, P_in=float(inlet.P),
            P_out=P_out, n_in=n_in, n_out=n_out, dH_flow_kW=dH_flow, duty_kW=dH_flow, outlet=out,
            dG0_J_mol=th['dG0'], dH0_J_mol=th['dH0'], ln_K=lnK, ln_K_phi=ln_Kphi,
            ln_K_y=ln_Ky + dnu * lnPratio, iterations=it)