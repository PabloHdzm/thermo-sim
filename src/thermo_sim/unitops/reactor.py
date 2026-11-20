"""Reactors built on `Stream`: stoichiometric (`StoichReactor`) and chemical equilibrium
(`EquilibriumReactor`).

The outlet composition is set by the reaction stoichiometry and either the conversion of a key
component or the extent of reaction; no chemical equilibrium is evaluated. The outlet is at
`T_out` (isothermal at the inlet temperature if omitted) and at P_in - dP. The duty is the heat
added to the stream, fn_out * H_out - fn_in * H_in [kW], so formation enthalpies are part of H
and the heat of reaction appears in the duty (negative for an exothermic reaction).

Calculation modes (all reactors), as in DWSIM: mode='isothermal' (outlet at the inlet temperature), mode='defined'
(outlet at the given `T_out`; the default when `T_out` is given) and mode='adiabatic'. In adiabatic mode T_out is the
unknown and the duty is zero; T_out solves the energy
balance fn_out * H_out(T_out) = fn_in * H_in, with the heat of reaction appearing as a temperature
change. `T_out` cannot be given in that mode.

Units: T [K], P [bar], flows [kmol/s], H [kJ/kmol], duty [kW].
`inlet` must provide `.at(T=..., P=..., fn=..., comp_mol=...)`, `.fn`, `.comp_mol` (active
components) and `.database` (all components, so products absent from the feed can be created).
"""
from __future__ import annotations

import warnings
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
    mode: str = 'isothermal'

    def report(self) -> str:
        sep = '-' * 60
        L = [sep, f"{self.label} ({self.mode}): T {self.T_out:.2f} K, P {self.P_out:.3f} bar, "
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


def _check_mode(mode, T_out) -> str:
    """Resolve the calculation mode as in DWSIM: 'isothermal' (outlet at the inlet temperature), 'defined' (outlet
    at T_out) or 'adiabatic' (T_out is solved). mode=None picks 'defined' if T_out is given, else 'isothermal'."""
    if mode is None:
        mode = 'defined' if T_out is not None else 'isothermal'
    mode = str(mode).lower()
    if mode in ('defined_temperature', 'defined temperature', 'temperature'):
        mode = 'defined'
    if mode not in ('isothermal', 'defined', 'adiabatic'):
        raise ValueError("mode must be 'isothermal', 'defined' or 'adiabatic'.")
    if mode == 'adiabatic' and T_out is not None:
        raise ValueError("T_out cannot be given in adiabatic mode (it is the unknown).")
    if mode == 'isothermal' and T_out is not None:
        raise ValueError("T_out is given: use mode='defined' (isothermal keeps the inlet temperature).")
    if mode == 'defined' and T_out is None:
        raise ValueError("mode='defined' needs T_out.")
    return mode


def _solve_adiabatic_T(f, T0: float, dT0: float = 50.0, T_min: float = 150.0, T_max: float = 3000.0) -> float:
    """Root of the energy-balance residual f(T) = fn_out H_out(T) - fn_in H_in [kW], increasing in T.
    Brackets by doubling steps from T0 (towards lower T when f(T0) > 0), then brentq. f may return NaN where
    the model fails (e.g. a second phase), which ends the search with an error."""
    a, fa = float(T0), f(T0)
    if not np.isfinite(fa):
        raise ValueError(f"The energy balance cannot be evaluated at the inlet temperature {T0:.2f} K.")
    if fa == 0.0:
        return a
    d, step = (1.0 if fa < 0 else -1.0), float(dT0)
    for _ in range(12):
        b = min(max(a + d * step, T_min), T_max)
        if b == a:
            break
        fb = f(b)
        if not np.isfinite(fb):
            raise ValueError(f"The adiabatic outlet temperature is outside the range of the model "
                             f"(no valid state at {b:.1f} K).")
        if fa * fb <= 0.0:
            return float(brentq(f, min(a, b), max(a, b), xtol=1e-10, rtol=1e-13, maxiter=200))
        a, fa = b, fb
        step *= 2.0
    raise ValueError("Could not bracket the adiabatic outlet temperature (energy balance never changes sign "
                     f"between {T_min:.0f} and {T_max:.0f} K).")


def _outlet(inlet, comp, stoich, xi, n_in, T_out, P_out, H_flow=None):
    """Outlet stream and moles for extent xi [kmol/s]. With H_flow [kW] instead of T_out the outlet T is solved
    (PH flash)."""
    n_out = {c: n_in[c] + stoich.get(c, 0.0) * xi for c in comp}
    if min(n_out.values()) < -1e-15:
        raise ValueError("Extent consumes more than the available reactant.")
    n_out = {c: max(v, 0.0) for c, v in n_out.items()}
    fn_out = sum(n_out.values())
    cm = {c: v / fn_out for c, v in n_out.items()}
    if H_flow is None:
        out = inlet.at(T=T_out, P=P_out, fn=fn_out, comp_mol=cm)
    else:
        out = inlet.at(P=P_out, fn=fn_out, comp_mol=cm, H_flow=H_flow, T_guess=float(inlet.T))
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
                 extent: float | None = None, T_out: float | None = None, dP: float = 0.0,
                 mode: str | None = None):
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
        self.mode = _check_mode(mode, T_out)

    def run(self, inlet) -> ReactorResult:
        comp = _names(inlet)
        stoich = _check_stoich(self.stoich, comp)
        key = None if self.key is None else str(self.key).lower()
        n_in = _inlet_moles(inlet, comp)
        if self.conversion is not None:
            xi = self.conversion * n_in[key] / -stoich[key]
        else:
            xi = float(self.extent)
        P_out = _out_pressure(inlet, self.dP)
        H_in = float(inlet.fn) * inlet.props.H_total_mix
        if self.mode == 'adiabatic':          # composition is fixed by xi: one PH flash gives T_out
            out, n_out, fn_out = _outlet(inlet, comp, stoich, xi, n_in, None, P_out, H_flow=H_in)
            T_out = float(out.T)
        else:
            T_out = float(inlet.T) if self.T_out is None else float(self.T_out)
            out, n_out, fn_out = _outlet(inlet, comp, stoich, xi, n_in, T_out, P_out)
        dH_flow = fn_out * out.props.H_total_mix - H_in
        _warn_beyond_equilibrium(inlet, stoich, n_out, fn_out, T_out, P_out)
        conv = (xi * -stoich[key] / n_in[key]) if key else float('nan')
        return ReactorResult(extent=xi, conversion=conv, T_in=float(inlet.T), T_out=T_out,
                             P_in=float(inlet.P), P_out=P_out, n_in=n_in, n_out=n_out,
                             dH_flow_kW=dH_flow, duty_kW=dH_flow, outlet=out, mode=self.mode)


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


def _warn_beyond_equilibrium(inlet, stoich, n_out, fn_out, T, P_out, margin: float = 0.01) -> None:
    """Warn when the outlet composition lies beyond chemical equilibrium at T_out, i.e. the specified extent
    would make the reaction run backwards. Ideal-gas test (phi = 1) with K(T) from the formation data."""
    try:
        if any(n_out[c] <= 0.0 for c in stoich):
            return
        R = float(inlet._eos.R)
        lnK = reaction_thermo(inlet.database, stoich, T, R)['ln_K']
        lnQ = sum(v * np.log(n_out[c] / fn_out) for c, v in stoich.items()) \
            + sum(stoich.values()) * np.log(P_out / P_STD)
        if not (np.isfinite(lnK) and np.isfinite(lnQ)):
            return
    except Exception:                                     # missing formation data: nothing to compare
        return
    if lnQ > lnK + margin:
        warnings.warn(f"The specified conversion lies beyond chemical equilibrium at T_out = {T:.1f} K "
                      f"(ln Q = {lnQ:.3f} > ln K = {lnK:.3f}, ideal-gas test): the reaction would run backwards.",
                      UserWarning, stacklevel=3)


class EquilibriumReactor:
    """Chemical equilibrium of one reaction in a single vapor phase.

    stoich: {component: coefficient}, negative for reactants, positive for products. The
    extent solves  prod_i (phi_i y_i P / P0)^nu_i = K(T)  with P0 = 1 bar, K from the formation
    Gibbs energies and the ideal-gas Cp (`reaction_thermo`), and phi_i from the stream's EOS at
    the outlet composition (successive substitution on phi). The reaction can run backwards if
    the inlet contains products. The outlet is at `T_out` (inlet T if omitted) and P_in - dP.
    The duty is fn_out H_out - fn_in H_in, as in `StoichReactor`.
    mode='adiabatic': T_out solves fn_out H_out(T_out) = fn_in H_in (K(T), phi and the extent are re-solved
    at every trial T); the equilibrium of an exothermic reaction shifts back as T rises, so the root is unique.

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
                 fugacity: bool = True, mode: str | None = None):
        if P_ref <= 0:
            raise ValueError("P_ref must be positive.")
        self.mode = _check_mode(mode, T_out)
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

    def _at_T(self, inlet, ctx, T_out):
        """Equilibrium at fixed T_out: extent, outlet stream and the quantities of the result."""
        stoich, n_in, fn_in, comp, P_out, R, lo, hi = (ctx[k] for k in
                                                       ('stoich', 'n_in', 'fn_in', 'comp', 'P_out', 'R', 'lo', 'hi'))
        if self.lnK is None:
            th = reaction_thermo(inlet.database, stoich, T_out, R)
        else:
            th = dict(dH0=float('nan'), dS0=float('nan'), dG0=float('nan'),
                      ln_K=float(self.lnK(T_out)))
        lnK = th['ln_K']
        lnPratio = float(np.log(P_out / self.P_ref))

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
        return dict(xi=xi, out=out, n_out=n_out, fn_out=fn_out, th=th, lnK=lnK, lnphi=lnphi, it=it,
                    lnPratio=lnPratio)

    def run(self, inlet) -> EquilibriumResult:
        comp = _names(inlet)
        stoich = _check_stoich(self.stoich, comp)
        stoich = {c: v for c, v in stoich.items() if v != 0.0}
        n_in = _inlet_moles(inlet, comp)
        fn_in = float(inlet.fn)
        P_out = _out_pressure(inlet, self.dP)
        R = float(inlet._eos.R)
        lo, hi = self._bracket(stoich, n_in)
        ctx = dict(stoich=stoich, n_in=n_in, fn_in=fn_in, comp=comp, P_out=P_out, R=R, lo=lo, hi=hi)
        H_in = fn_in * inlet.props.H_total_mix

        if self.mode == 'adiabatic':
            def f(T):
                try:
                    st = self._at_T(inlet, ctx, T)
                except (ValueError, RuntimeError, FloatingPointError):
                    return float('nan')
                return st['fn_out'] * st['out'].props.H_total_mix - H_in
            T_out = _solve_adiabatic_T(f, float(inlet.T))
        else:
            T_out = float(inlet.T) if self.T_out is None else float(self.T_out)
        st = self._at_T(inlet, ctx, T_out)
        xi, out, n_out, fn_out = st['xi'], st['out'], st['n_out'], st['fn_out']
        y = {c: n_out[c] / fn_out for c in stoich}
        ln_Ky = sum(v * np.log(y[c]) for c, v in stoich.items())
        ln_Kphi = sum(v * st['lnphi'][c] for c, v in stoich.items())
        dnu = sum(stoich.values())
        dH_flow = fn_out * out.props.H_total_mix - H_in
        key = min(stoich, key=lambda c: stoich[c])        # most negative coefficient: reported conversion
        conv = (xi * -stoich[key] / n_in[key]) if n_in[key] > 0 else float('nan')
        th = st['th']
        return EquilibriumResult(
            extent=xi, conversion=conv, T_in=float(inlet.T), T_out=T_out, P_in=float(inlet.P),
            P_out=P_out, n_in=n_in, n_out=n_out, dH_flow_kW=dH_flow, duty_kW=dH_flow, outlet=out,
            dG0_J_mol=th['dG0'], dH0_J_mol=th['dH0'], ln_K=st['lnK'], ln_K_phi=ln_Kphi,
            ln_K_y=ln_Ky + dnu * st['lnPratio'], iterations=st['it'], mode=self.mode)