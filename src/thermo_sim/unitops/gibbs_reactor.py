"""Gibbs reactor: minimizes the Gibbs energy of one vapor phase at fixed T and P, subject to the element balance.

No reactions are given. The species are those of the feed (or the list `components`, which may include products
that are absent from the feed); atoms are conserved through an element matrix (formulas of the usual species are
built in, any other can be passed in `elements`). Solution: minimize

    G / RT = sum_i n_i [ G0_i(T) / RT + ln(phi_i P / P0) + ln(n_i / n_t) ]      subject to   A n = A n_in

in the null space of A with damped Newton steps that keep n_i > 0. phi_i come from the stream's EOS at the current
composition (successive substitution, as in `EquilibriumReactor`). G0_i(T) is the standard-state (ideal gas, 1 bar)
Gibbs energy built from the formation data and the ideal-gas Cp of the database, as in `reaction_thermo`: it differs
from the true Gibbs energy of formation by terms linear in the atom counts, which do not change the minimum because
the atoms are conserved.

Modes as in the other reactors: 'isothermal', 'defined' (T_out given, the default then) and 'adiabatic'.
mode='adiabatic': T_out solves fn_out H_out = fn_in H_in; the minimization is repeated at every trial T, started from
the previous solution. T_out cannot be given in that mode.

Limitations: one vapor phase (the outlet must be a vapor), no solids, no liquid. Units as in `reactor.py`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .reactor import (ReactorResult, _check_mode, _inlet_moles, _names, _out_pressure, _solve_adiabatic_T, P_STD,
                      reaction_thermo)

# Atom counts of the species of the database (lower-case names, as used by Stream.comp_mol)
FORMULAS = {
    'water': {'H': 2, 'O': 1}, 'carbonmonoxide': {'C': 1, 'O': 1}, 'carbondioxide': {'C': 1, 'O': 2},
    'hydrogen': {'H': 2}, 'nitrogen': {'N': 2}, 'oxygen': {'O': 2}, 'methane': {'C': 1, 'H': 4},
    'ethane': {'C': 2, 'H': 6}, 'propane': {'C': 3, 'H': 8}, 'butane': {'C': 4, 'H': 10},
    'pentane': {'C': 5, 'H': 12}, 'ammonia': {'N': 1, 'H': 3}, 'methanol': {'C': 1, 'H': 4, 'O': 1},
    'ethanol': {'C': 2, 'H': 6, 'O': 1}, 'benzene': {'C': 6, 'H': 6}, 'toluene': {'C': 7, 'H': 8},
    'acetone': {'C': 3, 'H': 6, 'O': 1},
}


@dataclass
class GibbsResult(ReactorResult):
    label: str = 'Gibbs reactor'
    conversions: dict = field(default_factory=dict)   # species -> fraction consumed (negative if produced)
    element_residue: float = float('nan')             # max |A n_out - A n_in| / fn_in
    elements_in: dict = field(default_factory=dict)   # element -> kmol/s
    iterations: int = 0

    def _extra_lines(self) -> list:
        return [f"element residue      : {self.element_residue:.3e}", f"iterations           : {self.iterations}"]


class GibbsReactor:
    """T_out: outlet T [K] (inlet T if omitted). dP: pressure drop [bar]. components: species allowed in the
    minimization (default: those present in the feed). elements: {species: {element: count}} to add or replace
    formulas. fugacity=False uses ideal-gas fugacity (phi = 1)."""

    def __init__(self, T_out: float | None = None, dP: float = 0.0, components: list | None = None,
                 elements: dict | None = None, tol: float = 1e-12, max_iter: int = 200,
                 phi_tol: float = 1e-12, fugacity: bool = True, mode: str | None = None):
        if dP < 0:
            raise ValueError("dP must be >= 0.")
        self.mode = _check_mode(mode, T_out)
        self.T_out, self.dP = T_out, float(dP)
        self.components = None if components is None else [str(c).lower() for c in components]
        self.formulas = dict(FORMULAS)
        for k, v in (elements or {}).items():
            self.formulas[str(k).lower()] = dict(v)
        self.tol, self.max_iter, self.phi_tol, self.fugacity = tol, int(max_iter), phi_tol, bool(fugacity)

    # ---------------------------------------------------------------- minimization at fixed phi
    @staticmethod
    def _nullspace(A):
        u, s, vt = np.linalg.svd(A)
        rank = int(np.sum(s > 1e-10 * s[0]))
        return vt[rank:].T

    def _minimize(self, g0, lnphi, lnPr, A, b, n_start, N):
        """Damped Newton in the null space. Returns n and the number of steps."""
        n = n_start.copy()
        if N.shape[1] == 0:
            return n, 0

        def F(n):
            nt = n.sum()
            return float(n @ (g0 + lnphi + lnPr + np.log(n / nt)))

        for it in range(1, self.max_iter + 1):
            nt = n.sum()
            mu = g0 + lnphi + lnPr + np.log(n / nt)
            gx = N.T @ mu
            if np.max(np.abs(gx)) <= self.tol * 1e3:                     # in units of RT per atom-scale
                return n, it
            Hn = np.diag(1.0 / n) - 1.0 / nt
            Hx = N.T @ Hn @ N
            dx = -np.linalg.solve(Hx, gx)
            dn = N @ dx
            neg = dn < 0
            amax = 1.0 if not neg.any() else min(1.0, 0.95 * float(np.min(-n[neg] / dn[neg])))
            a, f0 = amax, F(n)
            for _ in range(60):
                if F(n + a * dn) <= f0 + 1e-4 * a * float(gx @ dx) + 1e-15:
                    break
                a *= 0.5
            n = n + a * dn
            if a * np.max(np.abs(dn)) <= 1e-16 * nt:
                return n, it
        raise RuntimeError(f"Gibbs minimization did not converge in {self.max_iter} iterations.")

    # ---------------------------------------------------------------- run
    def _at_T(self, inlet, ctx, T_out, n_start):
        """Minimization at fixed T_out (outer loop on the fugacity coefficients). Returns n, outlet stream, iterations."""
        species, A, b, N = ctx['species'], ctx['A'], ctx['b'], ctx['N']
        P_out, R, lnPr, fn_in = ctx['P_out'], ctx['R'], ctx['lnPr'], ctx['fn_in']
        g0 = np.array([reaction_thermo(inlet.database, {c: 1.0}, T_out, R)['dG0'] for c in species]) / (R * T_out)
        n = n_start.copy()
        lnphi, n_old, out, it_tot = np.zeros(len(species)), None, None, 0
        for outer in range(1, self.max_iter + 1):
            n, k = self._minimize(g0, lnphi, lnPr, A, b, n, N)
            it_tot += k
            fn_out = float(n.sum())
            comp_mol = {c: float(v) / fn_out for c, v in zip(species, n)}
            out = inlet.at(T=T_out, P=P_out, fn=fn_out, comp_mol=comp_mol)
            if out.props.phase != 'vapor':
                raise ValueError(f"The Gibbs outlet is not a single vapor phase (phase = '{out.props.phase}' "
                                 f"at {T_out:.2f} K, {P_out:.2f} bar).")
            if not self.fugacity:
                break
            names = out.component_data['Component'].astype(str).str.lower().tolist()
            phi = dict(zip(names, np.asarray(out.props.phi_v, float)))
            lnphi = np.array([np.log(phi[c]) for c in species])
            if n_old is not None and np.max(np.abs(n - n_old)) <= self.phi_tol * fn_in:
                break
            n_old = n.copy()
        else:
            raise RuntimeError("The fugacity coefficients did not converge.")
        return n, out, it_tot

    def run(self, inlet) -> GibbsResult:
        comp = _names(inlet)
        n_all = _inlet_moles(inlet, comp)
        fn_in = float(inlet.fn)
        species = self.components or [c for c in comp if n_all[c] > 0.0]
        unknown = [c for c in species if c not in comp]
        if unknown:
            raise ValueError(f"Components not in the database: {unknown}.")
        missing = [c for c in species if c not in self.formulas]
        if missing:
            raise ValueError(f"No formula for {missing}: pass `elements={{name: {{'C': 1, ...}}}}`.")
        if any(n_all[c] > 0.0 and c not in species for c in comp):
            raise ValueError("The feed contains components that are not in `components`.")
        els = sorted({e for c in species for e in self.formulas[c]})
        A = np.array([[self.formulas[c].get(e, 0.0) for c in species] for e in els], float)
        n_in = np.array([n_all[c] for c in species], float)
        b = A @ n_in
        P_out = _out_pressure(inlet, self.dP)
        R = float(inlet._eos.R)
        ctx = dict(species=species, A=A, b=b, N=self._nullspace(A), P_out=P_out, R=R,
                   lnPr=float(np.log(P_out / P_STD)), fn_in=fn_in)

        n0 = np.maximum(n_in, 1e-9 * n_in.sum())                     # interior start on the element balance
        n0 = n0 + A.T @ np.linalg.lstsq(A @ A.T, b - A @ n0, rcond=None)[0]
        if n0.min() <= 0.0:
            raise ValueError("The element balance has no interior point with these species (a species that "
                             "contains an element absent from the feed cannot be formed).")

        H_in = fn_in * inlet.props.H_total_mix
        if self.mode == 'adiabatic':
            warm = {'n': n0}

            def f(T):
                try:
                    n, out, _ = self._at_T(inlet, ctx, T, warm['n'])
                except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError):
                    return float('nan')
                warm['n'] = n                                      # start of the next trial T (element balance holds)
                return float(n.sum()) * out.props.H_total_mix - H_in
            T_out = _solve_adiabatic_T(f, float(inlet.T))
        else:
            T_out = float(inlet.T) if self.T_out is None else float(self.T_out)
        n, out, it_tot = self._at_T(inlet, ctx, T_out, n0)
        fn_out = float(n.sum())

        n_out = {c: n_all[c] for c in comp}
        n_out.update({c: float(v) for c, v in zip(species, n)})
        conv = {c: (1.0 - n_out[c] / n_all[c]) if n_all[c] > 0 else float('nan') for c in species}
        dH = fn_out * out.props.H_total_mix - H_in
        return GibbsResult(extent=float('nan'), conversion=float('nan'), T_in=float(inlet.T), T_out=T_out,
                           P_in=float(inlet.P), P_out=P_out, n_in=n_all, n_out=n_out, dH_flow_kW=dH,
                           duty_kW=dH, outlet=out, conversions=conv,
                           element_residue=float(np.max(np.abs(A @ n - b)) / fn_in),
                           elements_in=dict(zip(els, b)), iterations=it_tot, mode=self.mode)