"""`Stream`: a process stream (composition, flows, T, P) with full thermodynamic
properties computed from a cubic EOS.

Specify exactly one thermal condition: T (TP flash, or VF-P flash if `V_frac` is given)
or H_flow + T_guess (PH flash). Streams support algebra: a + b (adiabatic mixing),
a - b, f * a, a / f and split().
"""
from __future__ import annotations

import copy
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from .equilibrium import SolverOptions
from .methods import ThermoMethod, build_eos, get_method
from .properties import StreamProperties
from .stream_equilibrium import EquilibriumMixin
from .volume import build_phase_volume

# extensive properties, scaled when a stream is multiplied or divided
_EXTENSIVE = ('V', 'L', 'fn_liq', 'fn_gas', 'fm_liq', 'fm_gas', 'V_m', 'L_m', 'fv',
              'molar_flows', 'mass_flows')

_PROP_BASES = (('H', 'h'), ('S', 's'), ('Cp', 'cp'))   # (molar, mass)
_KINDS = ('ig', 'dep', 'total')
_SCOPES = ('mix', 'indiv')
_PHASES = ('gas', 'liq')


# Ideal-gas Cp polynomial: Cp = sum_k c_k T^k [J/(mol K)], coefficients in the columns
# A_Cp_Gas, B_Cp_Gas, C_Cp_Gas, ... of the data table. The degree is whatever columns exist,
# so a database with A..E (quartic) and one with A..G (sextic) both work.
_CP_GAS_LETTERS = "ABCDEFGH"


def _cp_gas_coeffs(cd: pd.DataFrame) -> np.ndarray:
    """(n_components, degree + 1) matrix of ideal-gas Cp coefficients (ascending powers of T)."""
    names = [f"{L}_Cp_Gas" for L in _CP_GAS_LETTERS]
    present = [n in cd.columns for n in names]
    n = present.index(False) if False in present else len(names)
    if n == 0:
        raise KeyError("component_data has no 'A_Cp_Gas' column.")
    if any(present[n:]):
        raise ValueError("The ideal-gas Cp columns must be contiguous (A, B, C, ...): "
                         f"found {[x for x, p in zip(names, present) if p]}.")
    return cd[names[:n]].to_numpy(dtype=float)


def _poly_cp(C: np.ndarray, T: float) -> np.ndarray:
    """Cp(T) = sum_k c_k T^k."""
    return sum(C[:, k] * T**k for k in range(C.shape[1]))


def _poly_int(C: np.ndarray, T_a: float, T_b: float) -> np.ndarray:
    """Integral of Cp dT from T_a to T_b."""
    return sum(C[:, k] / (k + 1) * (T_b**(k + 1) - T_a**(k + 1)) for k in range(C.shape[1]))


def _poly_int_over_T(C: np.ndarray, T_a: float, T_b: float) -> np.ndarray:
    """Integral of Cp/T dT from T_a to T_b."""
    return (C[:, 0] * np.log(T_b / T_a)
            + sum(C[:, k] / k * (T_b**k - T_a**k) for k in range(1, C.shape[1])))

def _cp_gas_limits(cd: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Fitted T interval [Tmin_Cp_Gas, Tmax_Cp_Gas] per component. Missing column or NaN
    means no limit (the polynomial is then used as is)."""
    n = len(cd)
    lo = cd['Tmin_Cp_Gas'].to_numpy(dtype=float) if 'Tmin_Cp_Gas' in cd.columns else np.full(n, -np.inf)
    hi = cd['Tmax_Cp_Gas'].to_numpy(dtype=float) if 'Tmax_Cp_Gas' in cd.columns else np.full(n, np.inf)
    return np.where(np.isnan(lo), -np.inf, lo), np.where(np.isnan(hi), np.inf, hi)


def _poly_dcp(C: np.ndarray, T) -> np.ndarray:
    """dCp/dT of the polynomial."""
    return sum((k * C[:, k] * T**(k - 1) for k in range(1, C.shape[1])), np.zeros(C.shape[0]))


def _poly_H(C: np.ndarray, T) -> np.ndarray:
    """Antiderivative of Cp (lower limit 0 K)."""
    return _poly_int(C, 0.0, T)


def _poly_S(C: np.ndarray, T) -> np.ndarray:
    """Antiderivative of Cp/T (lower limit 1 K, so the constant is fixed)."""
    return _poly_int_over_T(C, 1.0, T)


def _ext_parts(C, lo, hi, T):
    """Value and slope of the polynomial at the nearest point of the fitted interval."""
    Tc = np.clip(T, lo, hi)
    return Tc, _poly_cp(C, Tc), _poly_dcp(C, Tc)


def _cp_ext(C, lo, hi, T):
    """Cp: the polynomial inside [lo, hi]; outside, a straight line with the polynomial's
    value and slope at the nearest limit (the degree-6 polynomial diverges quickly there)."""
    Tc, c, s = _ext_parts(C, lo, hi, T)
    return c + s * (T - Tc)


def _H_ext(C, lo, hi, T):
    """Antiderivative of _cp_ext (continuous across the limits)."""
    Tc, c, s = _ext_parts(C, lo, hi, T)
    d = T - Tc
    return _poly_H(C, Tc) + c * d + 0.5 * s * d**2


def _S_ext(C, lo, hi, T):
    """Antiderivative of _cp_ext / T (continuous across the limits)."""
    Tc, c, s = _ext_parts(C, lo, hi, T)
    return _poly_S(C, Tc) + (c - s * Tc) * np.log(T / Tc) + s * (T - Tc)


def _int_ext(C, lo, hi, T_a, T_b):
    """Integral of _cp_ext dT from T_a to T_b."""
    return _H_ext(C, lo, hi, T_b) - _H_ext(C, lo, hi, T_a)


def _int_over_T_ext(C, lo, hi, T_a, T_b):
    """Integral of _cp_ext / T dT from T_a to T_b."""
    return _S_ext(C, lo, hi, T_b) - _S_ext(C, lo, hi, T_a)


class Stream(EquilibriumMixin):
    def __init__(self,
                 description:     str,
                 fm:              float | None,
                 comp_mass:       dict | list | np.ndarray | None,
                 fn:              float | None,
                 comp_mol:        dict | list | np.ndarray | None,
                 V_frac:          float | None,
                 T:               float | None,
                 P:               float,
                 Tref:            float,
                 Pref:            float,
                 gamma:           list | np.ndarray | None,
                 component_data:  pd.DataFrame,
                 kij:             pd.DataFrame | np.ndarray | None = None,
                 k_mode:          str = 'cubic',
                 method:          str | ThermoMethod = 'PR',
                 forced_cubic:    set[str] | None = None,
                 solver_options:  SolverOptions | None = None,
                 emit_warnings:   bool = True,
                 psat_factor:     float | None = None,
                 H_flow:          float | None = None,
                 T_guess:         float | None = None):
        """
        Flows: fm is the mass flow [kg/s] and fn the molar flow [kmol/s]. Give exactly one of
        (fn + comp_mol), (fm + comp_mass), (fm + comp_mol) or (fn + comp_mass).

        component_data: per-component property table (rows = components, same layout as the
        packaged database).

        Thermal specification (exactly one):
          * T                          -> TP flash (or VF-P if V_frac is given)
          * T=None + H_flow + T_guess  -> PH flash (T is solved; H_flow = fn * H_total_mix)

        method: a preset from METHODS ('PR', 'SRK' ...) or a ThermoMethod. 
        It defines the EOS form, alpha (global and per component), the
        effective critical constants, etc. E.g. method=METHODS['PR'].but(alpha='bm')
        kij: preferably a DataFrame indexed by component name (any capitalization).
        An ndarray is accepted only if it matches the ACTIVE components sorted by Tc.
        """
        if T is None:
            if H_flow is None or T_guess is None:
                raise ValueError("If T=None you must give H_flow and T_guess.")
            if V_frac is not None:
                raise ValueError("H_flow + V_frac over-specify the stream.")
        elif H_flow is not None:
            raise ValueError("Give T or H_flow, not both.")

        self.description = description
        self.fm = fm
        self.comp_mass = comp_mass
        self.fn = fn
        self.comp_mol = comp_mol
        self.gamma, self.forced_cubic = gamma, forced_cubic
        self.gamma_psat_factor = psat_factor
        self.component_data = component_data.copy(deep=True).sort_values(by='Tc (K)')
        self._database = self.component_data.copy()      # all components, before dropping inactive ones
        self.kij = kij
        self.k_mode = k_mode
        self.method = get_method(method)
        self.P = P
        self._solve_T = T is None
        self.T = float(T_guess) if T is None else float(T)
        self.V_frac_spec = V_frac
        self.Tref = Tref
        self.Pref = Pref
        self.num_components = len(component_data)
        self.solver_options = solver_options or SolverOptions()
        self.emit_warnings = emit_warnings

        self._props = StreamProperties()
        self._resolve_fractions()
        self._build_eos()
        if self._solve_T:
            self.z = self.component_data['frac_mol'].to_numpy(dtype=float)
            self.T = self._T_from_H(float(H_flow), self.T)
        self._equilibrium()
        self._phase_flows()
        self._compute()

    @property
    def V_frac(self) -> float:
        """Resulting vapor fraction (alias of props.V_frac)."""
        return self._props.V_frac

    # kij

    @staticmethod
    def _normalize_kij(kij, names) -> pd.DataFrame | None:
        """DataFrame with lower-case index/columns (or None)."""
        if kij is None:
            return None
        if isinstance(kij, pd.DataFrame):
            df = kij.astype(float).copy()
            df.index = [str(c).lower() for c in df.index]
            df.columns = [str(c).lower() for c in df.columns]
            return df
        arr = np.asarray(kij, float)
        n = len(names)
        if arr.shape != (n, n):
            raise ValueError(f"kij ndarray {arr.shape} != ({n}, {n}) active components. "
                             "Use a DataFrame indexed by component name.")
        return pd.DataFrame(arr, index=list(names), columns=list(names))

    @staticmethod
    def _kij_to_matrix(kij_df, names) -> np.ndarray:
        """(n, n) matrix aligned with `names`. Missing pair => 0. Must be symmetric."""
        n = len(names)
        M = np.zeros((n, n))
        if kij_df is None:
            return M

        def _get(a, b):
            if a in kij_df.index and b in kij_df.columns:
                v = kij_df.at[a, b]
                if np.isfinite(v):
                    return float(v)
            return None

        for i, a in enumerate(names):
            for j, b in enumerate(names):
                if i == j:
                    continue
                v = _get(a, b)
                if v is None:
                    v = _get(b, a)
                M[i, j] = 0.0 if v is None else v
        if not np.allclose(M, M.T, atol=1e-12):
            raise ValueError("kij is not symmetric for the active components.")
        return M

    @staticmethod
    def _merge_kij(streams) -> pd.DataFrame | None:
        """Merge the kij of several streams by name. Repeated pair: the first stream wins."""
        dfs = [c.kij for c in streams if c.kij is not None]
        if not dfs:
            return None
        out = dfs[0]
        for d in dfs[1:]:
            out = out.combine_first(d)
        return out

    @property
    def props(self) -> StreamProperties:
        return self._props

    @property
    def database(self) -> pd.DataFrame:
        """Property table of all components, including those inactive in this stream."""
        return self._database
    
    def at(self, T: float | None = None, P: float | None = None,
           description: str | None = None, *, fn: float | None = None,
           comp_mol: dict | None = None, V_frac: float | None = None,
           H_flow: float | None = None, T_guess: float | None = None) -> 'Stream':
        """Same thermodynamic setup at another state. Thermal specification (at most one):
        T (TP flash), V_frac (VF-P flash, T is solved), H_flow in kW (PH flash, T is solved).
        With none of them the current T is kept. `fn` and `comp_mol` replace the flow and the
        composition (used to build phase outlets of a flash drum)."""
        if sum(x is not None for x in (T, V_frac, H_flow)) > 1:
            raise ValueError("Give at most one of T, V_frac, H_flow.")
        guess = self.T if T_guess is None else T_guess
        if H_flow is not None:
            thermal = dict(T=None, V_frac=None, H_flow=H_flow, T_guess=guess)
        elif V_frac is not None:
            thermal = dict(T=guess, V_frac=V_frac)
        else:
            thermal = dict(T=self.T if T is None else T, V_frac=None)
        return Stream(description or self.description, fm=None, comp_mass=None,
                      fn=self.fn if fn is None else fn,
                      comp_mol=self.comp_mol if comp_mol is None else comp_mol,
                      P=self.P if P is None else P, Tref=self.Tref, Pref=self.Pref,
                      gamma=self.gamma, component_data=self._database, kij=self.kij,
                      k_mode=self.k_mode, method=self.method, forced_cubic=self.forced_cubic,
                      solver_options=self.solver_options, emit_warnings=self.emit_warnings,
                      psat_factor=self.gamma_psat_factor, **thermal)
    @property
    def MM(self):
        return self.component_data['MM'].to_numpy(dtype=float)

    def __repr__(self) -> str:
        p = self._props
        return (f"Stream('{self.description}', "
                f"T={self.T:.2f} K, P={self.P:.2f} bar, "
                f"fm={self.fm:.4g} kg/s, phase='{p.phase}')")

    def __str__(self) -> str:
        p = self._props
        return '\n'.join([
            f"Stream: {self.description}",
            f"  T={self.T:.2f} K   P={self.P:.2f} bar   phase={p.phase}",
            f"  fm={self.fm:.4g} kg/s   fn={self.fn:.4g} kmol/s",
            f"  TBubble={p.TBubble:.2f} K   TDew={p.TDew:.2f} K",
            f"  V_frac={p.V_frac:.4f}   dens_mix={p.dens_mix:.2f} kg/m3",
        ])

    def _warn_cp_range(self) -> None:
        """Warn if T is outside the fit range of the ideal-gas Cp (the Cp is extrapolated linearly)."""
        cd = self.component_data
        if 'Tmin_Cp_Gas' not in cd.columns or 'Tmax_Cp_Gas' not in cd.columns:
            return
        outside = ((self.T < cd['Tmin_Cp_Gas']) | (self.T > cd['Tmax_Cp_Gas'])).to_numpy()
        if outside.any():
            names = cd['Component'].to_numpy()[outside].tolist()
            self._warn(f"T={self.T:.1f} K is outside the fit range of the ideal-gas Cp of {names}; "
                                        "the Cp is extrapolated linearly (ideal H and S are less reliable).", 4)

    def _compute(self) -> None:
        self._warn_cp_range()
        self._prepare_phases()
        self._enthalpies()
        self._entropies()
        self._departure_properties()
        self._assemble_properties()
        self._density()
        self._viscosity()
        self._conductivity()

    # helpers

    def _composition_vector(self, comp, label: str) -> np.ndarray:
        """Convert comp (dict / list / ndarray / Series) to a vector aligned with
        component_data['Component'], validate it (NaN, negatives, unknown keys, length)
        and normalize it to sum 1."""
        self.component_data['Component'] = self.component_data['Component'].astype(str).str.lower()
        names = self.component_data['Component'].to_numpy()
        n = len(names)

        if isinstance(comp, dict):
            pos = {c: i for i, c in enumerate(names)}
            unknown = [c for c in comp if c not in pos]
            if unknown:
                raise ValueError(f"{label}: components not found in the database: {unknown}")
            v = np.zeros(n, dtype=float)
            for c, x in comp.items():
                v[pos[c]] = x
        else:
            v = np.asarray(comp, dtype=float).ravel()
            if len(v) != n:
                raise ValueError(
                    f"{label} has {len(v)} elements; the database has {n} components.")
            v = v.copy()

        if np.isnan(v).any():
            raise ValueError(f"{label} contains NaN.")
        if (v < 0).any():
            raise ValueError(f"{label} contains negative fractions.")

        total = v.sum()
        if total <= 0:
            raise ValueError(f"The {label} fractions sum to 0.")
        if abs(total - 1.0) > 1e-6 and self.emit_warnings:
            warnings.warn(
                f"{label} is not normalized (sum={total:.6f}). Normalizing automatically.",
                UserWarning)
        return v / total

    def _active_MM(self, active: np.ndarray) -> np.ndarray:
        """Full MM vector, validated only for the active components. NaNs in inactive
        components are ignored (they are never used: everything is indexed with `active`)."""
        MM = self.MM
        bad = active & ~(np.isfinite(MM) & (MM > 0))
        if bad.any():
            names = self.component_data['Component'].to_numpy()[bad].tolist()
            raise ValueError(
                f"Components with fraction > 0 but no valid molar mass (MM): {names}")
        return MM

    @staticmethod
    def _validate_flow(value, label: str) -> float:
        if value is None or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{label} must be a finite positive number (got: {value}).")
        return float(value)

    def _store_fractions(self, frac_mol, frac_mass, molar_flows, mass_flows, active) -> None:
        """Store the results; the dicts hold the active components only."""
        comps = self.component_data['Component'].to_numpy()[active]
        self.frac_mol = frac_mol
        self.comp_mol = dict(zip(comps, frac_mol[active]))
        self.comp_mass = dict(zip(comps, frac_mass[active]))
        self.component_data['frac_mol'] = frac_mol
        self.component_data['frac_mass'] = frac_mass
        self._props.molar_flows = molar_flows
        self._props.mass_flows = mass_flows

    # fractions

    def _resolve_fractions(self) -> None:
        has_fn = self.fn is not None
        has_fm = self.fm is not None
        has_comp_mol = self.comp_mol is not None
        has_comp_mass = self.comp_mass is not None

        if has_fn and has_comp_mol and not has_comp_mass:
            self._from_molar()                   # fn + comp_mol
        elif has_fm and has_comp_mass and not has_comp_mol:
            self._from_mass()                    # fm + comp_mass
        elif has_fm and has_comp_mol and not has_fn:
            self._from_fm_comp_mol()             # fm + comp_mol
        elif has_fn and has_comp_mass and not has_fm:
            self._from_fn_comp_mass()            # fn + comp_mass
        else:
            raise ValueError(
                "Invalid input combination. Provide exactly one of "
                "(fn+comp_mol), (fm+comp_mass), (fm+comp_mol) or (fn+comp_mass).")

        # drop inactive components
        mask = self.component_data['frac_mol'].to_numpy(dtype=float) > 0
        self.num_components = int(mask.sum())
        self.component_data = self.component_data[mask].reset_index(drop=True)

        # keep the vectors aligned with the filtered component_data
        self.frac_mol = self.frac_mol[mask]
        self._props.molar_flows = self._props.molar_flows[mask]
        self._props.mass_flows = self._props.mass_flows[mask]

        if self.gamma is not None and hasattr(self.gamma, '__len__') and not isinstance(self.gamma, dict):
            n_act = self.num_components
            if len(self.gamma) != n_act:
                raise ValueError(
                    f"gamma has {len(self.gamma)} elements. Expected {n_act} (active components).")

    def _from_molar(self) -> None:
        """fn + comp_mol -> fm, comp_mass"""
        fn = self._validate_flow(self.fn, "fn")
        frac_mol = self._composition_vector(self.comp_mol, "comp_mol")
        active = frac_mol > 0
        MM = self._active_MM(active)

        molar_flows = fn * frac_mol
        mass_flows = np.zeros_like(frac_mol)
        mass_flows[active] = molar_flows[active] * MM[active]

        self.fm = float(mass_flows.sum())
        frac_mass = mass_flows / self.fm

        self._store_fractions(frac_mol, frac_mass, molar_flows, mass_flows, active)

    def _from_mass(self) -> None:
        """fm + comp_mass -> fn, comp_mol"""
        fm = self._validate_flow(self.fm, "fm")
        frac_mass = self._composition_vector(self.comp_mass, "comp_mass")
        active = frac_mass > 0
        MM = self._active_MM(active)

        mass_flows = fm * frac_mass
        molar_flows = np.zeros_like(frac_mass)
        molar_flows[active] = mass_flows[active] / MM[active]

        self.fn = float(molar_flows.sum())
        frac_mol = molar_flows / self.fn

        self._store_fractions(frac_mol, frac_mass, molar_flows, mass_flows, active)

    def _from_fm_comp_mol(self) -> None:
        """fm + comp_mol -> fn, comp_mass"""
        fm = self._validate_flow(self.fm, "fm")
        frac_mol = self._composition_vector(self.comp_mol, "comp_mol")
        active = frac_mol > 0
        MM = self._active_MM(active)

        MM_mix = float(np.sum(frac_mol[active] * MM[active]))   # mixture molar mass (kg/kmol)
        self.fn = fm / MM_mix                                   # kmol/s

        molar_flows = self.fn * frac_mol
        mass_flows = np.zeros_like(frac_mol)
        mass_flows[active] = molar_flows[active] * MM[active]
        frac_mass = mass_flows / mass_flows.sum()

        self._store_fractions(frac_mol, frac_mass, molar_flows, mass_flows, active)

    def _from_fn_comp_mass(self) -> None:
        """fn + comp_mass -> fm, comp_mol"""
        fn = self._validate_flow(self.fn, "fn")
        frac_mass = self._composition_vector(self.comp_mass, "comp_mass")
        active = frac_mass > 0
        MM = self._active_MM(active)

        MM_mix = float(1.0 / np.sum(frac_mass[active] / MM[active]))   # harmonic mean
        self.fm = fn * MM_mix                                          # kg/s

        mass_flows = self.fm * frac_mass
        molar_flows = np.zeros_like(frac_mass)
        molar_flows[active] = mass_flows[active] / MM[active]
        frac_mol = molar_flows / molar_flows.sum()

        self._store_fractions(frac_mol, frac_mass, molar_flows, mass_flows, active)

    # EOS construction

    def _build_eos(self) -> None:
        cd = self.component_data

        def _get(standard: str, original: str) -> np.ndarray:
            if original in cd.columns:
                return cd[original].to_numpy(dtype=float)
            if standard in cd.columns:
                return cd[standard].to_numpy(dtype=float)
            raise KeyError(f"Neither column '{original}' nor '{standard}' was found.")

        df_eos = pd.DataFrame({
            'Component': cd['Component'].values,
            'Tc':        _get('Tc',    'Tc (K)'),
            'Pc':        _get('Pc',    'Pc (bar)'),
            'omega':     _get('omega', 'w'),
            'frac_mol':  cd['frac_mol'].to_numpy(dtype=float),
        })
        for col in ['A_Antoine_K', 'B_Antoine_K', 'C_Antoine_K',
                    'D_Antoine_K', 'E_Antoine_K', 'Vc (cm3/mol)', 'Vc (L/m)', 'Vc', 'Zc','Z_RA']:
            if col in cd.columns:
                df_eos[col] = cd[col].to_numpy(dtype=float)

        # optional per-component alpha parameters (if the database has them)
        for col in ['kappa1', 'c1', 'c2', 'c3']:
            if col in cd.columns:
                df_eos[col] = cd[col].to_numpy(dtype=float)
        names = [str(c).lower() for c in df_eos['Component']]

        # kij -> DataFrame by name (kept on the stream) + matrix for the EOS
        self.kij = self._normalize_kij(self.kij, names)
        kij_mat = self._kij_to_matrix(self.kij, names)

        # EOS = form + per-component alpha + constants, all described by self.method.
        # strict=False: the stream only knows its ACTIVE components, so an override
        # (e.g. {'hydrogen': 'gd'}) for an absent component simply does not apply.
        self._eos = build_eos(self.method, df_eos, kij_mat, strict=False)
        self._phase_volume = build_phase_volume(self.method, self._eos, self.component_data)
        p = self._props
        p.Vc, p.Zc, p.Zra, p.zra_source = self._eos.Vc, self._eos.Zc, self._eos.Zra, self._eos.zra_source
        p.liquid_volume_model = self.method.liquid_volume
        self.eos_notes = list(self._eos.notes)      # trace of substituted constants

        antoine_available = any(c in df_eos.columns
                                for c in ['A_Antoine_K', 'ANT_A', 'A_Antoine'])
        use_gamma = False
        if self.k_mode == 'gamma':
            if not antoine_available:
                warnings.warn("k_mode='gamma' without Antoine data. Using cubic.", RuntimeWarning)
            else:
                use_gamma = True
        elif self.k_mode == 'auto':
            Tc_arr = np.asarray(self._eos.Tc, dtype=float)
            Tr_mix = self.T / float(Tc_arr[df_eos['frac_mol'].to_numpy(dtype=float) > 0].mean())
            use_gamma = antoine_available and self.P < 15 and Tr_mix < 1

        self._build_phase_model(df_eos, use_gamma)

    # flows by phase

    def _phase_flows(self) -> None:
        p = self._props
        p.fn_liq = p.xi * p.L
        p.fn_gas = p.yi * p.V
        p.fm_liq = p.fn_liq * self.MM
        p.fm_gas = p.fn_gas * self.MM
        p.V_m = float(np.sum(p.fm_gas))
        p.L_m = float(np.sum(p.fm_liq))
        zeros = np.zeros_like(self.MM)
        if p.phase in ('vapor'):
            p.V_frac_m = p.fm_gas / p.V_m if p.V_m > 0 else zeros.copy()
            p.L_frac_m = zeros.copy()
        elif p.phase == 'liquid':
            p.V_frac_m = zeros.copy()
            p.L_frac_m = p.fm_liq / p.L_m if p.L_m > 0 else zeros.copy()
        else:
            p.V_frac_m = p.fm_gas / p.V_m if p.V_m > 0 else zeros.copy()
            p.L_frac_m = p.fm_liq / p.L_m if p.L_m > 0 else zeros.copy()

        p.V_frac_m_total = p.V_m / self.fm if self.fm > 0 else 0.0
        p.L_frac_m_total = p.L_m / self.fm if self.fm > 0 else 0.0

    # enthalpy and ideal-gas helpers

    @staticmethod
    def _int_cp_gas(cd: pd.DataFrame, T_a: float, T_b: float) -> pd.Series:
        """Integral of Cp_gas dT from T_a to T_b [J/mol]."""
        lo, hi = _cp_gas_limits(cd)
        return pd.Series(_int_ext(_cp_gas_coeffs(cd), lo, hi, T_a, T_b), index=cd.index)

    @staticmethod
    def _int_cp_liq(cd: pd.DataFrame, T_a: float, T_b: float) -> np.ndarray:
        return (cd['A_Cp_Liquid'] * (T_b - T_a)
                + cd['B_Cp_Liquid'] / 2.0 * (T_b**2 - T_a**2)
                + cd['C_Cp_Liquid'] / 3.0 * (T_b**3 - T_a**3)
                + cd['D_Cp_Liquid'] / 4.0 * (T_b**4 - T_a**4))

    @staticmethod
    def _int_cp_gas_over_T(cd: pd.DataFrame, T_a: float, T_b: float) -> np.ndarray:
        """Integral of Cp_gas/T dT from T_a to T_b [J/(mol K)]."""
        lo, hi = _cp_gas_limits(cd)
        return pd.Series(_int_over_T_ext(_cp_gas_coeffs(cd), lo, hi, T_a, T_b), index=cd.index)

    @staticmethod
    def _int_cp_liq_over_T(cd: pd.DataFrame, T_a: float, T_b: float) -> np.ndarray:
        """Integral of Cp_liq/T dT from T_a to T_b [J/(mol K)]."""
        return (cd['A_Cp_Liquid'] * np.log(T_b / T_a)
                + cd['B_Cp_Liquid']        * (T_b    - T_a)
                + cd['C_Cp_Liquid'] / 2.0  * (T_b**2 - T_a**2)
                + cd['D_Cp_Liquid'] / 3.0  * (T_b**3 - T_a**3))

    @staticmethod
    def _h_ig_mix(T: float, Tref: float,
                  cd: pd.DataFrame, x: np.ndarray,
                  phase: str, T_dew: float,
                  eos_liq: bool = False) -> float:
        """Ideal-gas H of the mixture [J/mol], weighted by x. Exact replica of what
        _enthalpies does for H_ig_indiv_*.

        phase='v':                  integral of Cp_gas (Tref -> T) + hf
        phase='l' and eos_liq:      same as 'v' (same ideal gas for both phases)
        phase='l' and gamma-phi:    Cp_gas integral (Tref -> T_dew) - Hvap(T_dew)
                                    + Cp_liq integral (T_dew -> T)
                                    (non-finite T_dew -> T_dew = T, as in _enthalpies)
        Stateless: only uses the data table and scalars.
        """
        hf = (cd['hf_298_k'] * 1000.0).to_numpy(dtype=float)

        if phase == 'v' or eos_liq:
            H_ig = (Stream._int_cp_gas(cd, Tref, T).to_numpy(dtype=float) + hf)
        else:
            T_d = T_dew if np.isfinite(T_dew) else T
            H_sat = Stream._int_cp_gas(cd, Tref, T_d).to_numpy(dtype=float)
            tau = np.clip(1.0 - T_d / cd['Tc (K)'], 0.0, 1.0)
            Hvap = (cd['A_HVav'] * tau**cd['n_HVav']).to_numpy(dtype=float) * 1000.0
            H_liq = Stream._int_cp_liq(cd, T_d, T).to_numpy(dtype=float)
            H_ig = H_sat - Hvap + H_liq + hf
        return float(np.dot(x, H_ig))

    # phase helpers

    def _gas_active(self) -> bool:
        return self._props.phase in ('vapor', 'two_phase')

    def _liq_active(self) -> bool:
        return self._props.phase in ('liquid', 'two_phase')

    def _phase_compositions(self):
        """Composition of each phase (if a phase has no moles, falls back to z)."""
        p = self._props
        yi = p.yi if p.yi.sum() > 0 else self.z
        xi = p.xi if p.xi.sum() > 0 else self.z
        return np.asarray(yi, float), np.asarray(xi, float)

    def _prepare_phases(self) -> None:
        """Set everything to nan before computing, so an absent phase stays nan and no
        value from a previous calculation is carried over."""
        p = self._props
        yi, xi = self._phase_compositions()
        n = len(self.z)
        p.MM_g = float(np.dot(yi, self.MM)) if self._gas_active() else np.nan
        p.MM_l = float(np.dot(xi, self.MM)) if self._liq_active() else np.nan
        for X, x in _PROP_BASES:
            for kind in _KINDS:
                for scope in _SCOPES:
                    for suffix in ('_gas', '_liq', ''):
                        for name in (X, x):
                            empty = np.full(n, np.nan) if scope == 'indiv' else np.nan
                            setattr(p, f'{name}_{kind}_{scope}{suffix}', empty)

    def _store_ig(self, X: str, arr_gas: np.ndarray, arr_liq: np.ndarray,
                  yi: np.ndarray, xi: np.ndarray, mixing=None) -> None:
        """Store the ideal-gas (molar) part of X for each phase.
        arr_*: value per component. mixing: optional function f(w) that adds the
        ideal mixing term (only for S)."""
        p = self._props
        for ph, arr, w, active in (('gas', arr_gas, yi, self._gas_active()),
                                   ('liq', arr_liq, xi, self._liq_active())):
            if not active:
                continue
            val = float(np.dot(arr, w))
            if mixing is not None:
                val += mixing(w)
            setattr(p, f'{X}_ig_indiv_{ph}', np.asarray(arr, float).copy())
            setattr(p, f'{X}_ig_mix_{ph}', val)

    # ideal-gas enthalpy and Cp

    def _enthalpies(self) -> None:
        cd = self.component_data
        p = self._props
        T = self.T
        eos_liq = self._liquid_model is None     # 'cubic' mode: EOS in both phases
        yi, xi = self._phase_compositions()

        Cp_gas = _cp_ext(_cp_gas_coeffs(cd), *_cp_gas_limits(cd), T)
        Cp_liq_dippr = (cd['A_Cp_Liquid'] + cd['B_Cp_Liquid'] * T
                        + cd['C_Cp_Liquid'] * T**2 + cd['D_Cp_Liquid'] * T**3
                        ).to_numpy(dtype=float)
        Cp_liq = Cp_gas if eos_liq else Cp_liq_dippr

        # informational (not part of the phase structure)
        p.Cp_liq_dippr = Cp_liq_dippr
        p.cp_liq_dippr = Cp_liq_dippr / self.MM

        T_dew_s = p.TDew if np.isfinite(p.TDew) else T
        tau = np.clip(1.0 - T_dew_s / cd['Tc (K)'], 0.0, 1.0)
        p.HVAP = (cd['A_HVav'] * tau**cd['n_HVav']).to_numpy(dtype=float)   # informational

        hf = (cd['hf_298_k'] * 1000).to_numpy(dtype=float)
        H_gas = self._int_cp_gas(cd, self.Tref, T).to_numpy(dtype=float) + hf
        if eos_liq:
            H_liq = H_gas.copy()
        else:                                    # gamma-phi mode
            H_sat = self._int_cp_gas(cd, self.Tref, T_dew_s).to_numpy(dtype=float)
            H_liq = (H_sat - p.HVAP * 1000.0
                     + self._int_cp_liq(cd, T_dew_s, T).to_numpy(dtype=float) + hf)

        self._store_ig('Cp', Cp_gas, Cp_liq, yi, xi)
        self._store_ig('H',  H_gas,  H_liq,  yi, xi)

    # ideal-gas entropy

    def _entropies(self) -> None:
        cd = self.component_data
        p = self._props
        T = self.T
        eos_liq = self._liquid_model is None
        yi, xi = self._phase_compositions()

        def _smix(x):                    # -R * sum(x ln x)  [J/(mol K)]
            m = x > 0
            return -self._eos.R * float(np.sum(x[m] * np.log(x[m])))

        # Formation data are tabulated for a 1 bar standard state; refer the formation entropy
        # to Pref: (n_gas_el - 1) * R * ln(Pref / 1 bar), zero for elements and for Pref = 1 bar.
        if 'n_gas_el' not in cd.columns:
            raise KeyError("component_data has no 'n_gas_el' column; regenerate the database "
                           "with scripts/build_db.py.")
        S_std = ((cd['n_gas_el'] - 1.0) * self._eos.R * np.log(self.Pref / 1.0)).to_numpy(dtype=float)
        S_gas = (self._int_cp_gas_over_T(cd, self.Tref, T).to_numpy(dtype=float)
                 + ((cd['hf_298_k'] - cd['gf_298_k']) * 1000 / 298.15).to_numpy(dtype=float)
                 + S_std)
        
        if eos_liq:
            S_liq = S_gas.copy()
        else:                                    # gamma-phi mode
            T_dew_s = p.TDew if np.isfinite(p.TDew) else T
            S_sat = self._int_cp_gas_over_T(cd, self.Tref, T_dew_s).to_numpy(dtype=float)
            tau = np.clip(1.0 - T_dew_s / cd['Tc (K)'], 0.0, 1.0)
            HVAP = (cd['A_HVav'] * tau**cd['n_HVav']).to_numpy(dtype=float)
            S_liq = (S_sat - HVAP * 1000.0 / T_dew_s
                     + self._int_cp_liq_over_T(cd, T_dew_s, T).to_numpy(dtype=float))

        self._store_ig('S', S_gas, S_liq, yi, xi, mixing=_smix)

    # departure (residual) properties

    def _departure_properties(self) -> None:
        p = self._props
        eos = self._eos
        if not hasattr(eos, 'residual_properties'):
            return
        n = len(self.z)
        yi, xi = self._phase_compositions()
        dS_P = -self._eos.R * np.log(self.P / self.Pref)

        def _dep_indiv(phase):
            # Computed for ALL components, whether or not they have moles in the phase:
            # a pure component's property does not depend on the composition, and this
            # avoids nan (e.g. argon at zero fraction) when weighting.
            H, S, Cp = (np.full(n, np.nan) for _ in range(3))
            for i in range(n):
                x_pure = np.zeros(n); x_pure[i] = 1.0
                try:
                    r = eos.residual_properties(self.T, self.P, x_pure, phase)
                    H[i], S[i], Cp[i] = r['H_res'], r['S_res'] + dS_P, r['Cp_res']
                except Exception as e:
                    warnings.warn(f"dep_indiv i={i} phase={phase}: {e}", RuntimeWarning)
            return H, S, Cp

        def _dep_mix(x, phase):
            try:
                r = eos.residual_properties(self.T, self.P, x, phase)
                return r['H_res'], r['S_res'] + dS_P, r['Cp_res']
            except Exception as e:
                warnings.warn(f"dep_mix phase={phase}: {e}", RuntimeWarning)
                return np.nan, np.nan, np.nan

        for ph, phase, w, active in (('gas', 'v', yi, self._gas_active()),
                                     ('liq', 'l', xi, self._liq_active())):
            if not active:
                continue
            Hi, Si, Cpi = _dep_indiv(phase)
            Hm, Sm, Cpm = _dep_mix(w, phase)
            for X, vi, vm in (('H', Hi, Hm), ('S', Si, Sm), ('Cp', Cpi, Cpm)):
                setattr(p, f'{X}_dep_indiv_{ph}', vi)
                setattr(p, f'{X}_dep_mix_{ph}',  vm)

    # totals, mass basis and aliases (done once)

    def _assemble_properties(self) -> None:
        p = self._props
        gas = self._gas_active()

        def _copy(v):
            return v.copy() if isinstance(v, np.ndarray) else v

        def _alias(g, l):
            """Two-phase: molar-weighted V*g + L*l. Single phase: that phase."""
            if p.phase == 'two_phase':
                return p.V_frac * g + p.L_frac * l
            return _copy(g if gas else l)

        # 1) total per phase = ig + dep (nan + nan = nan in the absent phase)
        for X, _ in _PROP_BASES:
            for scope in _SCOPES:
                for ph in _PHASES:
                    ig = getattr(p, f'{X}_ig_{scope}_{ph}')
                    dep = getattr(p, f'{X}_dep_{scope}_{ph}')
                    setattr(p, f'{X}_total_{scope}_{ph}', ig + dep)

        # 2) mass basis per phase and 3) aliases without suffix (molar and mass)
        MM = {('indiv', 'gas'): self.MM, ('indiv', 'liq'): self.MM,
              ('mix',   'gas'): p.MM_g,  ('mix',   'liq'): p.MM_l}
        MM_alias = {'indiv': self.MM, 'mix': _alias(p.MM_g, p.MM_l)}

        for X, x in _PROP_BASES:
            for kind in _KINDS:
                for scope in _SCOPES:
                    for ph in _PHASES:
                        setattr(p, f'{x}_{kind}_{scope}_{ph}',
                                getattr(p, f'{X}_{kind}_{scope}_{ph}') / MM[(scope, ph)])
                    Xal = _alias(getattr(p, f'{X}_{kind}_{scope}_gas'),
                                 getattr(p, f'{X}_{kind}_{scope}_liq'))
                    setattr(p, f'{X}_{kind}_{scope}', Xal)
                    setattr(p, f'{x}_{kind}_{scope}', Xal / MM_alias[scope])

    # density

    def _density(self) -> None:
        p, pv = self._props, self._phase_volume
        T, P_Pa, n = self.T, self.P * 1e5, self._eos.n
        RT = self._eos.R * T
        yi, xi = self._phase_compositions()
        gas, liq = self._gas_active(), self._liq_active()

        def _pure_densities(vol_fn):
            """Density of each pure component [kg/m3] using the given volume function."""
            d = np.empty(n)
            for i in range(n):
                e = np.zeros(n); e[i] = 1.0
                d[i] = self.MM[i] * 1e-3 / vol_fn(e)
            return d

        # Gas: always the EOS, no volume translation. Absent phase -> nan.
        if gas:
            p.Vm_gas = p.Vm_gas_eos = pv.gas(T, P_Pa, yi)
            p.Z_gas_eos = p.Vm_gas * P_Pa / RT
            p.dens_mix_gas = p.MM_g * 1e-3 / p.Vm_gas
            dg = _pure_densities(lambda e: pv.gas(T, P_Pa, e))
        else:
            p.Vm_gas = p.Vm_gas_eos = p.Z_gas_eos = p.dens_mix_gas = np.nan
            dg = np.full(n, np.nan)

        # Liquid: the selected volume model, plus the plain EOS root as a diagnostic.
        # Absent phase -> nan.
        if liq:
            p.Vm_liq_eos = pv.liquid_eos(T, P_Pa, xi)
            p.Z_liq_eos = p.Vm_liq_eos * P_Pa / RT
            p.Vm_liq, p.liquid_volume_source = pv.liquid(T, P_Pa, xi)
            p.dens_mix_liq = p.MM_l * 1e-3 / p.Vm_liq
            p.Z_liq_rho = p.Vm_liq * P_Pa / RT
            dl = _pure_densities(lambda e: pv.liquid(T, P_Pa, e)[0])
        else:
            p.Vm_liq = p.Vm_liq_eos = p.Z_liq_eos = p.Z_liq_rho = p.dens_mix_liq = np.nan
            p.liquid_volume_source = ''
            dl = np.full(n, np.nan)
        p.dens_pure_gas, p.dens_pure_liq = dg, dl

        # Totals by phase
        if p.phase in ('vapor'):
            p.dens_mix, p.dens_pure = p.dens_mix_gas, dg
            p.Z_mix, p.Z_gas, p.Z_liq = p.Z_gas_eos, p.Z_gas_eos, np.nan
        elif p.phase == 'liquid':
            p.dens_mix, p.dens_pure = p.dens_mix_liq, dl
            p.Z_mix, p.Z_gas, p.Z_liq = p.Z_liq_rho, np.nan, p.Z_liq_rho
        else:                                                  # two-phase: volumes are mixed by mass
            wv, wl = p.V_frac_m_total, p.L_frac_m_total
            den = wv / p.dens_mix_gas + wl / p.dens_mix_liq
            p.dens_mix = 1.0 / den if den > 0 else np.nan
            p.dens_pure = 1.0 / (wv / dg + wl / dl)
            p.Z_mix, p.Z_gas, p.Z_liq = np.nan, p.Z_gas_eos, p.Z_liq_rho
                # molar densities [kmol/m3]: mass density divided by the molar mass
        p.dens_mol_mix_gas  = p.dens_mix_gas / p.MM_g
        p.dens_mol_mix_liq  = p.dens_mix_liq / p.MM_l
        p.dens_mol_pure_gas = dg / self.MM
        p.dens_mol_pure_liq = dl / self.MM
        p.dens_mol_pure     = p.dens_pure / self.MM
        p.dens_mol_mix      = p.dens_mix * self.fn / self.fm

        p.fv = self.fm / p.dens_mix if p.dens_mix > 0 else np.nan

    # transport properties

    def _viscosity(self) -> None:
        cd = self.component_data
        p = self._props
        p.visc_g = ((cd['A_Gas_Viscos_uP']
                     + (cd['B_Gas_Viscos_uP'] * self.T)
                     + (cd['C_Gas_Viscos_uP'] * self.T**2) + (cd['D_Gas_Viscos_uP'] * self.T**3)) * 1e-7).to_numpy(dtype=float)
        p.visc_l = ((10**(cd['A_Liq_Viscos_cP'] + (cd['B_Liq_Viscos_cP'] / self.T) + (cd['C_Liq_Viscos_cP'] * self.T) + (cd['D_Liq_Viscos_cP'] * self.T**2))) * 1e-3).to_numpy(dtype=float)

        if p.phase in ('vapor'):
            p.visc_mix = float(np.sum(p.visc_g * p.V_frac_m))
        elif p.phase == 'liquid':
            p.visc_mix = float(np.sum(p.visc_l * p.L_frac_m))
        else:
            vg = float(np.sum(p.visc_g * p.V_frac_m))
            vl = float(np.sum(p.visc_l * p.L_frac_m))
            p.visc_mix = vg * p.V_frac_m_total + vl * p.L_frac_m_total

    def _surface_tension(self) -> None:
        """Disabled for the current database (NIST and Cengel data); `_compute` does not call it."""
        cd = self.component_data
        p = self._props
        cond = (1 - self.T / cd['Tc (K)']) >= 0
        ind_sf = np.where(
            cond,
            (cd['A_Surface_Tension'] * (1 - self.T / cd['Tc (K)'])**cd['n_Surface_Tension']
             ).to_numpy(dtype=float), 0)
        p.ind_sf = ind_sf
        p.sf_mix = float(np.sum(ind_sf * p.xi))

    def _conductivity(self) -> None:
        cd = self.component_data
        p = self._props
        p.k_g = (cd['A_K_gas-Thermal_Cond']
                 + cd['B_K_gas-Thermal_Cond'] * self.T
                 + cd['C_K_gas-Thermal_Cond'] * self.T**2).to_numpy(dtype=float)
        p.k_l = (cd['A_K_liq-Thermal_Cond']
                 + cd['B_K_liq-Thermal_Cond'] * self.T
                 + cd['C_K_liq-Thermal_Cond'] * self.T**2).to_numpy(dtype=float)
        if p.phase in ('vapor'):
            p.k_mix = float(np.sum(p.k_g * p.V_frac_m))
        elif p.phase == 'liquid':
            p.k_mix = float(np.sum(p.k_l * p.L_frac_m))
        else:
            kg = float(np.sum(p.k_g * p.V_frac_m))
            kl = float(np.sum(p.k_l * p.L_frac_m))
            p.k_mix = kg * p.V_frac_m_total + kl * p.L_frac_m_total

    # PH flash

    def _H_molar_light(self, T: float) -> float:
        """Total molar H of the mixture [J/mol] at (T, self.P) = ideal + residual (+hf).
        Does not touch the stream state or compute TBubble/TDew, fugacities, density, etc.
        Consistent with _enthalpies + _departure_properties + _assemble_properties."""
        cd, P, z, eos = self.component_data, self.P, self.z, self._eos
        eos_liq = self._liquid_model is None
        T_dew = self._props.TDew          # only used in gamma-phi (see _T_from_H)

        def _h(x, phase):
            return (self._h_ig_mix(T, self.Tref, cd, x, phase, T_dew, eos_liq)
                    + eos.H_residual(T, P, x, phase))

        fl = self._engine.flash_TP(T, P, z)
        if fl.phase == 'two_phase':
            return fl.beta * _h(fl.y, 'v') + (1.0 - fl.beta) * _h(fl.x, 'l')
        return _h(z, 'l' if fl.phase == 'liquid' else 'v')

    def _T_from_H(self, H_target: float, T0: float, dT0: float = 25.0,
                  T_min: float = 10.0, T_max: float = 5000.0) -> float:
        """T such that fn * H(T, P) = H_target. H(T) is increasing: bracket by steps, then brentq."""
        if self._liquid_model is not None:
            # gamma-phi: T_dew enters the liquid H_ig and does not depend on the solver's T,
            # so it is computed once
            self._props.TDew = self._saturation_temperatures()[1]

        def f(T):
            try:
                return self.fn * self._H_molar_light(T) - H_target
            except Exception:
                return np.nan

        a, fa = float(T0), f(T0)
        if fa == 0.0:
            return a
        d = -1.0 if (np.isfinite(fa) and fa > 0) else 1.0     # excess enthalpy => lower T
        step = dT0
        for _ in range(14):
            b = min(max(a + d * step, T_min), T_max)
            if b == a:
                break
            fb = f(b)
            if np.isfinite(fa) and np.isfinite(fb) and fa * fb <= 0:
                return float(brentq(f, min(a, b), max(a, b), xtol=1e-8, rtol=1e-12, maxiter=100))
            if np.isfinite(fb):
                a, fa = b, fb
            step *= 2.0
        raise ValueError(f"Could not balance H={H_target:.6g} at P={self.P} bar "
                         "(outside the EOS range?).")

    @staticmethod
    def _validate_compatible(streams) -> None:
        ref = streams[0]
        key = lambda c: (c.Tref, c.Pref, c.method, c.k_mode)
        for c in streams[1:]:
            if key(c) != key(ref):
                raise ValueError("Streams must share Tref, Pref, method and k_mode.")

    # stream algebra

    @classmethod
    def mix(cls, streams, signs=None, P=None, description=None,
            T_guess=None) -> 'Stream':
        """Adiabatic mixing: component molar balance + enthalpy balance (PH flash).

        signs: +1 for an inlet, -1 for a stream that is subtracted. P: defaults to the
        minimum P of the streams.
        Notes: the user's gamma (gamma-phi mode) is not propagated; kij are merged by name
        (repeated pair: the first stream wins); mixing ignores the PV term between different
        pressures (H at P = min(P)).
        """
        cs = list(streams)
        if not cs:
            raise ValueError("No streams given.")
        signs = [1.0] * len(cs) if signs is None else [float(s) for s in signs]
        if len(signs) != len(cs):
            raise ValueError("signs and streams must have the same length.")
        cls._validate_compatible(cs)
        ref = cs[0]

        # 1) molar balance per component (by name)
        flow: dict[str, float] = {}
        for s, c in zip(signs, cs):
            for nm, nf in zip(c.component_data['Component'], c._props.molar_flows):
                flow[nm] = flow.get(nm, 0.0) + s * float(nf)
        scale = sum(abs(v) for v in flow.values())
        if scale == 0.0:
            raise ValueError("The resulting flow is zero.")
        comp_mol = {}
        for nm, v in flow.items():
            if v < -1e-9 * scale:
                raise ValueError(f"Negative flow of '{nm}' ({v:.4g}): the subtraction exceeds what is available.")
            if v > 1e-9 * scale:
                comp_mol[nm] = v
        fn_r = sum(comp_mol.values())
        if fn_r <= 0:
            raise ValueError("The resulting flow is zero.")

        # 2) enthalpy balance (H_total_mix = ig + departure + hf, molar basis)
        H_flow = 0.0
        for s, c in zip(signs, cs):
            h = c._props.H_total_mix
            if not np.isfinite(h):
                raise ValueError(f"H_total_mix is not finite in '{c.description}'.")
            H_flow += s * c.fn * h

        # 3) component data, kij, pressure and T seed
        data = (pd.concat([c.component_data for c in cs], ignore_index=True)
                  .drop_duplicates(subset=['Component']).reset_index(drop=True))
        kij = cls._merge_kij(cs)
        P_r = min(c.P for c in cs) if P is None else float(P)

        if T_guess is None:
            pos = [(c.fn, c.T) for c in cs if c.fn > 0]
            if all(s > 0 for s in signs) and pos:
                T_guess = sum(w * T for w, T in pos) / sum(w for w, _ in pos)
            else:
                T_guess = ref.T

        return cls(description or ' + '.join(c.description for c in cs),
                   fm=None, comp_mass=None, fn=fn_r, comp_mol=comp_mol, V_frac=None,
                   T=None, P=P_r, Tref=ref.Tref, Pref=ref.Pref, gamma=None,
                   component_data=data, kij=kij, k_mode=ref.k_mode, method=ref.method,
                   forced_cubic=ref.forced_cubic,
                   solver_options=ref.solver_options, emit_warnings=ref.emit_warnings,
                   psat_factor=ref.gamma_psat_factor, H_flow=H_flow, T_guess=T_guess)

    def __add__(self, o):
        if not isinstance(o, Stream):
            return NotImplemented
        return Stream.mix([self, o])

    def __radd__(self, o):                      # allows sum([c1, c2, c3])
        return self if o == 0 else NotImplemented

    def __sub__(self, o):
        if not isinstance(o, Stream):
            return NotImplemented
        return Stream.mix([self, o], signs=[1, -1],
                          description=f'{self.description} - {o.description}')

    def _scaled(self, f: float, desc: str) -> 'Stream':
        """Same T, P, phase and composition; only the flows change. Accepts f = 0."""
        if not np.isfinite(f) or f < 0:
            raise ValueError(f"Invalid factor: {f}")
        new = copy.copy(self)                          # shares _eos and _engine (read-only)
        new.component_data = self.component_data.copy(deep=True)
        new._props = copy.deepcopy(self._props)
        for a in _EXTENSIVE:
            v = getattr(new._props, a, None)
            if v is not None:
                setattr(new._props, a, v * f)
        new.fm, new.fn = self.fm * f, self.fn * f
        new.comp_mass, new.comp_mol = dict(self.comp_mass), dict(self.comp_mol)
        new.description = desc
        return new

    def __mul__(self, f):
        return self._scaled(float(f), f'{self.description} x {f:g}')

    __rmul__ = __mul__

    def __truediv__(self, f):
        return self._scaled(1.0 / float(f), f'{self.description} / {f:g}')

    def split(self, fractions) -> list['Stream']:
        """Split into identical streams (same T, P, composition) carrying the given fractions
        of the flow. If sum(fractions) < 1, a last stream with the remainder is appended."""
        fr = [float(f) for f in fractions]
        if any(f < 0 for f in fr):
            raise ValueError("Negative fractions.")
        tot = sum(fr)
        if tot > 1 + 1e-9:
            raise ValueError(f"The fractions sum to {tot:.6f} > 1.")
        if tot < 1 - 1e-9:
            fr.append(1.0 - tot)
        return [self._scaled(f, f'{self.description} [{i + 1}/{len(fr)}]')
                for i, f in enumerate(fr)]

    def deviation(self, other, T: bool = False, P: bool = False, relative: bool = False) -> float:
        """Deviation of the per-component mass flows (plus optional |dT| and |dP|).
        relative=True divides by self.fm (useful as a convergence criterion in recycles)."""
        comps = set(self.comp_mass) | set(other.comp_mass)
        d = sum(abs(self.fm * self.comp_mass.get(c, 0.0) - other.fm * other.comp_mass.get(c, 0.0))
                for c in comps)
        if T:
            d += abs(self.T - other.T)
        if P:
            d += abs(self.P - other.P)
        return d / self.fm if (relative and self.fm > 0) else d
