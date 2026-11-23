"""Thermodynamic functions of one equilibrium stage, for column solvers.

A `Stream` computes much more than a stage needs (flash, stability, densities, ...) and is too slow to be called
thousands of times per Newton iteration. `StageThermo` takes the property method of a template stream (normally the
column feed) and offers the three things a MESH solver needs, as cheap functions of (T, P, composition):

    ln_K(T, P, x, y)    ln(phi_L(x) / phi_V(y))      equilibrium ratio of each component
    h_liquid(T, P, x)   molar enthalpy of the liquid [J/mol]
    h_vapor(T, P, y)    molar enthalpy of the vapor  [J/mol]

The enthalpies include the formation term and are identical to what `Stream` reports for a single-phase stream of the
same composition (checked by the tests), so a column built on this class and a flowsheet built on `Stream` use one
enthalpy. This is the single place where an enthalpy is evaluated for a column: a change of the ideal-gas heat
capacity (for example to follow another simulator) goes through ``h_ig_correction`` or here, never in the column.

``h_ig_correction``: optional callable ``f(T) -> array`` of J/mol per component, added to the ideal-gas enthalpy of
both phases. Default None. It exists to separate effects when a result is compared with a simulator that uses another
ideal-gas Cp polynomial; it is not a validated model option.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .stream import _cp_gas_coeffs, _cp_gas_limits, _int_ext


class StageThermo:
    def __init__(self, template, h_ig_correction: Callable[[float], np.ndarray] | None = None):
        self.names = [str(n).lower() for n in template.component_data['Component']]
        self.n = len(self.names)
        self.eng = template._engine
        self.eos = template._eos
        if template._liquid_model is not None:
            raise NotImplementedError('StageThermo supports the cubic EOS in both phases (no gamma-phi liquid yet).')
        cd = template.component_data
        self.Tref = float(template.Tref)
        self._C = _cp_gas_coeffs(cd)
        self._lo, self._hi = _cp_gas_limits(cd)
        self._hf = (cd['hf_298_k'] * 1000.0).to_numpy(dtype=float)
        self.MM = cd['MM'].to_numpy(dtype=float)
        self.h_ig_correction = h_ig_correction

    # ------------------------------------------------------------------------------------------------------------
    def h_ig(self, T: float) -> np.ndarray:
        """Ideal-gas enthalpy of each component [J/mol] (formation term included)."""
        h = _int_ext(self._C, self._lo, self._hi, self.Tref, T) + self._hf
        if self.h_ig_correction is not None:
            h = h + np.asarray(self.h_ig_correction(T), float)
        return h

    def ln_phi_liquid(self, T, P, x):
        return self.eng.k.phi(T, P, x, 'l')[0]

    def ln_phi_vapor(self, T, P, y):
        return self.eng.k.phi_eos(T, P, y, 'v')[0]

    def ln_K(self, T, P, x, y):
        return self.ln_phi_liquid(T, P, x) - self.ln_phi_vapor(T, P, y)

    def h_liquid(self, T, P, x) -> float:
        x = np.asarray(x, float)
        return float(x @ self.h_ig(T) + self.eos.H_residual(T, P, x, 'l'))

    def h_vapor(self, T, P, y) -> float:
        y = np.asarray(y, float)
        return float(y @ self.h_ig(T) + self.eos.H_residual(T, P, y, 'v'))
