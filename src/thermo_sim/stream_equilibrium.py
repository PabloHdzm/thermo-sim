"""Equilibrium behaviour of a stream: phase model, saturation temperatures,
phase assignment (TP / VF-P flash) and fugacity outputs.

Mixin for `Stream`; it expects the host class to provide `_eos`, `T`, `P`, `fn`,
`V_frac_spec`, `num_components`, `solver_options`, `emit_warnings`, `_props` and
`component_data`.

Phase labels are 'vapor', 'liquid' and 'two_phase'. When T and P are above the
critical T and P of every component, the phase is still reported with one of those
labels (from the EOS root) and a warning says that the critical point of the mixture
is not evaluated.
"""
from __future__ import annotations

import warnings

import numpy as np

from .activity import GammaPhiModel
from .eos import _eos_label
from .equilibrium import (BubblePointSolver, DewPointSolver, EquilibriumEngine,
                          FlashResult)


class EquilibriumMixin:

    def _build_phase_model(self, df_eos, use_gamma: bool) -> None:
        """Called at the end of _build_eos.

        use_gamma=False -> 'cubic': EOS in both phases.
        use_gamma=True  -> 'gamma': gamma-phi liquid (user-defined gamma for now;
                           NRTL/UNIFAC/Wilson + Henry later), EOS vapor.
        """
        if use_gamma:
            self._liquid_model = GammaPhiModel(
                self._eos, df_eos, gamma=self.gamma, forced_cubic=self.forced_cubic,
                psat_factor=getattr(self, 'gamma_psat_factor', None))
        else:
            self._liquid_model = None
        self._k_calc = self._liquid_model            # compatibility alias
        self._engine_obj = None                      # the engine is created on first use

    @property
    def _engine(self) -> EquilibriumEngine:
        if getattr(self, '_engine_obj', None) is None:
            self._engine_obj = EquilibriumEngine(self._eos, self.solver_options,
                                                 liquid_model=self._liquid_model)
        return self._engine_obj

    @property
    def _dew_solver(self) -> DewPointSolver:
        return DewPointSolver(self._eos, k_calc=self._liquid_model, engine=self._engine)

    @property
    def _bubble_solver(self) -> BubblePointSolver:
        return BubblePointSolver(self._eos, k_calc=self._liquid_model, engine=self._engine)

    def _equilibrium(self) -> None:
        p = self._props
        z = self.component_data['frac_mol'].to_numpy(dtype=float)
        self.z = z
        p.Ki = self._engine._wilson(self.T, self.P)           # informational seed
        p.TBubble, p.TDew = self._saturation_temperatures()
        if self.V_frac_spec is None:                           # a specification, not a result
            self._phase_from_T(z)
        else:
            self._phase_from_Vfrac(z)
        self._fugacity_properties(z)
        if self._engine.above_component_criticals(self.T, self.P, z):
            self._warn(f"Above component criticals (T={self.T:.1f} K, P={self.P:.1f} bar). "
                       f"The phase is reported as '{p.phase}' from the EOS root; the critical "
                       "point of the mixture is not evaluated.", 4)
        p.T = self.T

    def _warn(self, msg: str, stacklevel: int = 3) -> None:
        if self.emit_warnings:
            warnings.warn(msg, RuntimeWarning, stacklevel=stacklevel)

    def _saturation_temperatures(self) -> tuple[float, float]:
        """(TBubble, TDew). NaN if they do not exist."""
        eos, eng, z, P = self._eos, self._engine, self.z, self.P
        zi = eos.zi
        Tc_max = float(eos.Tc[zi > 0].max())
        Pc_max = float(eos.Pc[zi > 0].max())

        if self.T > Tc_max * 1.05 and P > Pc_max * 1.05 and self.V_frac_spec is None:
            # Far above the component criticals: skip the search. The warning is issued
            # once, in _equilibrium, together with the phase assignment.
            self._props.sat_status = {'dew': 'above_component_criticals',
                                      'bubble': 'above_component_criticals'}
            return np.nan, np.nan

        try:
            dew = eng.dew_T(P, z)
            bub = eng.bubble_T(P, z)
        except Exception as e:                                  # unexpected numerical error
            self._warn(f"TBubble/TDew: {type(e).__name__}: {e}", 4)
            self._props.sat_status = {'dew': 'error', 'bubble': 'error'}
            return np.nan, np.nan

        self._props.sat_status = {'dew': dew.status, 'bubble': bub.status}
        TDew = dew.T if dew.status in ('ok', 'near_critical') else np.nan
        TBub = bub.T if bub.status in ('ok', 'near_critical') else np.nan
        if dew.status == 'approx_bracket' or bub.status == 'approx_bracket':
            self._warn("TBubble/TDew: the refinement did not converge (approx_bracket); discarded.", 4)

        if self.num_components == 1:                           # pure: a single Tsat
            Tsat = TDew if np.isfinite(TDew) else TBub
            return Tsat, Tsat

        if np.isfinite(TBub) and np.isfinite(TDew) and TDew < TBub - 0.5:
            self._warn(f"TDew={TDew:.3f} K < TBubble={TBub:.3f} K: inconsistent result, "
                       "reporting TBubble only.", 4)
            TDew = np.nan
        return TBub, TDew

    # phase assignment

    def _apply_flash(self, fl, z: np.ndarray) -> None:
        """Copy a FlashResult into self._props. Does NOT touch self.V_frac_spec."""
        p = self._props
        if fl.phase == 'two_phase':
            beta = float(fl.beta)
            p.phase = 'two_phase'
            p.V_frac, p.L_frac = beta, 1.0 - beta
            p.xi, p.yi = np.asarray(fl.x, float).copy(), np.asarray(fl.y, float).copy()
            p.V, p.L = beta * self.fn, (1.0 - beta) * self.fn
            p.Ki = np.asarray(fl.K, float).copy()
        elif fl.phase == 'liquid':
            p.phase = 'liquid'
            p.V_frac, p.L_frac = 0.0, 1.0
            p.xi, p.yi = z.copy(), np.zeros_like(z)
            p.V, p.L = 0.0, self.fn
        else:                                                   # 'vapor'
            p.phase = 'vapor'
            p.V_frac, p.L_frac = 1.0, 0.0
            p.xi, p.yi = np.zeros_like(z), z.copy()
            p.V, p.L = self.fn, 0.0

    def _flash_TP_stream(self, z: np.ndarray):
        fl = self._engine.flash_TP(self.T, self.P, z)
        if not fl.converged:
            self._warn(f"TP flash did not fully converge at T={self.T:.2f} K, P={self.P:.2f} bar.", 4)
        return fl

    def _phase_from_T(self, z: np.ndarray) -> None:
        """Phase from the stability test + TP flash (NOT by comparing with TBubble/TDew)."""
        self._apply_flash(self._flash_TP_stream(z), z)

    def _phase_from_Vfrac(self, z: np.ndarray) -> None:
        """Flash at specified pressure and vapor fraction."""
        p, eng = self._props, self._engine
        VF, P = float(self.V_frac_spec), self.P
        TB, TD = p.TBubble, p.TDew
        n = len(z)
        zeros, ones = np.zeros(n), np.ones(n)
        eos = self._eos
        Tc_max = float(eos.Tc[eos.zi > 0].max())
        Pc_max = float(eos.Pc[eos.zi > 0].max())

        if not np.isfinite(TB) and not np.isfinite(TD):
            if P > Pc_max and self.T > Tc_max:                  # above the component criticals: no L-V equilibrium
                if VF >= 1.0:
                    fl = FlashResult('vapor', 1.0, zeros, z.copy(), ones)
                elif VF <= 0.0:
                    fl = FlashResult('liquid', 0.0, z.copy(), zeros, ones)
                else:
                    fl = self._flash_TP_stream(z)
                self._apply_flash(fl, z)
                return
            self._warn("No TBubble/TDew exists with V_frac specified and a condition below the component criticals. "
                       f"TP flash at the given T ({self.T:.2f} K).")
            self._apply_flash(self._flash_TP_stream(z), z)
            return

        if self.num_components == 1:
            Tsat = TD if np.isfinite(TD) else TB
            self.T = Tsat
            if VF <= 0.0:
                self._apply_flash(FlashResult('liquid', 0.0, z.copy(), zeros, ones), z)
            elif VF >= 1.0:
                self._apply_flash(FlashResult('vapor', 1.0, zeros, z.copy(), ones), z)
            else:
                self._apply_flash(FlashResult('two_phase', VF, z.copy(), z.copy(), ones), z)
            return

        if VF >= 1.0 or VF <= 0.0:
            Tx = TD if VF >= 1.0 else TB
            if not np.isfinite(Tx):
                self._warn(f"V_frac={VF:g} but no {'dew' if VF >= 1 else 'bubble'} temperature exists. "
                           f"TP flash at the given T ({self.T:.2f} K).")
                self._apply_flash(self._flash_TP_stream(z), z)
                return
            self.T = Tx
            fl = (FlashResult('vapor', 1.0, zeros, z.copy(), ones) if VF >= 1.0
                  else FlashResult('liquid', 0.0, z.copy(), zeros, ones))
            self._apply_flash(fl, z)
            return

        T, fl = eng.flash_VP(P, VF, z)
        if not np.isfinite(T) or fl is None:
            self._warn(f"No T found with V_frac={VF:g} at P={P:.2f} bar (nearly non-condensable "
                       f"mixture, or no phase envelope?). TP flash at the given T ({self.T:.2f} K).")
            self._apply_flash(self._flash_TP_stream(z), z)
            return
        self.T = float(T)
        self._apply_flash(fl, z)

    # fugacities and output K-values

    def _fugacity_properties(self, z: np.ndarray) -> None:
        p, eos = self._props, self._eos
        k = self._engine.k
        T, P = self.T, self.P
        xi = p.xi if p.xi.sum() > 0 else z
        yi = p.yi if p.yi.sum() > 0 else z

        lnv, _, _ = k.phi(T, P, yi, 'v')
        lnl, _, _ = k.phi(T, P, xi, 'l')          # with gamma-phi: effective phi^L (gamma*phisat*Poynting*Psat/P)
        phi_v, phi_l = np.exp(lnv), np.exp(lnl)

        Tr = T / eos.Tc
        eos_name = _eos_label(eos)
        if self._liquid_model is not None:
            p.Ki_detail = self._liquid_model.detail(k, T, P, xi, yi)
        else:
            p.Ki_detail = {
                'Ki_method': np.where(Tr > 1, f'{eos_name} (Tr>1)', f'{eos_name} (phi-phi)'),
                'Tr':        Tr,
                'phi_l_mix': phi_l,
                'phi_v_mix': phi_v,
                'use_cubic': np.ones(len(z), dtype=bool),
                'Psat':      np.zeros(len(z)),
                'phi_sat':   np.ones(len(z)),
                'Poynting':  np.ones(len(z)),
                'gamma':     np.ones(len(z)),
            }

        if p.phase == 'two_phase':
            Ki_final = np.asarray(p.Ki, float)               # converged K-values from the flash
        else:
            Ki_pr = np.clip(phi_l / phi_v, 1e-10, 1e10)
            Ki_final = np.where(np.abs(Ki_pr - 1.0) < 1e-6, p.Ki, Ki_pr)
        p.Ki = Ki_final

        if p.phase == 'vapor':
            p.phi_mix, p.phi_v, p.phi_l = phi_v.copy(), phi_v.copy(), np.nan
        elif p.phase == 'liquid':
            p.phi_mix, p.phi_v, p.phi_l = phi_l.copy(), np.nan, phi_l.copy()
        else:                                                 # two-phase
            p.phi_v, p.phi_l, p.phi_mix = phi_v.copy(), phi_l.copy(), np.nan

        # Pure component: phi_i(T, P, x_pure) from the EOS (reference)
        phi_pure_v = np.ones(len(z))
        phi_pure_l = np.ones(len(z))
        for i in range(len(z)):
            e = np.zeros(len(z))
            e[i] = 1.0
            phi_pure_v[i] = np.exp(k.phi_eos(T, P, e, 'v')[0][i])
            phi_pure_l[i] = np.exp(k.phi_eos(T, P, e, 'l')[0][i])   # a single root -> same as vapor

        if p.phase == 'vapor':
            p.phi_pure, p.phi_pure_v, p.phi_pure_l = phi_pure_v.copy(), phi_pure_v, np.nan
        elif p.phase == 'liquid':
            p.phi_pure, p.phi_pure_v, p.phi_pure_l = phi_pure_l.copy(), np.nan, phi_pure_l
        else:
            p.phi_pure_v, p.phi_pure_l, p.phi_pure = phi_pure_v, phi_pure_l, np.nan