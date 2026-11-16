"""Liquid-phase activity-coefficient models and the gamma-phi K-value calculator."""
from __future__ import annotations

import copy
import warnings
from abc import ABC, abstractmethod

import numpy as np

from .eos import _eos_label
from .equilibrium import _Kernel, _SubEOS


class ActivityModel(ABC):
    """Minimal interface of an activity-coefficient model."""

    @abstractmethod
    def gamma(self, T: float, x: np.ndarray) -> np.ndarray:
        """gamma_i(T, x) for the liquid composition x (active components)."""

    def subset(self, mask: np.ndarray) -> "ActivityModel":
        """Model restricted to the components in `mask`. By default x is padded with zeros."""
        return _SubsetActivity(self, mask)


class _SubsetActivity(ActivityModel):
    def __init__(self, base: ActivityModel, mask: np.ndarray):
        self.base, self.mask = base, np.asarray(mask, bool)

    def gamma(self, T, x):
        full = np.zeros(self.mask.size)
        full[self.mask] = x
        return np.asarray(self.base.gamma(T, full), float)[self.mask]


class UserGamma(ActivityModel):
    """Constant, user-defined gamma (placeholder until NRTL/UNIFAC are available)."""

    def __init__(self, gamma):
        self._g = np.asarray(gamma, float).copy()

    def gamma(self, T, x):
        return self._g.copy()

    def subset(self, mask):
        return UserGamma(self._g[np.asarray(mask, bool)])


class CallableActivity(ActivityModel):
    def __init__(self, fn):
        self.fn = fn

    def gamma(self, T, x):
        return np.asarray(self.fn(T, x), float)


class GammaPhiModel:
    """Gamma-phi liquid phase for the equilibrium engine.
    Same structure as the activity-model property methods of process simulators:

        cubic (PR, SRK, RK-Soave...): EOS in both phases, no liquid model
        gamma (NRTL-RK, UNIFAC...):   liquid = gamma * phi_sat * Poynting * Psat, vapor = EOS
                                      (Henry's law for supercritical components: pending)

    The model returns an effective ln phi_i^L

        ln phi_i^L = ln gamma_i + ln phi_i^sat + ln(Psat_i / P) + ln Poynting_i

    so the engine (stability, bubble, dew, flash) treats a gamma-phi liquid like an EOS
    liquid, and K_i = phi_i^L / phi_i^V still equals
    gamma_i phi_i^sat Poynting_i Psat_i / (P phi_i^V).

    Components that do not use gamma-phi (the EOS is used in the liquid as well):
      * Tr_i > TR_THRESHOLD (supercritical; this is where Henry's law would go)
      * names in `forced_cubic` (N2, CO2, CH4, H2, H2S, He, Ar by default)
      * no Antoine parameters
      * invalid or extrapolated Psat_i (<= 0), or Psat_i > psat_factor * P

    gamma can be an array/list (constant, one value per active component), a dict
    {name: value}, a function f(T, x) -> array, or any ActivityModel (NRTL, UNIFAC,
    Wilson... in the future).
    """
    _FORCED_CUBIC_DEFAULT: frozenset = frozenset({
        'nitrogen', 'carbon dioxide', 'methane', 'hydrogen',
        'hydrogen sulfide', 'helium', 'argon',
    })

    TR_THRESHOLD = 1.05     # components above this Tr use the EOS (NH3 at Tr ~ 1.02 still uses Antoine)
    # Psat > psat_factor * P is treated as an invalid extrapolation and falls back to the EOS.
    # inf disables the rule (default); a finite value such as 2 can produce jumps in K.
    PSAT_FACTOR = np.inf

    def __init__(self, eos, df, gamma=None, forced_cubic=None, henry=None, psat_factor=None):
        if henry is not None:
            raise NotImplementedError(
                "Henry's law is not implemented yet. Hook point: GammaPhiModel.lnphi_liquid "
                "(components that use the EOS in the liquid phase).")
        self.psat_factor = self.PSAT_FACTOR if psat_factor is None else float(psat_factor)
        self.eos = eos
        self._eos_name = _eos_label(eos)
        self.n = int(eos.n)
        self.Tc = np.asarray(eos.Tc, float)
        self.forced_cubic = (frozenset(str(c).lower() for c in forced_cubic) if forced_cubic is not None
                             else self._FORCED_CUBIC_DEFAULT)
        self._names = (df['Component'].str.lower().values if 'Component' in df.columns
                       else np.array([f'comp_{i}' for i in range(self.n)]))
        self._cubic_by_name = np.array([nm in self.forced_cubic for nm in self._names], bool)

        def _find_col(cands):
            for c in cands:
                if c in df.columns:
                    return df[c].values.astype(float)
            return None

        self._ant_A = _find_col(['A_Antoine_K', 'ANT_A', 'A_Antoine'])
        self._ant_B = _find_col(['B_Antoine_K', 'ANT_B', 'B_Antoine'])
        self._ant_C = _find_col(['C_Antoine_K', 'ANT_C', 'C_Antoine'])
        self._ant_D = _find_col(['D_Antoine_K', 'ANT_D', 'D_Antoine'])
        self._ant_E = _find_col(['E_Antoine_K', 'ANT_E', 'E_Antoine'])
        self._Vc = _find_col(['Vc (cm3/mol)', 'Vc (cm3/m)', 'Vc'])
        self._Zc = _find_col(['Zc'])
        self._no_antoine = not (self._ant_A is not None and self._ant_B is not None
                                and self._ant_C is not None)
        self.activity = self._build_activity(gamma, eos)
        self.k = None
        self._cT = None
        self._cs = None

    def _build_activity(self, gamma, eos) -> ActivityModel:
        n = self.n
        if gamma is None:
            return UserGamma(np.ones(n))
        if isinstance(gamma, ActivityModel):
            return gamma
        if callable(gamma):
            return CallableActivity(gamma)
        if isinstance(gamma, dict):
            return UserGamma([float(gamma.get(nm, 1.0)) for nm in self._names])
        g = np.asarray(gamma, float)
        if g.size == 0:
            return UserGamma(np.ones(n))
        if g.size == n:
            return UserGamma(g)
        active = np.asarray(eos.zi) > 0
        if g.size == int(active.sum()):
            full = np.ones(n)
            full[active] = g
            return UserGamma(full)
        warnings.warn(f"gamma has {g.size} elements but there are {n} components. Using gamma = 1.",
                      RuntimeWarning)
        return UserGamma(np.ones(n))

    def bind(self, kernel: _Kernel):
        self.k = kernel
        self._cT, self._cs = None, None

    def subset(self, mask) -> "GammaPhiModel":
        """Copy restricted to the components in `mask` (used by the engine for z = 0 components)."""
        m = np.asarray(mask, bool)
        new = copy.copy(self)
        new.eos = _SubEOS(self.eos, m)
        new.n = int(m.sum())
        new.Tc = self.Tc[m]
        for a in ('_ant_A', '_ant_B', '_ant_C', '_ant_D', '_ant_E', '_Vc', '_Zc'):
            v = getattr(self, a)
            setattr(new, a, None if v is None else v[m])
        new._names = self._names[m]
        new._cubic_by_name = self._cubic_by_name[m]
        new.activity = self.activity.subset(m)
        new.k, new._cT, new._cs = None, None, None
        return new

    def _psat_antoine(self, T: float) -> np.ndarray:
        if self._no_antoine:
            return np.zeros(self.n)
        D = self._ant_D if self._ant_D is not None else np.zeros(self.n)
        E = self._ant_E if self._ant_E is not None else np.zeros(self.n)
        log_psat = (self._ant_A + self._ant_B / T
                    + self._ant_C * np.log10(T) + D * T + E * T ** 2)
        with np.errstate(over='ignore', invalid='ignore'):
            psat = 10.0 ** log_psat / 750.01        # mmHg -> bar
        return np.where(np.isfinite(psat) & (psat > 0) & (psat < 1e4), psat, 0.0)

    def _rackett_volume(self, T: float) -> np.ndarray:
        if self._Vc is None or self._Zc is None:
            return np.zeros(self.n)
        Tr = T / self.Tc
        Tr_safe = np.minimum(Tr, 1.0)
        exp_rack = (1.0 - Tr_safe) ** (2.0 / 7.0)
        return np.where(Tr <= 1.0, self._Vc * self._Zc ** exp_rack, 0.0)

    def _state(self, k: _Kernel, T: float) -> dict:
        """Everything that depends on T only (cached)."""
        if self._cT == T and self._cs is not None:
            return self._cs
        Tr = T / self.Tc
        psat = self._psat_antoine(T)
        eos_only = (Tr > self.TR_THRESHOLD) | self._cubic_by_name | self._no_antoine | (psat <= 0)
        invalid = (np.logical_not(self._no_antoine) & (psat <= 0) & (Tr < self.TR_THRESHOLD)
           & np.logical_not(self._cubic_by_name))
        if invalid.any():
            warnings.warn(f"Invalid Antoine Psat at T={T:.1f} K for {list(self._names[invalid])}; using the EOS.",
                          RuntimeWarning)
        v_liq = self._rackett_volume(T)
        lnphisat = np.zeros(self.n)
        for i in np.where(~eos_only)[0]:
            e = np.zeros(self.n)
            e[i] = 1.0
            lnphisat[i] = k.phi_eos(T, max(psat[i], 1e-6), e, "v")[0][i]
        self._cT = T
        self._cs = dict(Tr=Tr, psat=psat, eos_only=eos_only, v_liq=v_liq, lnphisat=lnphisat)
        return self._cs

    def _use_cubic(self, st: dict, P: float) -> np.ndarray:
        return st['eos_only'] | (st['psat'] > self.psat_factor * P)

    def _ln_poynting(self, st, T, P):
        v = st['v_liq']
        arg = (v / 1000.0) * (P - st['psat']) * 1e5 / (self.eos.R * T)
        return np.where(v > 0, np.clip(arg, -20.0, 20.0), 0.0)

    def lnphi_liquid(self, k: _Kernel, T: float, P: float, x):
        """(effective ln phi_i^L, liquid-root Z of the EOS, n_roots). Called by `_Kernel.phi(..., 'l')`."""
        x = np.asarray(x, float)
        lp_eos, Z, nr = k.phi_eos(T, P, x, "l")
        st = self._state(k, T)
        cub = self._use_cubic(st, P)
        if cub.all():
            return lp_eos, Z, nr
        gam = np.asarray(self.activity.gamma(T, x), float)
        g = ~cub
        lp = lp_eos.copy()
        lnpoy = self._ln_poynting(st, T, P)
        lp[g] = (np.log(np.maximum(gam[g], 1e-300)) + st['lnphisat'][g]
                 + np.log(st['psat'][g] / P) + lnpoy[g])
        return lp, Z, nr

    def detail(self, k: _Kernel, T: float, P: float, x_liq, x_vap) -> dict:
        """Per-component diagnostics (what the Stream used to get as `Ki_detail`)."""
        x_liq = np.asarray(x_liq, float)
        x_vap = np.asarray(x_vap, float)
        st = self._state(k, T)
        cub = self._use_cubic(st, P)
        lpl, _, _ = self.lnphi_liquid(k, T, P, x_liq)
        lpl_eos = k.phi_eos(T, P, x_liq, "l")[0]
        lpv = k.phi_eos(T, P, x_vap, "v")[0]
        Tr = st['Tr']
        name = self._eos_name
        no_ant = np.full(self.n, self._no_antoine, bool)
        method = np.where(Tr > self.TR_THRESHOLD, f'{name} (Tr>1)',
                 np.where(self._cubic_by_name, f'{name} (forced)',
                 np.where(no_ant, f'{name} (no Antoine)',
                 np.where(cub, f'{name} (invalid Psat)', 'gamma-phi'))))
        gam = np.asarray(self.activity.gamma(T, x_liq), float)
        return {
            'Tr': Tr, 'use_cubic': cub, 'Ki_method': method,
            'Psat': st['psat'].copy(),
            'phi_sat': np.where(cub, 1.0, np.exp(st['lnphisat'])),
            'Poynting': np.where(cub, 1.0, np.exp(self._ln_poynting(st, T, P))),
            'gamma': np.where(cub, 1.0, gam),
            'phi_v_mix': np.exp(lpv),
            'phi_l_mix': np.exp(lpl),          # effective (gamma-phi for volatiles, EOS for the rest)
            'phi_l_eos': np.exp(lpl_eos),      # EOS only, for reference
        }

    def calc_Ki(self, T: float, P_bar: float, x_liq, x_vap):
        """Return (Ki, detail). Same signature as the old KCalculator."""
        if self.k is None:
            self.bind(_Kernel(self.eos))
            self.k.liq = self
        k = self.k
        lpl, _, _ = self.lnphi_liquid(k, T, P_bar, np.asarray(x_liq, float))
        lpv, _, _ = k.phi_eos(T, P_bar, np.asarray(x_vap, float), "v")
        Ki = np.clip(np.exp(lpl - lpv), 1e-10, 1e10)
        return Ki, self.detail(k, T, P_bar, x_liq, x_vap)


KCalculator = GammaPhiModel     # alias: old name used by the Stream
