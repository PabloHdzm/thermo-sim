"""Temperature dependence (alpha) of cubic equations of state.

Each model is an interchangeable strategy. ThermoMethod decides which model
is applied to which component.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Mapping

import numpy as np

# Every model receives, for the group of components that use it:
#   Tr, omega, kappa_form (kappa(omega) of the EOS form), params (name -> array)
# and returns alpha = a(T) / a_c.
Params = Mapping[str, np.ndarray]


class AlphaModel(ABC):
    name: ClassVar[str]
    required: ClassVar[tuple[str, ...]] = ()  # per-component parameters that must be given
    optional: ClassVar[tuple[str, ...]] = ()
    allowed_forms: ClassVar[frozenset] = frozenset({'pr', 'srk','srk72'})
    only_components: ClassVar[frozenset | None] = None  # e.g. GD applies to H2 only

    @abstractmethod
    def alpha(self, Tr, omega, kappa_form, params: Params) -> np.ndarray: ...

    def sqrt_alpha(self, Tr, omega, kappa_form, params: Params) -> np.ndarray:
        """Signed sqrt(alpha), for the mixing rule a_ij = (1 - kij) s_i s_j.

        Alphas that are the square of an expression linear in sqrt(Tr) (standard, MC, PRSV)
        cross zero at high Tr (e.g. H2 with the 'gd' alpha, at ~587 K) and change sign.
        With sqrt(a_i a_j), which is always positive, a_mix(T) gets a corner there and H and S
        get a step. Keeping the sign keeps a_mix analytic and continuous in T.
        For an always-positive alpha (BM, GD, RK) this is just sqrt(alpha).
        """
        return np.sqrt(self.alpha(Tr, omega, kappa_form, params))


def _lin(Tr, kappa):
    """Signed sqrt of the standard alpha: 1 + kappa (1 - sqrt(Tr))."""
    return 1.0 + kappa * (1.0 - np.sqrt(Tr))


def _std(Tr, kappa):
    return _lin(Tr, kappa) ** 2


class StdAlpha(AlphaModel):
    """Standard Soave / Peng-Robinson form: [1 + kappa (1 - sqrt(Tr))]^2."""
    name = 'std'

    def alpha(self, Tr, omega, kappa_form, params):
        return _std(Tr, kappa_form)

    def sqrt_alpha(self, Tr, omega, kappa_form, params):
        return _lin(Tr, kappa_form)


class BostonMathias(AlphaModel):
    """Standard form for Tr <= 1, Boston-Mathias extrapolation for Tr > 1:

        alpha = exp[c (1 - Tr^d)]^2,  d = 1 + m/2,  c = 1 - 1/d

    where m is the EOS kappa. Continuous in value and first derivative at Tr = 1.
    """
    name = 'bm'

    def alpha(self, Tr, omega, kappa_form, params):
        d = 1.0 + kappa_form / 2.0
        c = 1.0 - 1.0 / d
        hi = np.exp(c * (1.0 - Tr ** d)) ** 2
        return np.where(Tr > 1.0, hi, _std(Tr, kappa_form))


class MathiasCopeman(AlphaModel):
    """Mathias-Copeman alpha.

    Tr > 1: [1 + c1 (1 - sqrt(Tr))]^2.
    Tr <= 1: full polynomial if c2 and c3 are finite, otherwise the standard form.
    """
    name = 'mc'
    required = ('c1',)
    optional = ('c2', 'c3')

    def alpha(self, Tr, omega, kappa_form, params):
        x = 1.0 - np.sqrt(Tr)
        c1 = params['c1']
        nan = np.full_like(c1, np.nan)
        c2, c3 = params.get('c2', nan), params.get('c3', nan)
        hi = (1.0 + c1 * x) ** 2
        full = np.isfinite(c2) & np.isfinite(c3)
        lo = np.where(full, (1.0 + c1 * x + c2 * x**2 + c3 * x**3) ** 2,
                      _std(Tr, kappa_form))
        return np.where(Tr > 1.0, hi, lo)

    def sqrt_alpha(self, Tr, omega, kappa_form, params):
        x = 1.0 - np.sqrt(Tr)
        c1 = params['c1']
        nan = np.full_like(c1, np.nan)
        c2, c3 = params.get('c2', nan), params.get('c3', nan)
        full = np.isfinite(c2) & np.isfinite(c3)
        lo = np.where(full, 1.0 + c1 * x + c2 * x**2 + c3 * x**3, _lin(Tr, kappa_form))
        return np.where(Tr > 1.0, 1.0 + c1 * x, lo)


class GraboskiDaubertH2(AlphaModel):
    """Graboski-Daubert alpha for H2: alpha = 1.202 exp(-0.30288 Tr). H2 and SRK only."""
    name = 'gd'
    allowed_forms = frozenset({'srk'})
    only_components = frozenset({'hydrogen'})

    def alpha(self, Tr, omega, kappa_form, params):
        return 1.202 * np.exp(-0.30288 * Tr)


class _PRSVBase(AlphaModel):
    required = ('kappa1',)
    allowed_forms = frozenset({'pr'})

    @staticmethod
    def kappa0(omega):
        return 0.378893 + 1.4897153 * omega - 0.17131848 * omega**2 + 0.0196554 * omega**3


class PRSVAlpha(_PRSVBase):
    """PRSV: kappa = kappa0 + kappa1 (1 + sqrt(Tr))(0.7 - Tr) for Tr < 1, kappa0 otherwise.

    The PRSV kappa0 polynomial is not the PR one. The two are within ~0.2 % at
    omega ~ 0.3 but differ a lot for omega < 0 (e.g. H2).
    """
    name = 'prsv'

    def _kappa(self, Tr, omega, params):
        k0, k1 = self.kappa0(omega), params['kappa1']
        return k0 + np.where(Tr < 1.0, k1 * (1.0 + np.sqrt(Tr)) * (0.7 - Tr), 0.0)

    def alpha(self, Tr, omega, kappa_form, params):
        return _std(Tr, self._kappa(Tr, omega, params))

    def sqrt_alpha(self, Tr, omega, kappa_form, params):
        return _lin(Tr, self._kappa(Tr, omega, params))


_IPRSV_A, _IPRSV_B, _IPRSV_C, _IPRSV_D, _IPRSV_E = 1.1, 0.25, 0.2, 1.2, 0.01


class IPRSVAlpha(_PRSVBase):
    """iPRSV (Melhem-Saini-Goodwin, 1989): kappa varies continuously with T."""
    name = 'iprsv'

    def _kappa(self, Tr, omega, params):
        k0, k1 = self.kappa0(omega), params['kappa1']
        Tx = _IPRSV_A - _IPRSV_D * (Tr + _IPRSV_B)
        Ty = np.sqrt(Tr**2 + _IPRSV_E)
        Tz = np.sqrt(Tr + _IPRSV_C)
        return k0 + k1 * (Tx + Ty) * Tz

    def alpha(self, Tr, omega, kappa_form, params):
        return _std(Tr, self._kappa(Tr, omega, params))

    def sqrt_alpha(self, Tr, omega, kappa_form, params):
        return _lin(Tr, self._kappa(Tr, omega, params))


class RKAlpha(AlphaModel):
    """Original Redlich-Kwong: alpha = Tr^-0.5."""
    name = 'rk'
    allowed_forms = frozenset({'rk'})

    def alpha(self, Tr, omega, kappa_form, params):
        return Tr ** (-0.5)


class PRSV2Alpha(_PRSVBase):
    """PRSV2 (Stryjek and Vera 1986):
    kappa = kappa0 + [k1 + k2 (k3 - Tr)(1 - sqrt(Tr))] (1 + sqrt(Tr)) (0.7 - Tr),
    applied at every Tr, as DWSIM does. kappa2 and kappa3 are optional (0 when absent)."""
    name = 'prsv2'
    optional = ('kappa2', 'kappa3')

    def _kappa(self, Tr, omega, params):
        k1 = params['kappa1']
        k2 = np.nan_to_num(params.get('kappa2', 0.0))
        k3 = np.nan_to_num(params.get('kappa3', 0.0))
        s = np.sqrt(Tr)
        return self.kappa0(omega) + (k1 + k2 * (k3 - Tr) * (1.0 - s)) * (1.0 + s) * (0.7 - Tr)

    def alpha(self, Tr, omega, kappa_form, params):
        return _std(Tr, self._kappa(Tr, omega, params))

    def sqrt_alpha(self, Tr, omega, kappa_form, params):
        return _lin(Tr, self._kappa(Tr, omega, params))

ALPHAS: dict[str, AlphaModel] = {m.name: m for m in (
    StdAlpha(), BostonMathias(), MathiasCopeman(), GraboskiDaubertH2(),
    PRSVAlpha(), IPRSVAlpha(), PRSV2Alpha(), RKAlpha())}


def get_alpha(name: str) -> AlphaModel:
    try:
        return ALPHAS[name.lower()]
    except KeyError:
        raise ValueError(f"unknown alpha: '{name}'. Options: {sorted(ALPHAS)}") from None
