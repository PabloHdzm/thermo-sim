"""The 'prsv2' alpha: Stryjek-Vera PRSV2 as implemented by DWSIM.

    kappa = kappa0 + [kappa1 + kappa2 (kappa3 - Tr)(1 - sqrt(Tr))] (1 + sqrt(Tr)) (0.7 - Tr)
    kappa0 = 0.378893 + 1.4897153 w - 0.17131848 w^2 + 0.0196554 w^3
    alpha  = [1 + kappa (1 - sqrt(Tr))]^2

The correction is applied at every Tr, as DWSIM does (the older 'prsv' alpha drops it above Tr = 1).
kappa2 and kappa3 are optional; without them the alpha is PRSV1 with the correction at every Tr.
"""
import numpy as np
import pandas as pd
import pytest

from thermo_sim import METHODS, build_eos
from thermo_sim.alpha import get_alpha

W = np.array([0.037, 0.256])                      # N2, NH3
K1, K2, K3 = np.array([0.01996, 0.001]), np.array([0.3162, -0.1265]), np.array([0.5350, 0.51])


def _kappa0(w):
    return 0.378893 + 1.4897153 * w - 0.17131848 * w**2 + 0.0196554 * w**3


def _ref_kappa(Tr, w, k1, k2, k3):
    s = np.sqrt(Tr)
    return _kappa0(w) + (k1 + k2 * (k3 - Tr) * (1 - s)) * (1 + s) * (0.7 - Tr)


@pytest.mark.parametrize('Tr', [0.5, 0.7, 0.95, 1.0, 1.8, 3.0])
def test_prsv2_alpha_matches_the_formula(Tr):
    m = get_alpha('prsv2')
    Tr_a = np.full(2, Tr)
    p = {'kappa1': K1, 'kappa2': K2, 'kappa3': K3}
    k = _ref_kappa(Tr_a, W, K1, K2, K3)
    assert m.alpha(Tr_a, W, None, p) == pytest.approx((1 + k * (1 - np.sqrt(Tr_a)))**2, rel=1e-12)
    assert m.sqrt_alpha(Tr_a, W, None, p) == pytest.approx(1 + k * (1 - np.sqrt(Tr_a)), rel=1e-12)


def test_prsv2_correction_is_not_dropped_above_tr_1():
    Tr = np.full(2, 2.0)
    p = {'kappa1': K1, 'kappa2': K2, 'kappa3': K3}
    a2 = get_alpha('prsv2').alpha(Tr, W, None, p)
    a1 = get_alpha('prsv').alpha(Tr, W, None, {'kappa1': K1})      # PRSV: kappa0 only above Tr = 1
    assert np.all(np.abs(a2 / a1 - 1.0) > 1e-3)


@pytest.mark.parametrize('Tr', [0.6, 1.5])
def test_prsv2_without_kappa2_kappa3_is_prsv1_at_every_tr(Tr):
    Tr_a = np.full(2, Tr)
    k = _kappa0(W) + K1 * (1 + np.sqrt(Tr_a)) * (0.7 - Tr_a)
    got = get_alpha('prsv2').alpha(Tr_a, W, None, {'kappa1': K1})
    assert got == pytest.approx((1 + k * (1 - np.sqrt(Tr_a)))**2, rel=1e-12)


def test_prsv2_agrees_with_prsv_below_tr_1_when_kappa2_is_zero():
    Tr_a = np.full(2, 0.8)
    a2 = get_alpha('prsv2').alpha(Tr_a, W, None, {'kappa1': K1, 'kappa2': np.zeros(2), 'kappa3': np.zeros(2)})
    a1 = get_alpha('prsv').alpha(Tr_a, W, None, {'kappa1': K1})
    assert a2 == pytest.approx(a1, rel=1e-12)


def test_prsv2_preset_builds_an_eos_with_a_per_component_override():
    df = pd.DataFrame({'Component': ['nitrogen', 'hydrogen'], 'Tc (K)': [126.2, 33.19],
                       'Pc (bar)': [33.98, 13.13], 'w': [0.037, -0.215993],
                       'Vc (cm3/mol)': [90.1, 64.147], 'frac_mol': [0.5, 0.5]})
    m = METHODS['PRSV2'].but(alpha_overrides={'hydrogen': 'std'},
                             alpha_params={'kappa1': {'nitrogen': 0.01996},
                                           'kappa2': {'nitrogen': 0.3162},
                                           'kappa3': {'nitrogen': 0.5350}})
    eos = build_eos(m, df)
    assert list(eos.alpha_names) == ['prsv2', 'std']
