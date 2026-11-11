import numpy as np
import pytest

from thermo_sim import ThermoMethod, build_eos
from thermo_sim.alpha import ALPHAS


def test_boston_mathias_continuous_at_tr_1():
    bm, k, w = ALPHAS['bm'], np.array([0.5]), np.array([0.2])
    lo = bm.alpha(np.array([1 - 1e-6]), w, k, {})
    hi = bm.alpha(np.array([1 + 1e-6]), w, k, {})
    assert np.allclose(lo, hi, rtol=1e-5)


def test_boston_mathias_equals_std_below_tr_1():
    k, w, tr = np.array([0.5]), np.array([0.2]), np.array([0.7])
    assert np.allclose(ALPHAS['bm'].alpha(tr, w, k, {}), ALPHAS['std'].alpha(tr, w, k, {}))


def test_mathias_copeman_with_c1_equal_kappa_matches_std_supercritical():
    k, w, tr = np.array([0.45]), np.array([0.1]), np.array([2.0])
    assert np.allclose(ALPHAS['mc'].alpha(tr, w, k, {'c1': k}), ALPHAS['std'].alpha(tr, w, k, {}))


def test_graboski_daubert_only_for_hydrogen(water):
    with pytest.raises(ValueError):
        build_eos(ThermoMethod('srk', 'gd'), water)


def test_graboski_daubert_only_for_srk(h2):
    with pytest.raises(ValueError):
        build_eos(ThermoMethod('pr', 'gd'), h2)


def test_prsv_requires_kappa1(h2):
    with pytest.raises(ValueError):
        build_eos('PRSV', h2)


def test_prsv_with_zero_kappa1_close_to_pr(water):
    prsv = build_eos(ThermoMethod('pr', 'prsv', alpha_params={'kappa1': {'water': 0.0}}), water)
    pr = build_eos('PR', water)
    assert abs(prsv.ai(500.0)[0] / pr.ai(500.0)[0] - 1) < 0.01


def test_constants_are_traceable(h2):
    method = ThermoMethod('srk', 'std', constants_user={'hydrogen': {'Tc': 40.0}})
    notes = build_eos(method, h2).notes
    assert any('Tc' in n and '40' in n for n in notes)


def test_missing_alpha_override_strict_vs_lenient(water):
    method = ThermoMethod('srk', 'std', alpha_overrides={'hydrogen': 'gd'})
    with pytest.raises(ValueError):
        build_eos(method, water)
    assert build_eos(method, water, strict=False).alpha_names == ['std']


def test_unknown_liquid_volume_model_is_rejected():
    with pytest.raises(ValueError):
        ThermoMethod('pr', 'std', liquid_volume='dippr')
