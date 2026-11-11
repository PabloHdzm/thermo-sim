import numpy as np
import pytest

from thermo_sim.activity import (ActivityModel, CallableActivity, KCalculator,
                                 GammaPhiModel, UserGamma)


def test_user_gamma_is_constant_and_returns_a_copy():
    m = UserGamma([1.2, 0.9, 1.0])
    g = m.gamma(300.0, np.array([0.2, 0.3, 0.5]))
    assert g == pytest.approx([1.2, 0.9, 1.0])
    g[0] = 99.0
    assert m.gamma(300.0, np.array([1, 0, 0]))[0] == 1.2


def test_user_gamma_subset_keeps_only_masked_components():
    m = UserGamma([1.2, 0.9, 1.0]).subset(np.array([True, False, True]))
    assert m.gamma(300.0, np.array([0.5, 0.5])) == pytest.approx([1.2, 1.0])


def test_callable_activity_receives_temperature_and_composition():
    m = CallableActivity(lambda T, x: np.exp(0.5 * (1 - x) ** 2) * (T / T))
    x = np.array([0.4, 0.6])
    assert m.gamma(350.0, x) == pytest.approx(np.exp(0.5 * (1 - x) ** 2))


def test_default_subset_expands_with_zeros_and_restricts_result():
    seen = {}

    def fn(T, x):
        seen['x'] = x.copy()
        return 1.0 + x

    sub = CallableActivity(fn).subset(np.array([True, False, True]))
    g = sub.gamma(300.0, np.array([0.3, 0.7]))
    assert seen['x'] == pytest.approx([0.3, 0.0, 0.7])
    assert g == pytest.approx([1.3, 1.7])


def test_activity_model_is_abstract():
    with pytest.raises(TypeError):
        ActivityModel()


def test_legacy_name_points_to_gamma_phi_model():
    assert KCalculator is GammaPhiModel
