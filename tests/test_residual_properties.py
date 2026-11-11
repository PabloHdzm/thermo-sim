"""Internal consistency of the residual properties.

Cp_res must equal dH_res/dT at constant P (this catches sign errors in Cv_res).
"""
import numpy as np
import pytest

from thermo_sim import build_eos

CASES = [(253.8, 150.0, 'v'), (253.8, 150.0, 'l'), (450.0, 100.0, 'v')]


@pytest.mark.parametrize('T,P,phase', CASES)
def test_cp_res_matches_numerical_derivative_of_h_res(h2_n2_nh3, T, P, phase):
    eos = build_eos('PR', h2_n2_nh3)
    x = h2_n2_nh3['frac_mol'].to_numpy()
    h = 0.05
    numeric = (eos.H_residual(T + h, P, x, phase) - eos.H_residual(T - h, P, x, phase)) / (2 * h)
    analytic = eos.Cp_residual(T, P, x, phase)
    assert analytic == pytest.approx(numeric, rel=2e-3, abs=1e-3)


def test_vapor_has_positive_cp_res_and_negative_h_res_at_high_pressure(h2_n2_nh3):
    eos = build_eos('PR', h2_n2_nh3)
    x = np.array([0.7333, 0.2444, 0.0223])
    r = eos.residual_properties(253.8, 150.0, x, 'v')
    assert r['Cp_res'] > 0
    assert r['H_res'] < 0


def test_gibbs_residual_is_h_minus_ts(h2_n2_nh3):
    eos = build_eos('PR', h2_n2_nh3)
    x = h2_n2_nh3['frac_mol'].to_numpy()
    r = eos.residual_properties(300.0, 50.0, x, 'v')
    assert r['G_res'] == pytest.approx(r['H_res'] - 300.0 * r['S_res'])
