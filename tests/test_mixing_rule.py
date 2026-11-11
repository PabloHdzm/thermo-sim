"""Mixing rule a_ij = (1 - kij) s_i s_j with s_i = sqrt(a_i) *keeping its sign*.

The standard alpha is the square of 1 + kappa (1 - sqrt(Tr)), which crosses zero at high Tr.
With a pseudo-critical H2 (Tc = 43.6 K, omega = 0) that happens at 587 K. Using
sqrt(a_i a_j) (always positive) leaves a corner in a_mix(T) and a ~73 kJ/kmol step in H and S
of H2-containing mixtures. Keeping the sign makes a_mix analytic and continuous.
"""

import numpy as np
import pytest

from thermo_sim import build_eos
from thermo_sim.alpha import ALPHAS
from thermo_sim.eos import mixing_aij
from thermo_sim.equilibrium import _Kernel


KIJ = np.array([[0.0, 0.1030], [0.1030, 0.0]])        # H2-N2


@pytest.fixture
def h2_n2(h2_n2_nh3):
    d = h2_n2_nh3.iloc[:2].assign(frac_mol=[0.5, 0.5]).reset_index(drop=True)
    d.loc[0, 'Tc (K)'] = 43.6          # pseudo-critical H2: sqrt(alpha) crosses zero at 587 K
    d.loc[0, 'w'] = 0.0
    return d


@pytest.fixture
def eos(h2_n2):
    return build_eos('PR', h2_n2, KIJ)


X = np.array([0.5, 0.5])


# alpha models
@pytest.mark.parametrize('name', sorted(ALPHAS))
def test_sqrt_alpha_squares_to_alpha(name):
    Tr = np.array([0.3, 0.8, 1.0, 1.5, 5.0, 14.0, 40.0])
    n = Tr.size
    params = {'c1': np.full(n, 0.5), 'c2': np.full(n, -0.1), 'c3': np.full(n, 0.05),
              'kappa1': np.full(n, 0.03)}
    omega, kappa = np.full(n, 0.2), np.full(n, 0.45)
    m = ALPHAS[name]
    s = m.sqrt_alpha(Tr, omega, kappa, params)
    assert s ** 2 == pytest.approx(m.alpha(Tr, omega, kappa, params), rel=1e-12)


def test_standard_alpha_sqrt_changes_sign_beyond_the_zero():
    Tr = np.array([5.0, 20.0])
    s = ALPHAS['std'].sqrt_alpha(Tr, np.zeros(2), np.full(2, 0.37464), {})
    assert s[0] > 0 > s[1]


@pytest.mark.parametrize('name', ['bm', 'gd', 'rk'])
def test_always_positive_alphas_keep_a_positive_sqrt(name):
    Tr = np.array([0.5, 5.0, 50.0])
    s = ALPHAS[name].sqrt_alpha(Tr, np.zeros(3), np.full(3, 0.4), {})
    assert (s > 0).all()


# EOS
@pytest.mark.parametrize('T', [300.0, 587.0, 700.0, 1000.0])
def test_sqrt_ai_squares_to_ai(eos, T):
    assert eos.sqrt_ai(T) ** 2 == pytest.approx(eos.ai(T), rel=1e-12)


@pytest.mark.parametrize('T', [250.0, 400.0, 580.0])
def test_matches_the_classic_rule_below_the_crossing(eos, T):
    classic = np.sqrt(np.outer(eos.ai(T), eos.ai(T))) * (1.0 - eos.kij)
    assert eos.aij(T) == pytest.approx(classic, rel=1e-12)


def test_h2_sqrt_a_changes_sign_at_587_K(eos):
    assert eos.sqrt_ai(580.0)[0] > 0 > eos.sqrt_ai(595.0)[0]
    assert (eos.sqrt_ai(1000.0)[1:] > 0).all()          # N2 does not cross in this range


def test_pure_component_is_unchanged_by_the_sign(eos):
    for T in (500.0, 700.0):
        assert eos.mixture_params(T, np.array([1.0, 0.0]))[0] == pytest.approx(eos.ai(T)[0], rel=1e-12)


def test_equilibrium_kernel_uses_the_same_mixing_matrix(eos):
    k = _Kernel(eos)
    for T in (300.0, 650.0):
        assert k.prep(T) == pytest.approx(eos.aij(T), rel=1e-12)


def test_mixing_aij_is_symmetric(eos):
    a = mixing_aij(eos.sqrt_ai(700.0), eos.kij)
    assert a == pytest.approx(a.T)


# continuity across the zero of the alpha
def test_residual_enthalpy_has_no_step_at_587_K(eos):
    """With sqrt(a_i a_j) the step across 587 K was +73.6 kJ/kmol per kelvin."""
    h = lambda T: eos.H_residual(T, 200.0, X, 'v')
    assert abs(h(587.5) - h(586.5)) < 3.0


def test_residual_entropy_and_cp_are_continuous_at_587_K(eos):
    s = lambda T: eos.S_residual(T, 200.0, X, 'v')
    cp = lambda T: eos.Cp_residual(T, 200.0, X, 'v')
    assert abs(s(587.2) - s(586.8)) < 0.005
    assert cp(587.2) == pytest.approx(cp(586.8), rel=0.01)


def test_cp_residual_still_matches_numerical_derivative_across_the_zero(eos):
    h = 0.05
    for T in (585.0, 589.0, 650.0):
        num = (eos.H_residual(T + h, 200.0, X, 'v') - eos.H_residual(T - h, 200.0, X, 'v')) / (2 * h)
        assert eos.Cp_residual(T, 200.0, X, 'v') == pytest.approx(num, rel=2e-3)


