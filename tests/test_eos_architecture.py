"""Single-source-of-truth and cost invariants of eos.py / equilibrium.py."""
import numpy as np
import pandas as pd
import pytest

from thermo_sim.eos import CubicEOS, physical_roots

DF = pd.DataFrame({'Component': ['h2', 'n2', 'nh3'],
                   'Tc': [43.6, 126.2, 405.65], 'Pc': [20.47, 33.94, 112.77],
                   'omega': [0.0, 0.045, 0.255], 'frac_mol': [0.6, 0.2, 0.2]})
X = DF['frac_mol'].to_numpy()


@pytest.fixture
def eos():
    kij = np.zeros((3, 3))
    kij[0, 2] = kij[2, 0] = 0.1
    return CubicEOS('pr', DF, ['std'] * 3, None, kij)


def test_residual_properties_solve_the_cubic_once(eos):
    calls = []
    original = eos._valid_roots
    eos._valid_roots = lambda A, B: (calls.append(1), original(A, B))[1]
    eos.residual_properties(300.0, 150.0, X, 'v')
    assert len(calls) == 1


@pytest.mark.parametrize('T', [250.0, 400.0, 590.0])
@pytest.mark.parametrize('phase', ['v', 'l'])
def test_residual_properties_match_individual_methods(eos, T, phase):
    r = eos.residual_properties(T, 150.0, X, phase)
    assert r['H_res'] == pytest.approx(eos.H_residual(T, 150.0, X, phase), rel=1e-13)
    assert r['S_res'] == pytest.approx(eos.S_residual(T, 150.0, X, phase), rel=1e-13)
    assert r['Cp_res'] == pytest.approx(eos.Cp_residual(T, 150.0, X, phase), rel=1e-13)
    assert r['G_res'] == pytest.approx(r['H_res'] - T * r['S_res'], rel=1e-13)


def test_valid_roots_is_physical_roots(eos):
    for A, B in [(0.5, 0.05), (1.2, 0.08), (3.0, 0.2), (0.01, 0.001)]:
        assert eos._valid_roots(A, B) == physical_roots(A, B, eos.delta1, eos.delta2)


def test_artificial_root_is_never_silent(monkeypatch):
    """If no root Z > B exists (not reachable with physical A, B, so the solver is stubbed),
    the fallback must warn instead of failing silently."""
    monkeypatch.setattr('thermo_sim.eos._cubic_fast', lambda c2, c1, c0: [0.1])
    with pytest.warns(RuntimeWarning, match='artificial root'):
        r = physical_roots(1.0, 0.5, 1 + np.sqrt(2), 1 - np.sqrt(2))
    assert r == [pytest.approx(0.5 * (1 + 1e-4))]
