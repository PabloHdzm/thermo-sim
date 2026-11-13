"""P-T phase envelope of a natural-gas-like mixture (5 components, PR) against DWSIM 10.2.12.0.

Bubble and dew branches (both DWSIM exports), critical point, cricondentherm, cricondenbar, phase
identification curve and the phase at the operating point (300 K, 1 atm). See _dwsim_phase_envelope.py
and reference_data/dwsim/README_phase_envelope_section.md.
Needs ethane, butane and pentane in the packaged database (scripts/build_db.py).
"""
import numpy as np
import pytest

from _dwsim_phase_envelope import (CFG, P_START, branch_distances, envelope, feed, missing_components,
                                   pip_diffs, points, special_points)

pytestmark = pytest.mark.skipif(bool(missing_components()),
                                reason='ethane, butane or pentane missing: run scripts/build_db.py')

DIST_TOL = 0.02          # K, or 2e-4 relative in P (observed: bubble 2.4e-3, dew 7e-3, chord error of the polyline)


def test_envelope_traced_to_the_end():
    _, r = envelope()
    assert r.status == 'ok'
    assert r.critical_point is not None


@pytest.mark.parametrize('section,branch', [('bubble_custom_init', 'bubble'), ('bubble_default_init', 'bubble'),
                                           ('dew', 'dew')])
def test_branch_points_vs_dwsim(section, branch):
    T, P, d = branch_distances(section, branch)
    assert d.max() < DIST_TOL, f'worst point {T[d.argmax()]} K, {P[d.argmax()]} bar'


def test_bubble_and_dew_point_at_1_atm():
    pe, _ = envelope()
    Td, Pd = points('dew')
    Tb, Pb = points('bubble_custom_init')
    z = pe.z
    assert pe.engine.dew_T(P_START, z).T == pytest.approx(Td[Pd.argmin()], abs=2e-3)
    assert pe.engine.bubble_T(P_START, z).T == pytest.approx(Tb[Pb.argmin()], abs=2e-2)


def test_special_points_vs_dwsim():
    d = special_points()
    assert d['critical dT [K]'] == pytest.approx(0.0, abs=5e-3)         # observed 1.4e-4
    assert d['critical dP [bar]'] == pytest.approx(0.0, abs=5e-3)       # observed 2.5e-5
    assert d['cricondentherm dT [K]'] == pytest.approx(0.0, abs=2e-3)   # observed 4e-4
    assert d['cricondenbar dP [bar]'] == pytest.approx(0.0, abs=1e-3)   # observed 2e-4
    assert abs(d['cricondenbar dT [K] (DWSIM grid 1 K)']) < 0.15        # flat maximum, DWSIM grid


def test_cricondentherm_pressure_inside_dwsim_plateau():
    _, r = envelope()
    Td, Pd = points('dew')
    plateau = Pd[Td >= Td.max() - 1e-9]
    assert plateau.min() - 1e-6 <= r.cricondentherm[1] <= plateau.max() + 1e-6


def test_phase_identification_curve_vs_dwsim():
    T, _, rel = pip_diffs()
    below = T <= 228.5                                                  # equal Gibbs energy of the two roots
    assert np.abs(rel[below]).max() < 1e-5
    assert np.abs(rel[~below]).max() < 1.5e-3                           # Pi = 1 (DWSIM finite differences)


def test_operating_point_is_vapor():
    s = feed()
    assert s.V_frac == pytest.approx(1.0)
    pe, _ = envelope()
    assert CFG['operating_point']['T_K'] > pe.engine.dew_T(CFG['operating_point']['P_bar'], pe.z).T


def test_equilibrium_residuals_recomputed_independently():
    """R_i = ln x_i + ln phi_i(x) - ln y_i - ln phi_i(y) and sum(y) - 1 recomputed from ln K with the roots
    asked for by name; observed 3e-14 and 4e-16. Near the critical point the residual only reflects the
    conditioning of the problem, so the threshold applies to ||ln K|| > 0.05."""
    _, r = envelope()
    assert r.residual_max(0.05) < 1e-8
    assert r.sum_error_max(0.05) < 1e-10


def test_phases_are_distinct_and_lnK_has_the_right_sign():
    _, r = envelope()
    far = [p for p in r.path if np.linalg.norm(p.lnK) > 0.05]
    assert min(p.dZ for p in far) > 1e-4                                # no trivial solution (observed 6e-3)
    assert all(p.lnK[0] > 0 for p in r.path if p.branch == 'bubble')    # methane richer in the vapor
    assert all(p.lnK[0] < 0 for p in r.path if p.branch == 'dew')       # incipient phase is the liquid