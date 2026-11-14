"""P-T phase envelopes (Peng-Robinson) against DWSIM 10.2.12.0, two mixtures.

Bubble and dew branches, critical point, cricondentherm, cricondenbar, phase identification curve,
equilibrium residuals and the phase at the operating point. See _dwsim_phase_envelope.py and
reference_data/dwsim/README_phase_envelope_section.md.
Needs ethane, butane, pentane (natgas) in the packaged database (scripts/build_db.py).
"""
import numpy as np
import pytest

from _dwsim_phase_envelope import (CASES, P_START, branch_distances, config, envelope, feed,
                                   missing_components, pip_diffs, points, sections, special_points)

DIST_TOL = 0.02          # K, or 2e-4 relative in P (observed: natgas 7e-3, n2co2 4e-3; chord error of the polyline)


@pytest.fixture(params=CASES)
def case(request):
    miss = missing_components(request.param)
    if miss:
        pytest.skip(f'{miss} missing: run scripts/build_db.py')
    return request.param


def test_envelope_traced_to_the_end(case):
    _, r = envelope(case)
    assert r.status == 'ok'
    assert r.critical_point is not None


@pytest.mark.parametrize('section,branch', [('bubble_custom_init', 'bubble'), ('bubble_default_init', 'bubble'),
                                           ('dew', 'dew')])
def test_branch_points_vs_dwsim(case, section, branch):
    if section not in sections(case):
        pytest.skip(f'no {section} export for {case}')
    T, P, d = branch_distances(case, section, branch)
    assert d.max() < DIST_TOL, f'worst point {T[d.argmax()]} K, {P[d.argmax()]} bar'


def test_bubble_and_dew_point_at_1_atm(case):
    pe, _ = envelope(case)
    Td, Pd = points(case, 'dew')
    Tb, Pb = points(case, 'bubble_custom_init')
    assert pe.engine.dew_T(P_START, pe.z).T == pytest.approx(Td[Pd.argmin()], abs=2e-3)
    assert pe.engine.bubble_T(P_START, pe.z).T == pytest.approx(Tb[Pb.argmin()], abs=2e-3)


def test_special_points_vs_dwsim(case):
    d = special_points(case)
    assert d['critical dT [K]'] == pytest.approx(0.0, abs=5e-3)         # observed 1.5e-4 (both cases)
    assert d['critical dP [bar]'] == pytest.approx(0.0, abs=5e-3)       # observed 2.4e-5 and 4e-5
    assert d['cricondentherm dT [K]'] == pytest.approx(0.0, abs=2e-3)   # observed 4e-4 and 5e-4
    assert d['cricondentherm P - nearest DWSIM plateau [bar]'] == pytest.approx(0.0, abs=1e-9)
    # DWSIM prints a 1 K grid: its maximum pressure can only be at or below the true one
    assert -1e-3 < d['cricondenbar dP [bar]'] < 0.01                    # observed 2e-4 and 4e-3
    assert abs(d['cricondenbar dT [K] (DWSIM grid 1 K)']) < 0.5         # flat maximum


def test_phase_identification_curve_vs_dwsim(case):
    T, _, rel = pip_diffs(case)
    vle = T <= config(case)['pip_vle_max_T_K']                          # equal Gibbs energy of the two roots
    assert np.abs(rel[vle]).max() < 1e-5
    assert np.abs(rel[~vle]).max() < 1.5e-3                             # Pi = 1 (DWSIM finite differences)


def test_operating_point_is_vapor(case):
    op = config(case)['operating_point']
    assert feed(case).V_frac == pytest.approx(1.0)
    pe, _ = envelope(case)
    assert op['T_K'] > pe.engine.dew_T(op['P_bar'], pe.z).T


def test_equilibrium_residuals_recomputed_independently(case):
    """R_i = ln x_i + ln phi_i(x) - ln y_i - ln phi_i(y) and sum(y) - 1 recomputed from ln K with the roots
    asked for by name; observed 3e-14 and 4e-16. Near the critical point the residual only reflects the
    conditioning of the problem, so the threshold applies to ||ln K|| > 0.05."""
    _, r = envelope(case)
    assert r.residual_max(0.05) < 1e-8
    assert r.sum_error_max(0.05) < 1e-10


def test_phases_are_distinct_and_lnK_has_the_right_sign(case):
    _, r = envelope(case)
    far = [p for p in r.path if np.linalg.norm(p.lnK) > 0.05]
    assert min(p.dZ for p in far) > 1e-4                                # no trivial solution (observed 6e-3)
    # component 0 is the lightest (data sorted by Tc): richer in the vapor on the bubble branch
    assert all(p.lnK[0] > 0 for p in r.path if p.branch == 'bubble')
    assert all(p.lnK[0] < 0 for p in r.path if p.branch == 'dew')       # incipient phase is the liquid