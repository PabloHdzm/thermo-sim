"""P-T phase envelopes (Peng-Robinson) against DWSIM 10.2.12.0, two mixtures.

Bubble and dew branches, critical point, cricondentherm, cricondenbar, phase identification curve,
equilibrium residuals, the phase at the operating point and the enthalpy and entropy along the envelope
(PH, PS, TH, TS, VP and VT diagrams). See _dwsim_phase_envelope.py and
reference_data/dwsim/README_phase_envelope_section.md.
Needs ethane, butane, pentane (natgas) in the packaged database (scripts/build_db.py).
"""
import numpy as np
import pytest

from _dwsim_phase_envelope import (CASES, P_START, PH_CASES, PS_CASES, TH_CASES, TS_CASES, VP_CASES, VT_CASES,
                                   branch_distances, branch_sections, config, critical_enthalpy, bubble_section,
                                   critical_entropy, entropy_offset, envelope, feed, missing_components,
                                   operating_point_enthalpy, operating_point_th, ph_distances, pip_diffs, points,
                                   ps_distances, special_points, th_distances, ts_distances, ts_offset, vp_distances,
                                   vt_distances, critical_volume)

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


@pytest.mark.parametrize('case,section,branch', [(c, s_, b_) for c in CASES for s_, b_ in branch_sections(c)])
def test_branch_points_vs_dwsim(case, section, branch):
    """Every export present: custom and default initialization of each branch (no skips)."""
    miss = missing_components(case)
    if miss:
        pytest.skip(f'{miss} missing: run scripts/build_db.py')
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


# ---- PH diagram: enthalpy [kJ/kg] along the envelope --------------------------------------------------
# Raw: thermo-sim with the database Cp against DWSIM. The database Cp differs from DWSIM's equation (1 to 3 %
# at 300 K, up to 17 % for butane at 114 K, below the 200 K limit of the fit), which moves the enthalpies by
# up to 1 kJ/kg out of a 460 kJ/kg range. Corrected: the integral of the Cp difference is taken out, which
# leaves the EOS (departure) part alone. Distances are in (kJ/kg, 100 ln P).

@pytest.fixture(params=PH_CASES)
def ph_case(request):
    miss = missing_components(request.param)
    if miss:
        pytest.skip(f'{miss} missing: run scripts/build_db.py')
    return request.param


@pytest.mark.parametrize('section,branch', [('bubble_custom_init', 'bubble'), ('dew', 'dew')])
def test_ph_enthalpy_with_database_cp_vs_dwsim(ph_case, section, branch):
    h, P, d = ph_distances(ph_case, section, branch, corrected=False)
    assert d.max() < 1.5, f'worst point H = {h[d.argmax()]} kJ/kg, {P[d.argmax()]} bar'   # observed 0.93 and 0.70


@pytest.mark.parametrize('section,branch', [('bubble_custom_init', 'bubble'), ('dew', 'dew')])
def test_ph_enthalpy_without_the_cp_difference_vs_dwsim(ph_case, section, branch):
    h, P, d = ph_distances(ph_case, section, branch, corrected=True)
    assert d.max() < 0.05, f'worst point H = {h[d.argmax()]} kJ/kg, {P[d.argmax()]} bar'  # observed 0.008 and 0.016


def test_ph_operating_point_enthalpy(ph_case):
    raw, corrected = operating_point_enthalpy(ph_case)
    assert abs(raw) < 0.1                                  # observed -0.024
    assert abs(corrected) < 0.01


def test_ph_critical_point_enthalpy(ph_case):
    raw, corrected = critical_enthalpy(ph_case)
    assert abs(raw) < 1.5
    assert abs(corrected) < 0.05


# ---- PS diagram: entropy [kJ/(kg K)] along the envelope -----------------------------------------------
# DWSIM's entropy differs from thermo-sim's by a constant (0.0821 kJ/(kg K), 1.74 J/(mol K), same on both
# branches within 5e-4): the curves are compared after taking it out, measured at the operating point.
# Raw: database Cp; corrected: without the integral of (Cp database - Cp DWSIM)/T. Distances in
# (J/(kg K), 100 ln P); the entropy range of the envelope is 5.0 kJ/(kg K).

@pytest.fixture(params=PS_CASES)
def ps_case(request):
    miss = missing_components(request.param)
    if miss:
        pytest.skip(f'{miss} missing: run scripts/build_db.py')
    return request.param


@pytest.mark.parametrize('section,branch', [('bubble_custom_init', 'bubble'), ('dew', 'dew')])
def test_ps_entropy_with_database_cp_vs_dwsim(ps_case, section, branch):
    s, P, d = ps_distances(ps_case, section, branch, corrected=False)
    assert d.max() < 4.0, f'worst point S = {s[d.argmax()]} kJ/(kg K), {P[d.argmax()]} bar'   # observed 2.7 and 2.3


@pytest.mark.parametrize('section,branch', [('bubble_custom_init', 'bubble'), ('dew', 'dew')])
def test_ps_entropy_without_the_cp_difference_vs_dwsim(ps_case, section, branch):
    s, P, d = ps_distances(ps_case, section, branch, corrected=True)
    assert d.max() < 0.5, f'worst point S = {s[d.argmax()]} kJ/(kg K), {P[d.argmax()]} bar'  # observed 0.145 and 0.021


def test_ps_critical_point_entropy(ps_case):
    raw, corrected = critical_entropy(ps_case)
    assert abs(raw) < 5e-3                                  # observed 2.3e-3
    assert abs(corrected) < 1e-4                            # observed 2e-6


def test_ps_constant_offset_to_dwsim(ps_case):
    """The entropy reference of DWSIM: documented, not explained. A change in thermo-sim's ideal-gas
    mixing or reference-pressure terms would move it."""
    assert entropy_offset(ps_case, corrected=True) == pytest.approx(config(ps_case)['ps_offset_kJ_kgK'], abs=1e-3)


# ---- TH and TS diagrams: the same curves with T as ordinate ---------------------------------------------
# Distances in (kJ/kg, K) and (J/(kg K), K). TH: DWSIM fails the natgas diagram (calculation failed, index
# out of range, with and without custom initialization) and n2co2 only runs without custom initialization;
# the cases are the files present. TS operating points are at the T of each export (natgas 300 K).

@pytest.fixture(params=TH_CASES)
def th_case(request):
    miss = missing_components(request.param)
    if miss:
        pytest.skip(f'{miss} missing: run scripts/build_db.py')
    return request.param


@pytest.fixture(params=TS_CASES)
def ts_case(request):
    miss = missing_components(request.param)
    if miss:
        pytest.skip(f'{miss} missing: run scripts/build_db.py')
    return request.param


@pytest.mark.parametrize('branch', ['bubble', 'dew'])
def test_th_enthalpy_with_database_cp_vs_dwsim(th_case, branch):
    sec = bubble_section(th_case, 'TH') if branch == 'bubble' else 'dew'
    h, T, d = th_distances(th_case, sec, branch, corrected=False)
    assert d.max() < 1.0, f'worst point H = {h[d.argmax()]} kJ/kg, {T[d.argmax()]} K'        # observed 0.13 and 0.41


@pytest.mark.parametrize('branch', ['bubble', 'dew'])
def test_th_enthalpy_without_the_cp_difference_vs_dwsim(th_case, branch):
    sec = bubble_section(th_case, 'TH') if branch == 'bubble' else 'dew'
    h, T, d = th_distances(th_case, sec, branch, corrected=True)
    assert d.max() < 0.02, f'worst point H = {h[d.argmax()]} kJ/kg, {T[d.argmax()]} K'       # observed 0.003 and 0.005


def test_th_operating_point_enthalpy(th_case):
    raw, corrected = operating_point_th(th_case)
    assert abs(raw) < 1e-3 and abs(corrected) < 1e-3                                         # observed 2e-6


@pytest.mark.parametrize('branch', ['bubble', 'dew'])
def test_ts_entropy_with_database_cp_vs_dwsim(ts_case, branch):
    sec = bubble_section(ts_case, 'TS') if branch == 'bubble' else 'dew'
    s, T, d = ts_distances(ts_case, sec, branch, corrected=False)
    assert d.max() < 4.0, f'worst point S = {s[d.argmax()]} kJ/(kg K), {T[d.argmax()]} K'    # observed 0.36 and 2.3


@pytest.mark.parametrize('branch', ['bubble', 'dew'])
def test_ts_entropy_without_the_cp_difference_vs_dwsim(ts_case, branch):
    sec = bubble_section(ts_case, 'TS') if branch == 'bubble' else 'dew'
    s, T, d = ts_distances(ts_case, sec, branch, corrected=True)
    assert d.max() < 0.1, f'worst point S = {s[d.argmax()]} kJ/(kg K), {T[d.argmax()]} K'    # observed 0.021 and 0.021


def test_ts_offset_is_the_same_as_in_the_ps_diagram(ts_case):
    """The entropy offset to DWSIM does not depend on T (natgas: operating points at 300 K in TS, same offset)."""
    assert ts_offset(ts_case, corrected=True) == pytest.approx(config(ts_case)['ps_offset_kJ_kgK'], abs=1e-3)


# ---- VP and VT diagrams: molar volume of the saturated phases ---------------------------------------------
# DWSIM's volumes are Z R T / P of the cubic EOS (no volume translation), so thermo-sim's EOS roots agree to
# 1e-6 relative. Distances in (100 ln V, 100 ln P) and (100 ln V, K): 0.01 means 0.01 % of volume.

@pytest.fixture(params=VP_CASES)
def vp_case(request):
    miss = missing_components(request.param)
    if miss:
        pytest.skip(f'{miss} missing: run scripts/build_db.py')
    return request.param


@pytest.fixture(params=VT_CASES)
def vt_case(request):
    miss = missing_components(request.param)
    if miss:
        pytest.skip(f'{miss} missing: run scripts/build_db.py')
    return request.param


@pytest.mark.parametrize('section,branch', [('bubble_custom_init', 'bubble'), ('dew', 'dew')])
def test_vp_volume_vs_dwsim(vp_case, section, branch):
    v, P, d = vp_distances(vp_case, section, branch)
    assert d.max() < 0.05, f'worst point V = {v[d.argmax()]} m3/kmol, {P[d.argmax()]} bar'     # observed 0.005


@pytest.mark.parametrize('section,branch', [('bubble_custom_init', 'bubble'), ('dew', 'dew')])
def test_vt_volume_vs_dwsim(vt_case, section, branch):
    v, T, d = vt_distances(vt_case, section, branch)
    assert d.max() < 0.05, f'worst point V = {v[d.argmax()]} m3/kmol, {T[d.argmax()]} K'       # observed 0.003


def test_critical_volume_vs_dwsim(vp_case):
    """DWSIM prints this volume in m3/mol (the other points in m3/kmol)."""
    liquid, vapor = critical_volume(vp_case)
    assert abs(liquid) < 1e-4 and abs(vapor) < 1e-4                                          # observed 1e-10