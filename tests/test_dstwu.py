"""Shortcut column (DSTWU) on benzene/toluene: DWSIM 10.2.12.0, Peng-Robinson, reference_data/dwsim/dstwu_benzene_toluene.json.

Four feeds: 50/50 two-phase, 70/30 and 90/10 as 100 % vapor (370 K), and 90/10 as subcooled liquid (300 K).

Heater (300 K to 370 K, feed two-phase) -> shortcut column (total condenser, R = 3, x_LK,B = x_HK,D = 0.05, 1.01325 bar).

DWSIM values reproduced here: Rmin, Nmin, N, feed stage, internal flows, duties, product temperatures. The algorithm
(alpha at the feed state in Fenske and Underwood, with phi_L / phi_V of the stability trial phase for a single-phase
feed; Eduljee form of Gilliland; Fenske minimum stages of each section for the feed stage) was recovered from those
numbers; the tests of the options that DWSIM does not have (Winn, averaged alpha, Molokanov, n_stages as input,
partial condenser, several components) check internal consistency, not agreement with a simulator.
"""
import json
import math
from pathlib import Path

import pandas as pd
import pytest

from thermo_sim import METHODS, Stream, load_database, methods
from thermo_sim.unitops.cooler import Cooler
from thermo_sim.unitops.dstwu import DSTWU, ShortcutColumn, gilliland_X, gilliland_Y

REF = Path(__file__).resolve().parent.parent / 'reference_data' / 'dwsim' / 'dstwu_benzene_toluene.json'
P = 1.01325


def _ref():
    return json.loads(REF.read_text())


@pytest.fixture(scope='module')
def database():
    db = load_database()
    have = set(db['Component'].astype(str).str.lower())
    if not {'benzene', 'toluene'} <= have:
        pytest.skip('benzene or toluene is not in the properties database')
    return db


@pytest.fixture(scope='module')
def dwsim_method(tmp_path_factory, database):
    rows = [dict(table='dwsim_bt', Component=n, Tc=c['Tc'], Pc=c['Pc'], omega=c['omega'], Vc=c['Vc'], Zra=c['Zra'],
                 fitted_with='', source='DWSIM 10.2.12.0') for n, c in _ref()['constants_dwsim'].items()]
    path = tmp_path_factory.mktemp('ov') / 'overrides.csv'
    pd.DataFrame(rows).to_csv(path, index=False)
    old = methods.OVERRIDES_CSV_PATH
    methods.OVERRIDES_CSV_PATH = path
    methods.reload_overrides()
    yield METHODS['PR'].but(constants='dwsim_bt')
    methods.OVERRIDES_CSV_PATH = old
    methods.reload_overrides()


def _stream(db, method, x, fn, T, P_=P):
    return Stream('s', fm=None, comp_mass=None, fn=fn, comp_mol=x, V_frac=None, T=T, P=P_, Tref=298.15, Pref=1.01325,
                  psat_factor=None, gamma=None, component_data=db, kij=None, method=method, emit_warnings=False)


@pytest.fixture(scope='module')
def feed(database, dwsim_method):
    return _stream(database, dwsim_method, {'benzene': 0.5, 'toluene': 0.5}, 0.1, 300.0)


@pytest.fixture(scope='module')
def heater(feed):
    return Cooler(370.0, allow_condensation=True).run(feed)


@pytest.fixture(scope='module')
def hot(heater):
    return heater.outlet


def _col(**kw):
    a = dict(x_LK_bottoms=0.05, x_HK_distillate=0.05, reflux_ratio=3.0)
    a.update(kw)
    return ShortcutColumn('benzene', 'toluene', P, P, **a)


@pytest.fixture(scope='module')
def res(hot):
    return _col().run(hot)


def _kW(s):
    return s.fn * s.props.H_total_mix


# ------------------------------------------------------------------ heater and feed state

def test_heater_against_dwsim(heater):
    r = _ref()
    assert heater.duty_kW == pytest.approx(r['heater']['duty_kW'], rel=2e-3)        # liquid enthalpy, 0.15 %
    assert heater.outlet.props.V_frac == pytest.approx(r['heater']['outlet_vapor_mole_fraction'], abs=1e-4)


def test_feed_phase_compositions_against_dwsim(hot):
    s = _ref()['streams']['hot']
    assert hot.props.phase == 'two_phase'
    for i, n in enumerate(['benzene', 'toluene']):
        assert hot.props.yi[i] == pytest.approx(s['vapor_composition'][n], abs=1e-5)
        assert hot.props.xi[i] == pytest.approx(s['liquid_composition'][n], abs=1e-5)


# ------------------------------------------------------------------ the column against DWSIM

def test_minimum_reflux_ratio(res):
    assert res.R_min == pytest.approx(_ref()['column_results']['R_min'], abs=1e-4)


def test_minimum_number_of_stages(res):
    assert res.N_min == pytest.approx(_ref()['column_results']['N_min'], abs=1e-4)


def test_actual_number_of_stages_and_feed_stage(res):
    c = _ref()['column_results']
    assert res.N == pytest.approx(c['N_actual'], abs=1e-3)
    assert res.feed_stage == pytest.approx(c['optimum_feed_stage'], abs=1e-3)
    assert res.feed_stage == res.N_stripping and res.N_rectifying + res.N_stripping == pytest.approx(res.N)


def test_internal_flows(res):
    c = _ref()['column_results']
    assert res.L_rect * 1000 == pytest.approx(c['rectifying_liquid_mol_s'], abs=1e-9)
    assert res.V_rect * 1000 == pytest.approx(c['rectifying_vapor_mol_s'], abs=1e-9)
    assert res.L_strip * 1000 == pytest.approx(c['stripping_liquid_mol_s'], abs=5e-3)
    assert res.V_strip * 1000 == pytest.approx(c['stripping_vapor_mol_s'], abs=5e-3)


def test_duties(res):
    c = _ref()['column_results']
    assert res.Qc_kW == pytest.approx(c['condenser_duty_kW'], rel=1e-4)
    assert res.Qr_kW == pytest.approx(c['reboiler_duty_kW'], rel=3e-4)


def test_products_against_dwsim(res):
    s = _ref()['streams']
    assert res.D * 1000 == pytest.approx(50.0, abs=1e-9) and res.B * 1000 == pytest.approx(50.0, abs=1e-9)
    for key, st in (('distillate', res.distillate), ('bottoms', res.bottoms)):
        assert st.T == pytest.approx(s[key]['T_K'], abs=0.01), key
        assert st.props.phase == 'liquid' and st.P == pytest.approx(P)
        for n, x in s[key]['mole_fraction'].items():
            assert st.comp_mol[n] == pytest.approx(x, abs=1e-12)
    assert _kW(res.distillate) == pytest.approx(s['distillate']['energy_flow_kW'], rel=1e-3)
    assert _kW(res.bottoms) == pytest.approx(s['bottoms']['energy_flow_kW'], rel=1e-2)    # liquid enthalpy, 0.5 %


def test_height_estimate_matches_the_single_dwsim_point(res):
    """HYPOTHESIS with one data point (tray height 0.5 m): see ShortcutResult.height."""
    assert res.height == pytest.approx(_ref()['column_results']['estimated_height_m'], abs=1e-3)


def test_dwsim_uses_the_feed_alpha_in_fenske_and_underwood(hot):
    """The averaged alphas of the first thermo-sim version would not give the DWSIM numbers."""
    for mode, nmin, rmin in (('average', 6.7007, 1.8225), ('geometric', 6.7092, 1.8250)):
        r = _col(alpha_mode=mode).run(hot)
        assert r.N_min == pytest.approx(nmin, abs=1e-3) and r.R_min == pytest.approx(rmin, abs=1e-3)
        assert abs(r.N_min - _ref()['column_results']['N_min']) > 0.05
        assert abs(r.R_min - _ref()['column_results']['R_min']) > 0.015


def test_dwsim_uses_the_eduljee_form_of_gilliland(hot):
    r = _col(gilliland='molokanov').run(hot)
    assert r.N == pytest.approx(11.699, abs=2e-3)
    assert abs(r.N - _ref()['column_results']['N_actual']) > 0.1


# ------------------------------------------------------------------ other feed compositions (100 % vapor at 370 K)

CASES = ['feed_70_30', 'feed_90_10']


def _case_feed(db, method, key):
    z = _ref()['additional_cases'][key]['mole_fraction']
    return Cooler(370.0, allow_condensation=True).run(_stream(db, method, z, 0.1, 300.0)).outlet


@pytest.mark.parametrize('key', CASES)
def test_other_feeds_column_results(database, dwsim_method, key):
    case = _ref()['additional_cases'][key]
    c = case['column_results']
    h = _case_feed(database, dwsim_method, key)
    assert h.props.phase == 'vapor'
    r = _col().run(h)
    assert r.R_min == pytest.approx(c['R_min'], rel=5e-4)
    assert r.N_min == pytest.approx(c['N_min'], rel=5e-4)
    assert r.N == pytest.approx(c['N_actual'], rel=5e-4)
    assert r.feed_stage == pytest.approx(c['optimum_feed_stage'], rel=5e-4)
    assert r.L_rect * 1000 == pytest.approx(c['rectifying_liquid_mol_s'], abs=5e-3)
    assert r.V_rect * 1000 == pytest.approx(c['rectifying_vapor_mol_s'], abs=5e-3)
    assert r.L_strip * 1000 == pytest.approx(c['stripping_liquid_mol_s'], abs=1e-2)       # DWSIM q is 1e-5 off
    assert r.V_strip * 1000 == pytest.approx(c['stripping_vapor_mol_s'], abs=1e-2)
    assert r.Qc_kW == pytest.approx(c['condenser_duty_kW'], rel=2e-4)
    assert r.Qr_kW == pytest.approx(c['reboiler_duty_kW'], rel=3e-4)
    assert r.height == pytest.approx(c['estimated_height_m'], rel=5e-4)          # 0.5 * (N + 2), hypothesis


@pytest.mark.parametrize('key', CASES)
def test_other_feeds_products(database, dwsim_method, key):
    s = _ref()['additional_cases'][key]['streams']
    r = _col().run(_case_feed(database, dwsim_method, key))
    assert r.D * 1000 == pytest.approx(s['distillate']['molar_flow_mol_s'], abs=1e-3)
    assert r.B * 1000 == pytest.approx(s['bottoms']['molar_flow_mol_s'], abs=0.01)      # DWSIM rounds B
    for key, st in (('distillate', r.distillate), ('bottoms', r.bottoms)):
        assert st.T == pytest.approx(s[key]['T_K'], abs=0.01)
        for n, x in s[key]['mole_fraction'].items():
            assert st.comp_mol[n] == pytest.approx(x, abs=1e-3)      # DWSIM composition follows its rounded flows


@pytest.mark.parametrize('key', CASES)
def test_vapor_feed_alpha_uses_the_trial_liquid_phase(database, dwsim_method, key):
    """DWSIM: K = phi_L(x*) / phi_V(z) with x* the incipient liquid of the vapor feed (its stream table shows
    0.494523 / 0.505477 and the corresponding fugacity coefficients)."""
    hot = _ref()['additional_cases'][key]['hot_stream']
    h = _case_feed(database, dwsim_method, key)
    K = _col()._K_feed(h, ['benzene', 'toluene'], ['benzene', 'toluene'])
    pl, pv = hot['fugacity_coefficient_liquid'], hot['fugacity_coefficient_vapor']
    k_dwsim = {n: pl[n] / pv[n] for n in pl}
    for n in K:
        assert K[n] == pytest.approx(k_dwsim[n], rel=2e-4)
    assert K['benzene'] / K['toluene'] == pytest.approx(k_dwsim['benzene'] / k_dwsim['toluene'], rel=1e-4)
    # the Wilson-like K of the stable vapor (liquid root at z) would be 0.3 % off in alpha: not what DWSIM does
    assert abs(_col().run(h).alpha_feed['benzene'] / (k_dwsim['benzene'] / k_dwsim['toluene']) - 1) < 2e-4


def test_feed_stage_rule_fenske_sections_not_kirkbride(database, dwsim_method):
    """Reported feed stage = N n_S / (n_R + n_S); Kirkbride is 8 % off for 70/30 and 15 % for 90/10."""
    h = _case_feed(database, dwsim_method, 'feed_70_30')
    ref = _ref()['additional_cases']['feed_70_30']['column_results']['optimum_feed_stage']
    k = _col(feed_stage_method='kirkbride').run(h)
    assert abs(k.feed_stage - ref) > 0.4                      # 5.76 against 6.27 (and 6.2 against 7.3 for 90/10)
    f = _col().run(h)
    assert f.feed_stage == pytest.approx(ref, rel=5e-4)


# ------------------------------------------------------------------ subcooled liquid feed (no heater)

def _subcooled(database, dwsim_method):
    c = _ref()['additional_cases']['feed_90_10_subcooled']
    return c, _stream(database, dwsim_method, c['mole_fraction'], 0.1, c['feed_T_K'])


def test_subcooled_liquid_feed_against_dwsim(database, dwsim_method):
    c, f = _subcooled(database, dwsim_method)
    assert f.props.phase == 'liquid'
    r = _col().run(f)
    d = c['column_results']
    assert r.q == pytest.approx(1.0)
    assert r.R_min == pytest.approx(d['R_min'], abs=2e-4)                 # negative: Underwood with q = 1
    assert r.N_min == pytest.approx(d['N_min'], rel=5e-4)
    assert r.N == pytest.approx(d['N_actual'], rel=5e-4)
    assert r.feed_stage == pytest.approx(d['optimum_feed_stage'], rel=5e-4)
    for a, b in (('L_rect', 'rectifying_liquid_mol_s'), ('V_rect', 'rectifying_vapor_mol_s'),
                 ('L_strip', 'stripping_liquid_mol_s'), ('V_strip', 'stripping_vapor_mol_s')):
        assert getattr(r, a) * 1000 == pytest.approx(d[b], abs=1e-2)
    assert r.Qc_kW == pytest.approx(d['condenser_duty_kW'], rel=2e-4)
    assert r.Qr_kW == pytest.approx(d['reboiler_duty_kW'], rel=1e-3)      # liquid enthalpy of the feed, 0.03 %


def test_liquid_feed_alpha_uses_the_trial_vapor_phase(database, dwsim_method):
    """K = phi_L(z) / phi_V(y*), y* the incipient vapor (0.967284 / 0.032717 in the DWSIM stream table)."""
    c, f = _subcooled(database, dwsim_method)
    fc = c['feed_stream']
    pl, pv = fc['fugacity_coefficient_liquid'], fc['fugacity_coefficient_vapor']
    a_dwsim = (pl['benzene'] / pv['benzene']) / (pl['toluene'] / pv['toluene'])
    assert _col().run(f).alpha_feed['benzene'] == pytest.approx(a_dwsim, rel=1e-4)


# ------------------------------------------------------------------ internal consistency

def test_underwood_equations_hold(res, hot):
    z = {'benzene': 0.5, 'toluene': 0.5}
    a = res.alpha_used
    assert sum(a[n] * z[n] / (a[n] - res.theta) for n in z) == pytest.approx(1.0 - res.q, abs=1e-10)
    assert sum(a[n] * res.x_D[n] / (a[n] - res.theta) for n in z) - 1.0 == pytest.approx(res.R_min, abs=1e-12)
    assert 1.0 < res.theta < a['benzene']


def test_fenske_equation_holds(res):
    ratio = (res.x_D['benzene'] / res.x_D['toluene']) * (res.x_B['toluene'] / res.x_B['benzene'])
    assert res.N_min == pytest.approx(math.log(ratio) / math.log(res.alpha_used['benzene']), abs=1e-12)


def test_mass_and_energy_balances(res, hot):
    F = hot.fn
    assert res.D + res.B == pytest.approx(F, rel=1e-12)
    for n in ('benzene', 'toluene'):
        assert res.D * res.x_D[n] + res.B * res.x_B[n] == pytest.approx(F * 0.5, rel=1e-12)
    assert res.Qr_kW - res.Qc_kW == pytest.approx(_kW(res.distillate) + _kW(res.bottoms) - _kW(hot), abs=1e-9)
    assert res.energy_error_kW < 1e-9


def test_flows_from_q(res):
    F = 0.1
    assert res.q == pytest.approx(1.0 - 0.721619, abs=1e-5)
    assert res.L_strip == pytest.approx(res.L_rect + res.q * F, rel=1e-12)
    assert res.V_strip == pytest.approx(res.V_rect - (1 - res.q) * F, rel=1e-12)


def test_recoveries_specification_is_equivalent(hot, res):
    r = _col(x_LK_bottoms=None, x_HK_distillate=None, rec_LK=0.95, rec_HK=0.05).run(hot)
    assert r.R_min == pytest.approx(res.R_min, abs=1e-9) and r.N == pytest.approx(res.N, abs=1e-9)
    assert r.rec_LK == pytest.approx(0.95) and r.x_D['benzene'] == pytest.approx(0.95)


@pytest.mark.parametrize('g', ['eduljee', 'molokanov'])
def test_n_stages_input_returns_the_reflux_ratio(hot, g):
    base = _col(gilliland=g).run(hot)
    r = _col(reflux_ratio=None, n_stages=base.N, gilliland=g).run(hot)
    assert r.R == pytest.approx(3.0, abs=1e-8) and r.N == base.N
    assert r.Qc_kW == pytest.approx(base.Qc_kW, rel=1e-9)


def test_reflux_as_multiple_of_the_minimum(hot, res):
    r = _col(reflux_ratio=None, reflux_to_min=1.5).run(hot)
    assert r.R == pytest.approx(1.5 * res.R_min, rel=1e-12)
    assert r.N > res.N and r.Qc_kW < res.Qc_kW                    # less reflux: more stages, smaller condenser


def test_trade_off_between_reflux_and_stages(hot, res):
    close = _col(reflux_ratio=res.R_min * 1.01).run(hot)
    far = _col(reflux_ratio=100.0).run(hot)
    assert close.N > 3 * res.N_min and close.N < 4 * res.N_min + 3 and far.N == pytest.approx(res.N_min, rel=0.05)


def test_eduljee_stage_count_is_bounded_near_the_minimum_reflux(hot, res):
    """Known limitation of the correlation: N -> 4 Nmin + 3 as R -> Rmin, so more stages cannot be requested."""
    with pytest.raises(ValueError, match='Eduljee'):
        _col(reflux_ratio=None, n_stages=4 * res.N_min + 3.5).run(hot)


def test_gilliland_inverse():
    for g in ('eduljee', 'molokanov'):
        for X in (0.01, 0.2897, 0.7):
            assert gilliland_X(gilliland_Y(X, g), g) == pytest.approx(X, rel=1e-9)


def test_winn_is_close_to_fenske(hot, res):
    r = _col(nmin_method='winn').run(hot)
    assert r.N_min == pytest.approx(res.N_min, rel=0.01) and r.N_min != pytest.approx(res.N_min, abs=1e-6)


def test_averaged_alpha_uses_the_product_values(hot):
    r = _col(alpha_mode='average').run(hot)
    a = 0.5 * (r.alpha_top['benzene'] + r.alpha_bottom['benzene'])
    assert r.alpha_used['benzene'] == pytest.approx(a, rel=1e-12)
    assert r.alpha_bottom['benzene'] < r.alpha_feed['benzene'] < r.alpha_top['benzene']


def test_enthalpy_q_is_close_to_the_vapor_fraction_q(hot, res):
    r = _col(q_mode='enthalpy').run(hot)
    assert r.q == pytest.approx(res.q, abs=0.01) and r.q_mode == 'enthalpy'


def test_partial_condenser_vapor_distillate(hot, res):
    r = _col(condenser='partial').run(hot)
    assert r.distillate.props.V_frac == pytest.approx(1.0, abs=1e-6)
    assert r.T_top > res.T_top and r.Qc_kW < res.Qc_kW
    assert r.Qr_kW - r.Qc_kW == pytest.approx(_kW(r.distillate) + _kW(r.bottoms) - _kW(hot), abs=1e-9)


def test_dstwu_alias_and_report(hot):
    assert DSTWU is ShortcutColumn
    txt = _col().run(hot).report()
    assert 'Minimum reflux ratio' in txt and 'Reboiler duty' in txt


# ------------------------------------------------------------------ several components

@pytest.fixture(scope='module')
def three(database, dwsim_method):
    have = set(database['Component'].astype(str).str.lower())
    if 'pentane' not in have:
        pytest.skip('pentane is not in the properties database')
    f = _stream(database, dwsim_method, {'pentane': 0.2, 'benzene': 0.4, 'toluene': 0.4}, 0.1, 300.0)
    return Cooler(360.0, allow_condensation=True).run(f).outlet


def test_light_component_goes_to_the_distillate(three):
    r = ShortcutColumn('benzene', 'toluene', P, x_LK_bottoms=0.02, x_HK_distillate=0.02, reflux_to_min=1.3).run(three)
    assert r.components == ['pentane', 'benzene', 'toluene']
    assert 'pentane' not in r.x_B and r.D * r.x_D['pentane'] == pytest.approx(0.1 * 0.2, rel=1e-12)
    assert r.x_B['benzene'] == pytest.approx(0.02) and r.x_D['toluene'] == pytest.approx(0.02)
    a = r.alpha_used
    z = {'pentane': 0.2, 'benzene': 0.4, 'toluene': 0.4}
    assert sum(a[n] * z[n] / (a[n] - r.theta) for n in z) == pytest.approx(1.0 - r.q, abs=1e-10)
    assert r.energy_error_kW < 1e-9 and r.N > r.N_min and r.R_min > 0


# ------------------------------------------------------------------ input checks

def test_input_validation(hot):
    base = dict(x_LK_bottoms=0.05, x_HK_distillate=0.05, reflux_ratio=3.0)
    with pytest.raises(ValueError, match='not both'):
        ShortcutColumn('benzene', 'toluene', P, rec_LK=0.9, rec_HK=0.1, **base)
    with pytest.raises(ValueError, match='both'):
        ShortcutColumn('benzene', 'toluene', P, x_LK_bottoms=0.05, reflux_ratio=3.0)
    with pytest.raises(ValueError, match='exactly one'):
        ShortcutColumn('benzene', 'toluene', P, x_LK_bottoms=0.05, x_HK_distillate=0.05)
    with pytest.raises(ValueError, match='exactly one'):
        ShortcutColumn('benzene', 'toluene', P, n_stages=10, **base)
    with pytest.raises(ValueError, match='reflux_to_min'):
        ShortcutColumn('benzene', 'toluene', P, x_LK_bottoms=0.05, x_HK_distillate=0.05, reflux_ratio=-1.5)
    with pytest.raises(ValueError, match='different'):
        ShortcutColumn('benzene', 'benzene', P, **base)
    with pytest.raises(ValueError, match='P_top'):
        ShortcutColumn('benzene', 'toluene', 2.0, 1.0, **base)
    with pytest.raises(ValueError, match='condenser'):
        _col(condenser='none')
    with pytest.raises(NotImplementedError):
        _col(condenser='partial_vl')


def test_run_errors(hot):
    with pytest.raises(ValueError, match='below the minimum|exceed the minimum'):
        _col(reflux_ratio=1.0).run(hot)
    with pytest.raises(ValueError, match='exceed the minimum number of stages'):
        _col(reflux_ratio=None, n_stages=5.0).run(hot)
    with pytest.raises(ValueError, match='swap the keys'):
        ShortcutColumn('toluene', 'benzene', P, x_LK_bottoms=0.05, x_HK_distillate=0.05, reflux_ratio=3.0).run(hot)
    with pytest.raises(ValueError, match='not in the feed'):
        ShortcutColumn('benzene', 'ethanol', P, x_LK_bottoms=0.05, x_HK_distillate=0.05, reflux_ratio=3.0).run(hot)
    with pytest.raises(ValueError, match='infeasible'):
        _col(x_LK_bottoms=0.6, x_HK_distillate=0.6).run(hot)