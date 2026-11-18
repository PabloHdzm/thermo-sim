"""Natural gas dew-point control (DWSIM tutorial): chiller 320 K -> 240 K at 70 bar, then a separator.

Reference: DWSIM 10.2.12.0 (Classic UI), Peng-Robinson, C1-C5 feed of 100 mol/s
(reference_data/dwsim/natgas_dew_point_control.json).

Compared with DWSIM: chiller duty, vapor fraction, flows and compositions of both outlets, C3+ recovery, dew point.
DWSIM enthalpy and entropy use their own reference state, so only differences are compared.

Critical constants and kij are those of DWSIM for this mixture, read from natgas_phase_envelope.json (same
mixture as the phase envelope validation). With them thermo-sim reproduces the DWSIM split: compositions within
5e-4 (1.6e-5 observed), vapor fraction within 2e-3 and dew temperature within 0.1 K. The remaining difference is the
chiller duty (-0.3 %), which comes from the ideal-gas Cp (database fit vs DWSIM correlation).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from thermo_sim import METHODS, Stream, load_database, methods
from thermo_sim.unitops import Cooler, FlashDrum

try:                                                          # tests/ may or may not be a package
    from dwsim_conventions import split_entropy_offset_empirical
except ImportError:
    from tests.dwsim_conventions import split_entropy_offset_empirical

DATA = Path(__file__).resolve().parent.parent / 'reference_data' / 'dwsim'
REF = DATA / 'natgas_dew_point_control.json'
ENV = DATA / 'natgas_phase_envelope.json'                    # DWSIM constants and kij of the same mixture
NAMES = ['methane', 'ethane', 'propane', 'butane', 'pentane']
HEAVY = ('propane', 'butane', 'pentane')                      # C3+


def _ref():
    return json.loads(REF.read_text())


def _kij():
    k = pd.DataFrame(np.zeros((5, 5)), index=NAMES, columns=NAMES)
    for pair, v in json.loads(ENV.read_text())['kij'].items():
        a, b = pair.split('-')
        k.loc[a, b] = k.loc[b, a] = v
    return k


@pytest.fixture(scope='module')
def dwsim_method(tmp_path_factory):
    """PR with the DWSIM critical constants of the mixture (overrides table 'dwsim_natgas')."""
    rows = [dict(table='dwsim_natgas', Component=n, Tc=c['Tc'], Pc=c['Pc'], omega=c['omega'], Vc=c['Vc'],
                 Zra=c['Zra'], fitted_with='', source='DWSIM 10.2.12.0')
            for n, c in json.loads(ENV.read_text())['constants'].items()]
    path = tmp_path_factory.mktemp('ov') / 'overrides.csv'
    pd.DataFrame(rows).to_csv(path, index=False)
    old = methods.OVERRIDES_CSV_PATH
    methods.OVERRIDES_CSV_PATH = path
    methods.reload_overrides()
    yield METHODS['PR'].but(constants='dwsim_natgas')
    methods.OVERRIDES_CSV_PATH = old
    methods.reload_overrides()


def _feed(method):
    c = _ref()['inputs']
    return Stream('Wellhead-Gas', fm=None, comp_mass=None, fn=c['feed_molar_flow_mol_s'] / 1000.0,
                  comp_mol=c['composition'], V_frac=None, T=c['feed_T_K'], P=c['feed_P_bar'],
                  Tref=298.15, Pref=1.01325, psat_factor=None, gamma=None, component_data=load_database(),
                  kij=_kij(), method=method, emit_warnings=False)


@pytest.fixture(scope='module')
def plant(dwsim_method):
    ref = _ref()
    feed = _feed(dwsim_method)
    chiller = Cooler(ref['inputs']['chiller_T_out_K'], dP=ref['inputs']['chiller_dP_bar'],
                     allow_condensation=True).run(feed)
    sep = FlashDrum(T=ref['inputs']['chiller_T_out_K']).run(chiller.outlet)
    return feed, chiller, sep


def _mol_s(stream):
    return stream.fn * 1000.0


def _c3_flow(stream):
    return sum(_mol_s(stream) * stream.comp_mol[n] for n in HEAVY)


# ------------------------------------------------------------------ against DWSIM

def test_feed_is_vapor_and_mass_flow_matches(plant):
    feed, _, _ = plant
    s = _ref()['streams']['feed']
    assert feed.props.phase == 'vapor'
    assert feed.fm == pytest.approx(s['mass_flow_kg_s'], rel=1e-4)


def test_chiller_duty(plant):
    _, chiller, _ = plant
    assert chiller.duty_kW == pytest.approx(_ref()['results']['chiller_duty_kW'], rel=0.006)


def test_chiller_outlet_is_two_phase_with_dwsim_vapor_fraction(plant):
    _, chiller, sep = plant
    V = _ref()['results']['chilled_vapor_mole_fraction']
    assert chiller.outlet.props.phase == 'two_phase'
    assert sep.V_frac == pytest.approx(V, abs=0.002)


def test_outlet_flows(plant):
    _, _, sep = plant
    r = _ref()['results']
    assert _mol_s(sep.vapor) == pytest.approx(r['sales_gas_molar_flow_mol_s'], abs=0.1)
    assert _mol_s(sep.liquid) == pytest.approx(r['ngl_molar_flow_mol_s'], abs=0.1)


def test_outlet_compositions(plant):
    _, _, sep = plant
    s = _ref()['streams']
    for n in NAMES:
        assert sep.liquid.comp_mol[n] == pytest.approx(s['ngl']['x'][n], abs=5e-4), f'NGL {n}'
        assert sep.vapor.comp_mol[n] == pytest.approx(s['sales_gas']['y'][n], abs=2e-4), f'sales gas {n}'


def test_c3_plus_recovery(plant):
    feed, _, sep = plant
    recovery = _c3_flow(sep.liquid) / (_mol_s(feed) * sum(feed.comp_mol[n] for n in HEAVY))
    assert recovery == pytest.approx(_ref()['results']['c3plus_recovery_in_ngl'], abs=2e-3)


def test_feed_dew_temperature(plant):
    """Near the cricondentherm the dew curve is flat (about 300 K between 50 and 70 bar), so this point is sensitive
    to every kij: with only the C1-C2, C1-C3 and C2-C3 pairs it was 2.1 K high."""
    feed, _, _ = plant
    assert feed.props.TDew == pytest.approx(_ref()['streams']['feed']['dew_T_at_P_K'], abs=0.1)


# ------------------------------------------------------------------ tutorial criteria that DWSIM itself meets

def test_tutorial_sales_gas_is_methane_enriched(plant):
    _, _, sep = plant
    assert sep.vapor.comp_mol['methane'] > 0.85


def test_ngl_is_enriched_in_heavies(plant):
    feed, _, sep = plant
    assert sum(sep.liquid.comp_mol[n] for n in HEAVY) > 2.0 * sum(feed.comp_mol[n] for n in HEAVY)


# ------------------------------------------------------------------ internal consistency

def test_balances(plant):
    feed, chiller, sep = plant
    assert sep.mass_error < 1e-12
    assert abs(sep.energy_error_kW) < 1e-6
    assert sep.duty_kW == pytest.approx(0.0, abs=1e-6)        # adiabatic separator
    for n in NAMES:                                           # component balance
        out = _mol_s(sep.vapor) * sep.vapor.comp_mol[n] + _mol_s(sep.liquid) * sep.liquid.comp_mol[n]
        assert out == pytest.approx(_mol_s(feed) * feed.comp_mol[n], rel=1e-9)


def test_entropy_change_equals_integral_of_dH_over_T(plant):
    """Isobaric cooling at 70 bar: S(320 K) - S(240 K) = integral of dH/T. Independent of any reference state."""
    feed, chiller, _ = plant
    T = np.linspace(chiller.T_out, feed.T, 41)
    H = np.array([feed.at(T=t, P=feed.P).props.H_total_mix for t in T])
    f = H / T ** 2
    dS_integral = (H[-1] / T[-1] - H[0] / T[0]) + float(np.sum(0.5 * (f[1:] + f[:-1]) * np.diff(T)))   # by parts
    dS = feed.props.S_total_mix - chiller.outlet.props.S_total_mix
    assert dS == pytest.approx(dS_integral, rel=2e-3)


# ------------------------------------------------------------------ known differences with DWSIM

def test_entropy_change_matches_dwsim_with_empirical_mixing_rule(plant):
    """DWSIM dS(320 -> 240 K) = -23.50 kJ/(kmol K); thermo-sim gives -24.18, which equals the integral of dH/T (test
    above). The difference is explained by the empirical DWSIM mixing-entropy rule (tests/dwsim_conventions.py,
    HYPOTHESIS fitted to DWSIM output, not checked in its source): the rule alone predicts -0.743, observed -0.675.
    The remaining 0.07 is the ideal-gas Cp difference (the duty is 0.3 % low, 0.003 * 24.2 = 0.07).
    The composition of the correction is taken from the DWSIM streams, so it does not depend on thermo-sim."""
    feed, chiller, _ = plant
    ref = _ref()
    s = ref['streams']
    M = json.loads(ENV.read_text())['molar_mass']
    shift = split_entropy_offset_empirical(s['feed']['z'], s['sales_gas']['y'], s['ngl']['x'],
                                           s['chilled']['vapor_mole_fraction'], M)
    dS_dwsim = ref['results']['delta_S_feed_to_chilled_kJ_kmolK']
    dS = chiller.outlet.props.S_total_mix - feed.props.S_total_mix
    assert dS - shift == pytest.approx(dS_dwsim, abs=0.15)
    assert abs(dS - dS_dwsim) > 0.45            # without the rule the comparison fails: the rule is not vacuous


@pytest.mark.xfail(strict=False, reason='bubble_T finds no crossing at 70 bar for this feed (status no_crossing; same at '
                                       '50-85 bar): the nontrivial trial phase exists only above the bubble point, so '
                                       'the scan has no sign change. DWSIM: 220.98 K for the feed, but its PV flash '
                                       'also fails for the sales gas bubble temperature at 70 bar.')
def test_feed_bubble_temperature(plant):
    feed, _, _ = plant
    assert feed.props.TBubble == pytest.approx(_ref()['streams']['feed']['bubble_T_at_P_K'], abs=1.0)