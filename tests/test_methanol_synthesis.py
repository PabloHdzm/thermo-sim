"""Methanol synthesis flowsheet (single pass): DWSIM "Methanol Synthesis" tutorial.

Syngas (3 mol/s, H2:CO = 2:1, 300 K, 30 bar) -> compressor (adiabatic, 75 %, 80 bar) -> heater (525 K) ->
equilibrium reactor (CO + 2 H2 <-> CH3OH, isothermal 525 K, ln K = 11833/T - 29.07) -> cooler (308 K) -> separator.

Reference: DWSIM 10.2.12.0, Peng-Robinson, reference_data/dwsim/methanol_synthesis.json.

Two kinds of comparison:

* the whole chain run with thermo-sim (small differences accumulate: the equilibrium extent is 1.1 % above the DWSIM
  one, see the reactor tests);
* the cooler and the separator run on the DWSIM reactor outlet (composition from the DWSIM table), which isolates the
  thermodynamics of the cooled stream: phase split, compositions and energies.

The entropy tests at the end are evidence for the empirical DWSIM mixing-entropy rule (HYPOTHESIS, see
tests/dwsim_conventions.py and docs/validation_dwsim_entropy_mixing.md). They include a liquid and a two-phase stream.
Carbon monoxide and methanol must be in the database (tests are skipped otherwise).
"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from thermo_sim import METHODS, Stream, load_database, methods
from thermo_sim.unitops.compressor import Compressor
from thermo_sim.unitops.cooler import Cooler
from thermo_sim.unitops.flash_drum import FlashDrum
from thermo_sim.unitops.reactor import EquilibriumReactor

try:                                                          # tests/ may or may not be a package
    from dwsim_conventions import entropy_offset_empirical
except ImportError:
    from tests.dwsim_conventions import entropy_offset_empirical

REF = Path(__file__).resolve().parent.parent / 'reference_data' / 'dwsim' / 'methanol_synthesis.json'
NAMES = ['hydrogen', 'carbonmonoxide', 'methanol']
STOICH = {'carbonmonoxide': -1.0, 'hydrogen': -2.0, 'methanol': 1.0}
FORMULAS = {'hydrogen': {'H': 2}, 'carbonmonoxide': {'C': 1, 'O': 1}, 'methanol': {'C': 1, 'H': 4, 'O': 1}}


def _ref():
    return json.loads(REF.read_text())


def lnK_dwsim(T):
    return 11833.0 / T - 29.07


@pytest.fixture(scope='module')
def database():
    db = load_database()
    have = set(db['Component'].astype(str).str.lower())
    if not {'carbonmonoxide', 'methanol', 'hydrogen'} <= have:
        pytest.skip('hydrogen, carbon monoxide or methanol is not in the properties database')
    return db


@pytest.fixture(scope='module')
def dwsim_method(tmp_path_factory, database):
    """PR with the DWSIM critical constants (overrides table 'dwsim_methanol'). DWSIM has no Zra for methanol: the
    database value is used (it only enters the Rackett liquid volume)."""
    zra_db = database.assign(c=database['Component'].astype(str).str.lower()).set_index('c')['Z_RA']
    rows = []
    for n, c in _ref()['constants_dwsim'].items():
        zra = float(zra_db[n]) if n == 'methanol' else c['Zra']
        rows.append(dict(table='dwsim_methanol', Component=n, Tc=c['Tc'], Pc=c['Pc'], omega=c['omega'], Vc=c['Vc'],
                         Zra=zra, fitted_with='', source='DWSIM 10.2.12.0'))
    path = tmp_path_factory.mktemp('ov') / 'overrides.csv'
    pd.DataFrame(rows).to_csv(path, index=False)
    old = methods.OVERRIDES_CSV_PATH
    methods.OVERRIDES_CSV_PATH = path
    methods.reload_overrides()
    yield METHODS['PR'].but(constants='dwsim_methanol')
    methods.OVERRIDES_CSV_PATH = old
    methods.reload_overrides()


def _kij():
    k = pd.DataFrame(np.zeros((3, 3)), index=NAMES, columns=NAMES)
    k.loc['hydrogen', 'carbonmonoxide'] = k.loc['carbonmonoxide', 'hydrogen'] = _ref()['kij_pr_dwsim']['hydrogen-carbonmonoxide']
    return k


def _stream(db, method, x, fn, T, P):
    return Stream('s', fm=None, comp_mass=None, fn=fn, comp_mol=x, V_frac=None, T=T, P=P, Tref=298.15, Pref=1.01325,
                  psat_factor=None, gamma=None, component_data=db, kij=_kij(), method=method, emit_warnings=False)


def _x(stream):
    n = {c: float(stream.comp_mol.get(c, 0.0)) for c in NAMES}
    t = sum(n.values())
    return {c: v / t for c, v in n.items()}


def _kW(stream):
    return stream.fn * stream.props.H_total_mix


@pytest.fixture(scope='module')
def syngas(database, dwsim_method):
    s = _ref()['streams']['syngas']
    return _stream(database, dwsim_method, s['mole_fraction'], s['molar_flow_mol_s'] / 1000.0, s['T_K'], s['P_bar'])


@pytest.fixture(scope='module')
def comp(syngas):
    return Compressor(80.0, eta=0.75, mode='isentropic').run(syngas)


@pytest.fixture(scope='module')
def heater(comp):
    return Cooler(525.0).run(comp.outlet)


@pytest.fixture(scope='module')
def reactor(heater):
    return EquilibriumReactor(STOICH, lnK=lnK_dwsim, T_out=525.0, P_ref=1.01325, fugacity=True).run(heater.outlet)


@pytest.fixture(scope='module')
def cooler(reactor):
    return Cooler(308.0, allow_condensation=True).run(reactor.outlet)


@pytest.fixture(scope='module')
def drum(cooler):
    return FlashDrum(T=308.0).run(cooler.outlet)


@pytest.fixture(scope='module')
def dw_outlet(database, dwsim_method):
    """Reactor outlet built from the DWSIM table (composition, flow, 525 K, 80 bar)."""
    s = _ref()['streams']['reactor_out']
    return _stream(database, dwsim_method, s['mole_fraction'], s['molar_flow_mol_s'] / 1000.0, s['T_K'], s['P_bar'])

@pytest.fixture(scope='module')
def dw_hot(database, dwsim_method):
    s = _ref()['streams']['hot']
    return _stream(database, dwsim_method, s['mole_fraction'],
                   s['molar_flow_mol_s'] / 1000.0, s['T_K'], s['P_bar'])

@pytest.fixture(scope='module')
def dw_cooler(dw_outlet):
    return Cooler(308.0, allow_condensation=True).run(dw_outlet)


@pytest.fixture(scope='module')
def dw_drum(dw_cooler):
    return FlashDrum(T=308.0).run(dw_cooler.outlet)


# ------------------------------------------------------------------ feed, compressor, heater

def test_feed_energy_flow(syngas):
    s = _ref()['streams']['syngas']
    assert _kW(syngas) == pytest.approx(s['energy_flow_kW'], abs=0.01)
    assert syngas.fm == pytest.approx(s['mass_flow_kg_s'], rel=1e-5)


def test_compressor_against_dwsim(comp):
    b = _ref()['blocks']['compressor_C1']
    assert comp.T_out == pytest.approx(b['T_out_K'], abs=0.2)
    assert comp.W_ind_kW == pytest.approx(b['power_kW'], rel=1e-3)
    assert _kW(comp.outlet) == pytest.approx(_ref()['streams']['comp_out']['energy_flow_kW'], abs=0.01)


def test_compressor_isentropic_efficiency_is_the_dwsim_adiabatic_efficiency(comp):
    assert comp.eta_is_equiv == pytest.approx(0.75, abs=1e-12)
    assert comp.T_out_is < comp.T_out


def test_compressor_polytropic_mode_reproduces_the_dwsim_polytropic_efficiency(syngas):
    """DWSIM reports a polytropic efficiency of 78.1366 % for the 75 % adiabatic compressor. thermo-sim in polytropic
    mode with that efficiency gives the same outlet temperature within 0.1 K."""
    b = _ref()['blocks']['compressor_C1']
    p = Compressor(80.0, eta=b['polytropic_efficiency_pct'] / 100.0, mode='polytropic', n_steps=40).run(syngas)
    assert p.T_out == pytest.approx(b['T_out_K'], abs=0.1)
    assert p.eta_is_equiv == pytest.approx(0.75, abs=1e-3)


def test_heater_against_dwsim(heater):
    assert heater.duty_kW == pytest.approx(_ref()['blocks']['heater_H1']['duty_kW'], abs=0.03)
    assert _kW(heater.outlet) == pytest.approx(_ref()['streams']['hot']['energy_flow_kW'], abs=0.05)


# ------------------------------------------------------------------ equilibrium reactor

def test_reactor_extent_and_duty_against_dwsim(reactor):
    """1.1 % above the DWSIM extent, whose product does not satisfy its own expression exactly (see the next test)."""
    b = _ref()['blocks']['reactor_R1']
    assert reactor.extent * 1000.0 == pytest.approx(b['extent_mol_s'], rel=0.02)
    assert reactor.duty_kW == pytest.approx(b['duty_kW'], rel=0.02)
    assert 100.0 * reactor.conversion == pytest.approx(b['conversion_pct']['carbonmonoxide'], abs=1.5)


def test_reactor_composition_against_dwsim(reactor):
    ref = _ref()['streams']['reactor_out']['mole_fraction']
    y = {n: reactor.n_out[n] / reactor.outlet.fn for n in NAMES}
    for n in NAMES:
        assert y[n] == pytest.approx(ref[n], abs=0.01), n


def test_reactor_satisfies_the_expression(reactor):
    assert reactor.ln_K == pytest.approx(lnK_dwsim(525.0), abs=1e-12)
    assert reactor.ln_K_y + reactor.ln_K_phi == pytest.approx(reactor.ln_K, abs=1e-9)


@pytest.mark.parametrize('P_ref,fugacity', [(1.0, True), (1.01325, False), (1.0, False)])
def test_reactor_basis_closest_to_dwsim_is_fugacity_with_1_atm(heater, reactor, P_ref, fugacity):
    """Evidence for the DWSIM convention (tutorial text: fugacities relative to 1 atm). The other three choices give
    extents 1.6 %, 7.6 % and 7.0 % off. A modest difference between P_ref values (0.5 %), but in the right direction."""
    other = EquilibriumReactor(STOICH, lnK=lnK_dwsim, T_out=525.0, P_ref=P_ref, fugacity=fugacity).run(heater.outlet)
    target = _ref()['blocks']['reactor_R1']['extent_mol_s']
    assert abs(reactor.extent * 1000.0 - target) < abs(other.extent * 1000.0 - target)


def test_dwsim_outlet_is_short_of_its_own_expression(dw_outlet):
    """Same observation as in the WGS case: with thermo-sim fugacity coefficients at the DWSIM outlet, ln Q is below
    ln K (DWSIM stopped short of equilibrium), by 0.053 with P_ref = 1.01325 bar."""
    y = _ref()['streams']['reactor_out']['mole_fraction']
    ph = dict(zip(NAMES, dw_outlet.props.phi_v))
    lnQ = sum(v * math.log(y[n] * ph[n] * 80.0 / 1.01325) for n, v in STOICH.items())
    assert lnQ - lnK_dwsim(525.0) == pytest.approx(-0.053, abs=0.01)


def test_reaction_is_exothermic(reactor):
    assert reactor.duty_kW < 0 and 0.0 < reactor.conversion < 1.0


# ------------------------------------------------------------------ cooler and separator, whole chain

def test_cooler_against_dwsim(cooler):
    assert cooler.duty_kW == pytest.approx(_ref()['blocks']['cooler_CL1']['duty_kW'], rel=0.01)
    assert cooler.outlet.props.phase == 'two_phase'


def test_vapor_fraction_against_dwsim(drum):
    """0.580 against 0.5887: the extent difference of the reactor, not the flash (see the isolated tests below)."""
    assert drum.V_frac == pytest.approx(_ref()['blocks']['cooler_CL1']['vapor_mole_fraction'], abs=0.015)


def test_crude_methanol_against_dwsim(drum):
    ref = _ref()['streams']['crude_methanol']
    x = _x(drum.liquid)
    assert x['methanol'] == pytest.approx(ref['mole_fraction']['methanol'], abs=1e-4)
    assert drum.liquid.fn * 1000.0 == pytest.approx(ref['molar_flow_mol_s'], rel=0.015)
    assert x['methanol'] > 0.98                                   # tutorial text: purity 0.99


def test_recycle_gas_against_dwsim(drum):
    ref = _ref()['streams']['recycle_gas']
    y = _x(drum.vapor)
    for n in NAMES:
        assert y[n] == pytest.approx(ref['mole_fraction'][n], abs=1e-3), n
    assert y['hydrogen'] / y['carbonmonoxide'] == pytest.approx(2.0, abs=0.05)     # tutorial text; DWSIM 2.007


def test_tutorial_text_numbers(heater, reactor, cooler, comp):
    t = _ref()['tutorial_text']
    assert -reactor.duty_kW == pytest.approx(t['q_reactor_kW'], abs=1.0)
    assert -cooler.duty_kW == pytest.approx(t['q_cooler_kW'], abs=1.0)
    assert comp.W_ind_kW == pytest.approx(t['w_compressor_kW'], abs=0.1)
    assert 100.0 * reactor.conversion == pytest.approx(t['co_conversion_pct'], abs=1.5)


# ------------------------------------------------------------------ cooler and separator on the DWSIM reactor outlet

def test_isolated_cooler_duty_and_energy_flows(dw_outlet, dw_cooler):
    assert _kW(dw_outlet) == pytest.approx(_ref()['streams']['reactor_out']['energy_flow_kW'], abs=0.1)
    assert dw_cooler.duty_kW == pytest.approx(_ref()['blocks']['cooler_CL1']['duty_kW'], abs=0.1)
    assert _kW(dw_cooler.outlet) == pytest.approx(_ref()['streams']['cooled']['energy_flow_kW'], abs=0.1)


def test_isolated_vapor_fraction_matches_dwsim(dw_drum):
    """The two-phase split of thermo-sim reproduces DWSIM (0.588684) once the composition is the DWSIM one."""
    assert dw_drum.V_frac == pytest.approx(_ref()['blocks']['cooler_CL1']['vapor_mole_fraction'], abs=2e-4)


@pytest.mark.parametrize('key,phase', [('recycle_gas', 'vapor'), ('crude_methanol', 'liquid')])
def test_isolated_separator_outlets_match_dwsim(dw_drum, key, phase):
    ref = _ref()['streams'][key]
    s = getattr(dw_drum, phase)
    x = _x(s)
    for n in NAMES:
        assert x[n] == pytest.approx(ref['mole_fraction'][n], abs=2e-5), n
    assert s.fn * 1000.0 == pytest.approx(ref['molar_flow_mol_s'], rel=1e-4)
    assert s.fm == pytest.approx(ref['mass_flow_kg_s'], rel=2e-4)
    assert _kW(s) == pytest.approx(ref['energy_flow_kW'], abs=0.1)


def test_separator_report_of_dwsim_has_the_mass_flow_labels_swapped():
    """0.0215854 kg/s is the liquid (crude methanol), 0.0104565 kg/s the vapor: the stream table is consistent, the
    separator report is not."""
    r = _ref()
    assert r['streams']['crude_methanol']['mass_flow_kg_s'] == r['blocks']['separator_SEP1']['report_mass_flows_kg_s']['labelled_vapor']
    assert r['streams']['recycle_gas']['mass_flow_kg_s'] == r['blocks']['separator_SEP1']['report_mass_flows_kg_s']['labelled_liquid']


# ------------------------------------------------------------------ internal consistency

def test_mass_balance_over_the_flowsheet(syngas, drum):
    """The reaction conserves mass up to the molar masses in the database (CO + 2 H2 against CH3OH: 8e-6)."""
    assert drum.vapor.fm + drum.liquid.fm == pytest.approx(syngas.fm, rel=1e-4)
    assert drum.mass_error == pytest.approx(0.0, abs=1e-12)


def test_atoms_are_conserved_in_the_reactor(heater, reactor):
    def atoms(n_by_comp):
        out = {}
        for n, v in n_by_comp.items():
            for a, k in FORMULAS[n].items():
                out[a] = out.get(a, 0.0) + k * v
        return out
    a_in = atoms({n: heater.outlet.fn * _x(heater.outlet)[n] for n in NAMES})
    a_out = atoms({n: reactor.n_out[n] for n in NAMES})
    assert sum(v for n, v in reactor.n_out.items() if n not in NAMES) == 0.0     # no other species appear
    for a in a_in:
        assert a_out[a] == pytest.approx(a_in[a], rel=1e-12), a


def test_energy_balance_over_the_flowsheet(syngas, comp, heater, reactor, cooler, drum):
    """Duties and compressor power against the change of energy flow between feed and the two products."""
    supplied = comp.W_ind_kW + heater.duty_kW + reactor.duty_kW + cooler.duty_kW + drum.duty_kW
    gained = _kW(drum.vapor) + _kW(drum.liquid) - _kW(syngas)
    assert supplied == pytest.approx(gained, abs=1e-6)
    assert abs(drum.energy_error_kW) < 1e-6


def test_separator_is_at_cooler_temperature_and_pressure(drum, cooler):
    assert drum.T == pytest.approx(308.0) and drum.P == pytest.approx(cooler.outlet.P)
    assert drum.vapor.T == pytest.approx(308.0, abs=1e-3) and drum.liquid.T == pytest.approx(308.0, abs=1e-3)


# ------------------------------------------------------------------ DWSIM mixing entropy rule (empirical hypothesis)

STREAMS = ['syngas', 'comp_out', 'hot', 'reactor_out', 'recycle_gas', 'crude_methanol']


@pytest.fixture(scope='module')
def free_db(database):
    """Formation-free database so that S has the same reference as DWSIM."""
    db0 = database.copy()
    db0['hf_298_k'], db0['gf_298_k'], db0['n_gas_el'] = 0.0, 0.0, 1.0
    return db0


@pytest.fixture(scope='module')
def molar_mass(free_db):
    return free_db.assign(c=free_db['Component'].astype(str).str.lower()).set_index('c')['MM'].to_dict()


@pytest.mark.parametrize('key', STREAMS)
def test_entropy_against_dwsim_with_empirical_mixing_rule(free_db, dwsim_method, molar_mass, key):
    """HYPOTHESIS (tests/dwsim_conventions.py, docs/validation_dwsim_entropy_mixing.md). Single-phase streams built at
    the DWSIM composition, T and P, including a vapor of H2/CO/CH3OH (molar masses 2, 28, 32) and a liquid at
    99 % methanol. Offsets thermo-sim minus DWSIM are -3.6 to -24.7 kJ/(kmol K). The tolerance, 0.1, was chosen after
    seeing the residuals (0.001 to 0.074; the largest one, the 525 K product, goes with a 0.35 % enthalpy difference of
    the ideal-gas Cp: 26 kJ/kmol / 525 K = 0.05); it is not a pre-declared tolerance."""
    s = _ref()['streams'][key]
    ours = _stream(free_db, dwsim_method, s['mole_fraction'], s['molar_flow_mol_s'] / 1000.0, s['T_K'], s['P_bar'])
    offset = entropy_offset_empirical(s['mole_fraction'], molar_mass)
    assert abs(offset) > 3.0                                     # the rule is doing real work here
    assert ours.props.S_total_mix - offset == pytest.approx(s['molar_entropy_kJ_kmolK'], abs=0.1)


def test_entropy_of_the_two_phase_stream_follows_the_phase_by_phase_rule(free_db, dwsim_method, molar_mass):
    """Prediction written before looking at the result: the offset of a two-phase stream is V off(y) + (1 - V) off(x),
    with the DWSIM vapor fraction and phase compositions (each phase mixes on its own). Outcome: residual -0.0095
    kJ/(kmol K). The single-mixture offset of the overall composition (-24.4) would be wrong by 18. The tolerance, 0.1,
    is the one of the single-phase streams."""
    r = _ref()['streams']
    s = r['cooled']
    ours = _stream(free_db, dwsim_method, s['mole_fraction'], s['molar_flow_mol_s'] / 1000.0, s['T_K'], s['P_bar'])
    assert ours.props.phase == 'two_phase'
    V = s['vapor_mole_fraction']
    off = (V * entropy_offset_empirical(r['recycle_gas']['mole_fraction'], molar_mass)
           + (1.0 - V) * entropy_offset_empirical(r['crude_methanol']['mole_fraction'], molar_mass))
    assert ours.props.S_total_mix - off == pytest.approx(s['molar_entropy_kJ_kmolK'], abs=0.1)
    single = entropy_offset_empirical(s['mole_fraction'], molar_mass)
    assert abs(ours.props.S_total_mix - single - s['molar_entropy_kJ_kmolK']) > 10.0


#----------------------------------------------------------- Explicit comparison
def test_reactor_extent_and_duty_against_dwsim(reactor, cmp):
    b = _ref()['blocks']['reactor_R1']
    cmp('extent mol/s', reactor.extent * 1000.0, b['extent_mol_s'], rel=0.02)
    cmp('duty kW', reactor.duty_kW, b['duty_kW'], rel=0.02)
    cmp('conv CO %', 100 * reactor.conversion, b['conversion_pct']['carbonmonoxide'], abs=1.5)


def test_isolated_vapor_fraction_matches_dwsim(dw_drum, cmp):
    cmp('V_frac', dw_drum.V_frac, _ref()['blocks']['cooler_CL1']['vapor_mole_fraction'], abs=2e-4)


@pytest.mark.parametrize('key,phase', [('recycle_gas', 'vapor'), ('crude_methanol', 'liquid')])
def test_isolated_separator_outlets_match_dwsim(dw_drum, key, phase, cmp):
    ref = _ref()['streams'][key]
    s = getattr(dw_drum, phase)
    x = _x(s)
    for n in NAMES:
        cmp(f'{key} x_{n}', x[n], ref['mole_fraction'][n], abs=2e-5)
    cmp(f'{key} flow mol/s', s.fn * 1000.0, ref['molar_flow_mol_s'], rel=1e-4)
    cmp(f'{key} kW', _kW(s), ref['energy_flow_kW'], abs=0.1)
