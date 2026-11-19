"""Water-gas shift (CO + H2O <-> CO2 + H2) in three isothermal reactors: DWSIM "Reaction Systems" tutorial.

Reference: DWSIM 10.2.12.0 (Classic UI), Peng-Robinson, feed 10 mol/s at 600 K and 10.1325 bar, 40/40/10/10 mol %
CO/H2O/CO2/H2 (reference_data/dwsim/wgs_reactors.json).

* StoichReactor     : DWSIM Conversion reactor, 80 % of CO.
* EquilibriumReactor: DWSIM Equilibrium reactor, ln K = 4577.8/T - 4.33 (vapor, activity basis).
* GibbsReactor      : DWSIM Gibbs reactor (element balance, no reaction given).

DWSIM energy flow includes the formation enthalpy, as H_total_mix does here, so the duty is compared directly.
The critical constants of CO, CO2 and H2 are those of DWSIM; those of water are not stored yet (the database ones
are used), and carbon monoxide must be in the database (the tests are skipped until it is).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from thermo_sim import METHODS, Stream, load_database, methods
from thermo_sim.unitops.gibbs_reactor import GibbsReactor
from thermo_sim.unitops.reactor import EquilibriumReactor, StoichReactor

try:                                                          # tests/ may or may not be a package
    from dwsim_conventions import entropy_offset_empirical
except ImportError:
    from tests.dwsim_conventions import entropy_offset_empirical

REF = Path(__file__).resolve().parent.parent / 'reference_data' / 'dwsim' / 'wgs_reactors.json'
NAMES = ['water', 'carbonmonoxide', 'carbondioxide', 'hydrogen']
STOICH = {'carbonmonoxide': -1.0, 'water': -1.0, 'carbondioxide': 1.0, 'hydrogen': 1.0}


def _ref():
    return json.loads(REF.read_text())


def lnK_dwsim(T):
    return 4577.8 / T - 4.33


@pytest.fixture(scope='module')
def database():
    db = load_database()
    if 'carbonmonoxide' not in db['Component'].astype(str).str.lower().tolist():
        pytest.skip('carbon monoxide is not in the properties database yet')
    return db


@pytest.fixture(scope='module')
def dwsim_method(tmp_path_factory):
    """PR with the DWSIM critical constants of CO, CO2 and H2 (overrides table 'dwsim_wgs'; water: database)."""
    rows = [dict(table='dwsim_wgs', Component=n, Tc=c['Tc'], Pc=c['Pc'], omega=c['omega'], Vc=c['Vc'], Zra=c['Zra'],
                 fitted_with='', source='DWSIM 10.2.12.0')
            for n, c in _ref()['constants_dwsim'].items() if isinstance(c, dict)]
    path = tmp_path_factory.mktemp('ov') / 'overrides.csv'
    pd.DataFrame(rows).to_csv(path, index=False)
    old = methods.OVERRIDES_CSV_PATH
    methods.OVERRIDES_CSV_PATH = path
    methods.reload_overrides()
    yield METHODS['PR'].but(constants='dwsim_wgs')
    methods.OVERRIDES_CSV_PATH = old
    methods.reload_overrides()


def _kij():
    k = pd.DataFrame(np.zeros((4, 4)), index=NAMES, columns=NAMES)
    for pair, v in _ref()['kij_pr_dwsim'].items():
        a, b = pair.split('-')
        k.loc[a, b] = k.loc[b, a] = v
    return k


def _stream(db, method, x, fn, T=600.0, P=10.1325):
    return Stream('s', fm=None, comp_mass=None, fn=fn, comp_mol=x, V_frac=None, T=T, P=P, Tref=298.15, Pref=1.01325,
                  psat_factor=None, gamma=None, component_data=db, kij=_kij(), method=method, emit_warnings=False)


@pytest.fixture(scope='module')
def feed(database, dwsim_method):
    i = _ref()['inputs']
    return _stream(database, dwsim_method, i['mole_fraction'], i['molar_flow_mol_s'] / 1000.0, i['T_K'], i['P_bar'])


@pytest.fixture(scope='module')
def conv(feed):
    return StoichReactor(STOICH, conversion=0.8, key='carbonmonoxide').run(feed)


@pytest.fixture(scope='module')
def equil(feed):
    return EquilibriumReactor(STOICH, lnK=lnK_dwsim).run(feed)


@pytest.fixture(scope='module')
def equil_formation(feed):
    """Equilibrium with K(T) from the formation data of the database (what the Gibbs reactor uses)."""
    return EquilibriumReactor(STOICH).run(feed)


@pytest.fixture(scope='module')
def gibbs(feed):
    return GibbsReactor().run(feed)


def _y(res):
    return {n: res.n_out[n] / res.outlet.fn for n in NAMES}


def _mol_s(res, n):
    return res.n_out[n] * 1000.0


# ------------------------------------------------------------------ feed

def test_feed(feed):
    r = _ref()['streams']['feed']
    assert feed.props.phase == 'vapor'
    assert feed.fm == pytest.approx(r['mass_flow_kg_s'], rel=1e-4)


def test_feed_energy_flow_includes_formation(feed):
    """-1704.58 kW = sum(z Hf) + sensible enthalpy of the feed."""
    assert feed.fn * feed.props.H_total_mix == pytest.approx(_ref()['streams']['feed']['energy_flow_kW'], abs=0.2)


# ------------------------------------------------------------------ Conversion reactor

def test_conversion_flows(conv):
    s = _ref()['streams']['conversion_product']['molar_flow_by_component_mol_s']
    for n in NAMES:
        assert _mol_s(conv, n) == pytest.approx(s[n], abs=1e-9), n
    assert conv.extent * 1000.0 == pytest.approx(3.2, abs=1e-12)


def test_conversion_is_isothermal_and_isobaric(conv):
    assert conv.T_out == 600.0 and conv.P_out == pytest.approx(10.1325, abs=1e-12)
    assert conv.outlet.props.phase == 'vapor'


def test_conversion_duty(conv):
    assert conv.duty_kW == pytest.approx(_ref()['blocks']['conversion']['energy_flow_kW'], rel=2e-3)


def test_conversion_product_energy_flow(conv):
    assert conv.outlet.fn * conv.outlet.props.H_total_mix == pytest.approx(
        _ref()['streams']['conversion_product']['energy_flow_kW'], abs=0.3)


def test_tutorial_text_hydrogen_fraction_is_042(conv):
    """The tutorial text says about 0.28; the product has the 0.10 of the feed plus the 0.32 produced."""
    assert _y(conv)['hydrogen'] == pytest.approx(0.42, abs=1e-12)
    assert _y(conv)['carbonmonoxide'] == pytest.approx(0.08, abs=1e-12)


# ------------------------------------------------------------------ Equilibrium reactor

def test_equilibrium_composition_against_dwsim(equil):
    ref = _ref()['streams']['equilibrium_product']['mole_fraction']
    for n in NAMES:
        assert _y(equil)[n] == pytest.approx(ref[n], abs=5e-4), n


def test_equilibrium_extent_and_conversion(equil):
    b = _ref()['blocks']['equilibrium']
    assert equil.extent * 1000.0 == pytest.approx(b['extent_mol_s'], abs=0.01)
    assert 100.0 * equil.conversion == pytest.approx(b['conversion_pct']['carbonmonoxide'], abs=0.2)


def test_equilibrium_duty(equil):
    assert equil.duty_kW == pytest.approx(_ref()['blocks']['equilibrium']['energy_flow_kW'], rel=2e-3)


def test_equilibrium_satisfies_the_expression(equil):
    """sum(nu ln y) + sum(nu ln phi) = ln K(T) with the DWSIM expression."""
    assert equil.ln_K == pytest.approx(lnK_dwsim(600.0), abs=1e-12)
    assert equil.ln_K_y + equil.ln_K_phi == pytest.approx(equil.ln_K, abs=1e-9)


def test_equilibrium_fugacity_coefficients_match_dwsim(equil):
    """DWSIM sum(nu ln phi) from its stored fugacity coefficients: +0.0111."""
    ph = _ref()['fugacity_coefficient_vapor_equilibrium_product']
    dw = sum(v * np.log(ph[n]) for n, v in STOICH.items())
    assert equil.ln_K_phi == pytest.approx(dw, abs=2e-3)


def test_dwsim_equilibrium_product_vs_its_own_expression():
    """Known difference in the reference: the DWSIM product does not satisfy its own expression exactly
    (ln K from its mole fractions and fugacity coefficients is 3.3066 against 3.2997). Documents the size."""
    r = _ref()
    y = r['streams']['equilibrium_product']['mole_fraction']
    ph = r['fugacity_coefficient_vapor_equilibrium_product']
    lnK = sum(v * (np.log(y[n]) + np.log(ph[n])) for n, v in STOICH.items())
    assert lnK - lnK_dwsim(600.0) == pytest.approx(0.0069, abs=5e-4)


def test_equilibrium_is_exothermic_and_favors_products(equil):
    assert equil.ln_K > 0 and equil.duty_kW < 0 and equil.conversion > 0.5


# ------------------------------------------------------------------ Gibbs reactor

def test_gibbs_composition_against_dwsim(gibbs):
    """DWSIM Gibbs uses its own thermochemistry (implied ln K 3.337); thermo-sim uses NIST formation data and the
    database Cp (3.348): compositions agree within 4e-4."""
    ref = _ref()['streams']['gibbs_product']['mole_fraction']
    for n in NAMES:
        assert _y(gibbs)[n] == pytest.approx(ref[n], abs=1e-3), n


def test_gibbs_element_balance(gibbs):
    """DWSIM element matrix for 10 mol/s of feed: 10 H, 10 O, 5 C (mol/s of atoms)."""
    e = _ref()['inputs']['gibbs']['element_matrix_mol_s']
    for el in ('H', 'O', 'C'):
        assert gibbs.elements_in[el] * 1000.0 == pytest.approx(e[el], rel=1e-9)
    assert gibbs.element_residue < 1e-12


def test_gibbs_conversions_against_dwsim(gibbs):
    c = _ref()['blocks']['gibbs']['conversion_pct']
    assert 100.0 * gibbs.conversions['carbonmonoxide'] == pytest.approx(c['carbonmonoxide'], abs=0.3)
    assert 100.0 * gibbs.conversions['water'] == pytest.approx(c['water'], abs=0.3)
    assert 100.0 * gibbs.conversions['hydrogen'] == pytest.approx(c['hydrogen'], abs=1.0)


def test_gibbs_duty(gibbs):
    assert gibbs.duty_kW == pytest.approx(_ref()['blocks']['gibbs']['energy_flow_kW'], rel=3e-3)


def test_gibbs_equals_equilibrium_with_the_same_thermochemistry(gibbs, equil_formation):
    """Same K(T) (formation data) and same phi: two different algorithms, one answer."""
    for n in NAMES:
        assert gibbs.n_out[n] == pytest.approx(equil_formation.n_out[n], rel=1e-8, abs=1e-14), n
    assert gibbs.duty_kW == pytest.approx(equil_formation.duty_kW, rel=1e-8)


def test_gibbs_ideal_gas_option_gives_the_ideal_equilibrium(feed):
    g = GibbsReactor(fugacity=False).run(feed)
    e = EquilibriumReactor(STOICH, fugacity=False).run(feed)
    for n in NAMES:
        assert g.n_out[n] == pytest.approx(e.n_out[n], rel=1e-8, abs=1e-14)


def test_gibbs_with_an_extra_species_forms_methane(feed, gibbs):
    """Allowing CH4 (absent from the feed) drives methanation at 600 K, 10 bar: CO falls below the WGS-only value."""
    g = GibbsReactor(components=NAMES + ['methane']).run(feed)
    assert g.n_out['methane'] > 1e-4
    assert g.n_out['carbonmonoxide'] < gibbs.n_out['carbonmonoxide']
    assert g.element_residue < 1e-12


def test_gibbs_temperature_and_pressure_options(feed):
    g = GibbsReactor(T_out=800.0, dP=1.0).run(feed)
    assert g.T_out == 800.0 and g.P_out == pytest.approx(9.1325)
    assert g.element_residue < 1e-12
    assert g.conversions['carbonmonoxide'] < GibbsReactor().run(feed).conversions['carbonmonoxide']   # exothermic


def test_gibbs_input_errors(feed):
    with pytest.raises(ValueError, match='not in `components`'):
        GibbsReactor(components=['water', 'carbonmonoxide']).run(feed)
    with pytest.raises(ValueError, match='not in the database'):
        GibbsReactor(components=NAMES + ['foo']).run(feed)
    with pytest.raises(ValueError, match='interior point'):
        GibbsReactor(components=NAMES + ['nitrogen']).run(feed)
    with pytest.raises(ValueError, match='dP'):
        GibbsReactor(dP=-1.0)


# ------------------------------------------------------------------ three reactors together

def test_conversion_equilibrium_gibbs_are_close(conv, equil, gibbs):
    """Tutorial claim: equilibrium and Gibbs agree closely; the 80 % conversion is near the equilibrium value."""
    assert abs(equil.conversion - gibbs.conversions['carbonmonoxide']) < 5e-3
    assert abs(conv.conversion - equil.conversion) < 5e-3


def test_atoms_are_conserved_in_all_reactors(conv, equil, gibbs):
    F = {'carbonmonoxide': (1, 0, 1), 'water': (0, 2, 1), 'carbondioxide': (1, 0, 2), 'hydrogen': (0, 2, 0)}   # C, H, O
    for r in (conv, equil, gibbs):
        a_in = np.sum([np.array(F[n]) * r.n_in[n] for n in NAMES], axis=0)
        a_out = np.sum([np.array(F[n]) * r.n_out[n] for n in NAMES], axis=0)
        assert a_out == pytest.approx(a_in, rel=1e-9)


def test_duty_is_the_enthalpy_change(feed, conv, equil, gibbs):
    for r in (conv, equil, gibbs):
        assert r.duty_kW == pytest.approx(r.outlet.fn * r.outlet.props.H_total_mix - feed.fn * feed.props.H_total_mix,
                                          rel=1e-12)


# ------------------------------------------------------------------ DWSIM mixing entropy rule (empirical hypothesis)

@pytest.mark.parametrize('stream,key', [('feed', 'feed'), ('conversion_product', 'conv'),
                                        ('equilibrium_product', 'eq'), ('gibbs_product', 'gibbs')])
def test_entropy_against_dwsim_with_empirical_mixing_rule(database, dwsim_method, stream, key):
    """HYPOTHESIS (tests/dwsim_conventions.py, docs/validation_dwsim_entropy_mixing.md): DWSIM mixture entropy is
    reproduced by a mass-basis ideal mixing term. Here the thermo-sim minus DWSIM entropy is -19 to -30 kJ/(kmol K)
    (H2 with CO2: a large molar-mass contrast, the hardest case listed in the doc) and the rule predicts it within
    0.025. The tolerance (0.05) was set after seeing these residuals: they equal about dH/T of the 0.1 % enthalpy
    difference (ideal-gas Cp). Formation-free database so that S has the same reference as DWSIM."""
    db0 = database.copy()
    db0['hf_298_k'], db0['gf_298_k'], db0['n_gas_el'] = 0.0, 0.0, 1.0
    s = _ref()['streams'][stream]
    M = db0.assign(c=db0['Component'].astype(str).str.lower()).set_index('c')['MM'].to_dict()
    ours = _stream(db0, dwsim_method, s['mole_fraction'], s['molar_flow_mol_s'] / 1000.0)
    offset = entropy_offset_empirical(s['mole_fraction'], M)
    assert abs(offset) > 15.0                                     # the rule is doing real work here
    assert ours.props.S_total_mix - offset == pytest.approx(s['molar_entropy_kJ_kmolK'], abs=0.05)
    assert ours.props.H_total_mix == pytest.approx(s['molar_enthalpy_kJ_kmol'], abs=15.0)    # sensible, 0.15 %
