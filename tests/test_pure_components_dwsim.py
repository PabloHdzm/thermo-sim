"""Pure benzene and toluene at 1.01325 bar, 300 to 400 K: DWSIM 10.2.12.0, Peng-Robinson with the DWSIM constants.

Reference: reference_data/dwsim/pure_component_properties_benzene_toluene.json.

DWSIM prints sensible enthalpy and entropy (no formation terms), thermo-sim includes them, so the comparison is made
on differences (a constant offset per component and phase). The liquid heat capacity of thermo-sim is 0.4 to 0.8 %
below the one implied by the DWSIM enthalpies; the cause is the ideal gas Cp polynomial (database of thermo-sim against the
DWSIM one) and a test reproduces the liquid difference from it.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad

from thermo_sim import METHODS, Stream, load_database, methods

HERE = Path(__file__).resolve().parent.parent / 'reference_data' / 'dwsim'
REF = HERE / 'pure_component_properties_benzene_toluene.json'
CONSTANTS = HERE / 'dstwu_benzene_toluene.json'
P = 1.01325
NAMES = ('benzene', 'toluene')


def _ref():
    return json.loads(REF.read_text())


@pytest.fixture(scope='module')
def database():
    db = load_database()
    if not {'benzene', 'toluene'} <= set(db['Component'].astype(str).str.lower()):
        pytest.skip('benzene or toluene is not in the properties database')
    return db


@pytest.fixture(scope='module')
def dwsim_method(tmp_path_factory, database):
    consts = json.loads(CONSTANTS.read_text())['constants_dwsim']
    rows = [dict(table='dwsim_bt', Component=n, Tc=c['Tc'], Pc=c['Pc'], omega=c['omega'], Vc=c['Vc'], Zra=c['Zra'],
                 fitted_with='', source='DWSIM 10.2.12.0') for n, c in consts.items()]
    path = tmp_path_factory.mktemp('ov') / 'overrides.csv'
    pd.DataFrame(rows).to_csv(path, index=False)
    old = methods.OVERRIDES_CSV_PATH
    methods.OVERRIDES_CSV_PATH = path
    methods.reload_overrides()
    yield METHODS['PR'].but(constants='dwsim_bt')
    methods.OVERRIDES_CSV_PATH = old
    methods.reload_overrides()


@pytest.fixture(scope='module')
def points(database, dwsim_method):
    """(name, row of the DWSIM table, thermo-sim properties) for every temperature."""
    out = {}
    for n in NAMES:
        out[n] = []
        for row in _ref()[n]:
            s = Stream('s', fm=None, comp_mass=None, fn=0.1, comp_mol={n: 1.0}, V_frac=None, T=row['T_K'], P=P,
                       Tref=298.15, Pref=1.01325, psat_factor=None, gamma=None, component_data=database, kij=None,
                       method=dwsim_method, emit_warnings=False)
            out[n].append((row, s.props))
    return out


@pytest.mark.parametrize('name', NAMES)
def test_phase_and_density(points, name):
    for row, p in points[name]:
        assert p.phase == row['phase'], row['T_K']
        assert p.dens_mix == pytest.approx(row['density_kg_m3'], rel=2e-4), row['T_K']


@pytest.mark.parametrize('name', NAMES)
def test_vapor_enthalpy_and_entropy_differences(points, name):
    vap = [(r, p) for r, p in points[name] if r['phase'] == 'vapor']
    r0, p0 = vap[0]
    for r, p in vap[1:]:
        dh_ts = p.H_total_mix - p0.H_total_mix                      # J/mol
        dh_dw = r['molar_enthalpy_kJ_kmol'] - r0['molar_enthalpy_kJ_kmol']
        assert dh_ts == pytest.approx(dh_dw, rel=0.007, abs=6.0), r['T_K']     # ideal-gas Cp: 0.3 % (toluene 0.35 % low)
        ds_ts = p.S_total_mix - p0.S_total_mix
        ds_dw = r['molar_entropy_kJ_kmol_K'] - r0['molar_entropy_kJ_kmol_K']
        assert ds_ts == pytest.approx(ds_dw, rel=0.007, abs=0.03), r['T_K']


@pytest.mark.parametrize('name', NAMES)
def test_liquid_enthalpy_differences_and_the_known_bias(points, name):
    liq = [(r, p) for r, p in points[name] if r['phase'] == 'liquid']
    r0, p0 = liq[0]
    for r, p in liq[1:]:
        dh_ts = p.H_total_mix - p0.H_total_mix
        dh_dw = r['molar_enthalpy_kJ_kmol'] - r0['molar_enthalpy_kJ_kmol']
        # DWSIM sensible heat is larger by 0.4 to 0.8 % (liquid Cp bias); a regression above 1 % would be new
        assert dh_dw >= dh_ts
        assert dh_ts == pytest.approx(dh_dw, rel=0.01), r['T_K']


@pytest.mark.parametrize('name', NAMES)
def test_heat_of_vaporization_agrees_across_the_phase_change(points, name):
    """The constant offset (formation terms) is the same on both sides of the boiling point, so the latent heat
    plus the sensible heat over the step agrees: difference below 40 J/mol of a ~30 kJ/mol latent heat."""
    pts = points[name]
    i = next(k for k, (r, _) in enumerate(pts) if r['phase'] == 'vapor')
    (rl, pl), (rv, pv) = pts[i - 1], pts[i]
    off_l = pl.H_total_mix - rl['molar_enthalpy_kJ_kmol']
    off_v = pv.H_total_mix - rv['molar_enthalpy_kJ_kmol']
    assert abs(off_v - off_l) < 40.0
    so_l = pl.S_total_mix - rl['molar_entropy_kJ_kmol_K']
    so_v = pv.S_total_mix - rv['molar_entropy_kJ_kmol_K']
    assert abs(so_v - so_l) < 0.2


def test_formation_offsets_are_constants_per_component(points):
    """thermo-sim minus DWSIM in the vapor: benzene +82.86 kJ/mol, toluene +50.03 kJ/mol (formation enthalpy)."""
    for name, expected in (('benzene', 82858.0), ('toluene', 50027.0)):
        offs = [p.H_total_mix - r['molar_enthalpy_kJ_kmol'] for r, p in points[name] if r['phase'] == 'vapor']
        assert all(abs(o - expected) < 10.0 for o in offs), (name, offs)


def _cp_ig_dwsim(name, T):
    c = _ref()['ideal_gas_cp_dwsim']['coefficients'][name]
    return (c['A'] + np.exp(c['B'] / T + c['C'] + c['D'] * T + c['E'] * T ** 2)) / 1000.0          # J/mol/K


@pytest.mark.parametrize('name', NAMES)
def test_liquid_enthalpy_bias_is_the_ideal_gas_heat_capacity(database, dwsim_method, points, name):
    """The difference of the liquid enthalpy rise (DWSIM minus thermo-sim) equals the difference of the integrals of the
    two ideal gas Cp polynomials: the departure terms agree."""
    def cp_ts(T):
        s = Stream('s', fm=None, comp_mass=None, fn=0.1, comp_mol={name: 1.0}, V_frac=None, T=T, P=P, Tref=298.15,
                   Pref=1.01325, psat_factor=None, gamma=None, component_data=database, kij=None, method=dwsim_method,
                   emit_warnings=False)
        return s.props.Cp_ig_mix

    liq = [(r, p) for r, p in points[name] if r['phase'] == 'liquid']
    (r0, p0), (r1, p1) = liq[0], liq[-1]
    observed = (r1['molar_enthalpy_kJ_kmol'] - r0['molar_enthalpy_kJ_kmol']) - (p1.H_total_mix - p0.H_total_mix)
    predicted = quad(lambda t: _cp_ig_dwsim(name, t) - cp_ts(t), r0['T_K'], r1['T_K'])[0]
    assert observed == pytest.approx(predicted, abs=3.0)                     # 36.7 against 36.9 and 64.3 against 64.4
    assert observed > 30.0
