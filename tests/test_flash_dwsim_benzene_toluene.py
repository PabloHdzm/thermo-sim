"""Benzene/toluene flash checks against DWSIM 10.2.12.0 (Peng-Robinson, DWSIM constants, Inside-Out flash).

Reference data: reference_data/dwsim/flash_benzene_toluene.json.
  - saturation_points: bubble and dew temperatures, 5 compositions x 10 pressures (0.5 to 5 bar).
  - pt_flash_points: temperature sweeps through the two-phase window (1.01325 bar at z = 0.1, 0.5, 0.9 and 2 bar at 0.5).

What is compared
  - Bubble and dew temperatures, incipient phase and fugacity coefficients: no convention differences, compared directly.
  - Vapor fraction (lever rule on the DWSIM x and y) and phase compositions in the two-phase region.
  - Stability-trial compositions DWSIM prints for single-phase streams (the same fixed point as in the DSTWU tests).
  - Enthalpy, entropy and density, after removing the known differences:
      * formation terms (constant per component, taken from the pure-component data),
      * the DWSIM mixing-entropy rule (EMPIRICAL HYPOTHESIS, tests/dwsim_conventions.py),
      * the ideal gas heat capacity polynomial (DWSIM: A + exp(B/T + C + D T + E T^2) against the thermo-sim database
        polynomial): the enthalpy and entropy it produces are integrated and removed. Without that last correction thermo-sim
        is 20 to 90 J/mol below DWSIM in enthalpy, with it the residual is a few J/mol.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from thermo_sim import METHODS, Stream, load_database, methods

try:
    from dwsim_conventions import entropy_offset_empirical
except ImportError:                                                   # pragma: no cover
    from tests.dwsim_conventions import entropy_offset_empirical

HERE = Path(__file__).resolve().parent.parent / 'reference_data' / 'dwsim'
FLASH = HERE / 'flash_benzene_toluene.json'
PURE = HERE / 'pure_component_properties_benzene_toluene.json'
CONSTANTS = HERE / 'dstwu_benzene_toluene.json'
NAMES = ('benzene', 'toluene')
MM = {'benzene': 78.1118, 'toluene': 92.1384}


def _flash():
    return json.loads(FLASH.read_text())


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


def _stream(database, method, z_b, T, P, comp=None):
    comp = comp or {'benzene': z_b, 'toluene': 1.0 - z_b}
    return Stream('s', fm=None, comp_mass=None, fn=0.1, comp_mol=comp, V_frac=None, T=T, P=P, Tref=298.15,
                  Pref=1.01325, psat_factor=None, gamma=None, component_data=database, kij=None, method=method,
                  emit_warnings=False)


@pytest.fixture(scope='module')
def corrections(database, dwsim_method):
    """Formation offsets per component and the cumulative ideal gas Cp difference (DWSIM minus thermo-sim)."""
    pure = json.loads(PURE.read_text())
    off = {}
    for n in NAMES:
        r0 = pure[n][0]
        p = _stream(database, dwsim_method, None, r0['T_K'], 1.01325, {n: 1.0}).props
        off[n] = (p.H_total_mix - r0['molar_enthalpy_kJ_kmol'], p.S_total_mix - r0['molar_entropy_kJ_kmol_K'])
    grid = np.linspace(300.0, 450.0, 76)
    dcp = {}
    for n in NAMES:
        c = pure['ideal_gas_cp_dwsim']['coefficients'][n]
        dw = (c['A'] + np.exp(c['B'] / grid + c['C'] + c['D'] * grid + c['E'] * grid ** 2)) / 1000.0
        ts = np.array([_stream(database, dwsim_method, None, t, 1.01325, {n: 1.0}).props.Cp_ig_mix for t in grid])
        dcp[n] = dw - ts

    def bias(n, T, entropy=False):
        t = np.append(grid[grid <= T], T)
        v = np.append(dcp[n][grid <= T], np.interp(T, grid, dcp[n]))
        w = v / t if entropy else v
        return float(np.sum(0.5 * (w[1:] + w[:-1]) * np.diff(t)))

    return off, bias


def _offsets(corr, z_b, T, y=None, x=None, V=None):
    """(H, S) to subtract from thermo-sim to compare with DWSIM: formation terms, mixing rule, Cp polynomial."""
    off, bias = corr
    z = {'benzene': z_b, 'toluene': 1.0 - z_b}
    H = sum(z[n] * (off[n][0] - bias(n, T)) for n in NAMES)
    S = sum(z[n] * (off[n][1] - bias(n, T, True)) for n in NAMES)
    if V is None or not 0.0 < V < 1.0:
        S += entropy_offset_empirical(z, MM)
    else:
        yd, xd = dict(zip(NAMES, y)), dict(zip(NAMES, x))
        S += V * entropy_offset_empirical(yd, MM) + (1.0 - V) * entropy_offset_empirical(xd, MM)
    return H, S


# ---------------------------------------------------------------------------------------------------------------------
# Group 4: bubble and dew points
# ---------------------------------------------------------------------------------------------------------------------

@pytest.fixture(scope='module')
def sat_points(database, dwsim_method):
    out = []
    for r in _flash()['saturation_points']:
        s = _stream(database, dwsim_method, r['z_benzene'], 330.0, r['P_bar'])
        zf = np.array([r['z_benzene'], 1.0 - r['z_benzene']])
        eng = s._engine
        out.append((r, eng, eng.bubble_T(r['P_bar'], zf), eng.dew_T(r['P_bar'], zf), zf))
    return out


def test_bubble_and_dew_temperatures(sat_points):
    for r, _, b, d, _ in sat_points:
        assert b.status == 'ok' and d.status == 'ok'
        assert b.T == pytest.approx(r['T_bubble_K'], abs=0.002), r
        assert d.T == pytest.approx(r['T_dew_K'], abs=0.002), r


def test_incipient_vapor_at_bubble(sat_points):
    for r, _, b, _, _ in sat_points:
        assert float(b.w[0]) == pytest.approx(r['y_benzene_at_bubble'], abs=2e-6), r


def test_fugacity_coefficients_at_bubble(sat_points):
    """phi of the liquid (x = z) and of the DWSIM vapor, at the DWSIM bubble temperature. The liquid tolerance is set by
    the 6 printed digits of the temperature (d ln phi / dT is about 0.03 per K)."""
    for r, eng, _, _, zf in sat_points:
        sub, _, _ = eng._active(zf)
        T, P = r['T_bubble_K'], r['P_bar']
        yv = np.array([r['y_benzene_at_bubble'], 1.0 - r['y_benzene_at_bubble']])
        phiL = np.exp(sub.k.phi(T, P, zf, 'l')[0])
        phiV = np.exp(sub.k.phi_eos(T, P, yv, 'v')[0])
        assert phiL == pytest.approx(r['phi_liquid_at_bubble'], rel=5e-5), r
        assert phiV == pytest.approx(r['phi_vapor_at_bubble'], rel=5e-6), r


@pytest.mark.parametrize('side', ['bubble', 'dew'])
def test_saturated_enthalpy_entropy_density(database, dwsim_method, corrections, side):
    """Saturated liquid at the bubble point and saturated vapor at the dew point (thermo-sim evaluated 2 mK inside the
    single-phase region so that a 0.5 mK temperature difference cannot put a trace of the other phase in the stream)."""
    for r in _flash()['saturation_points']:
        sign, key = (-1.0, 'bubble') if side == 'bubble' else (1.0, 'dew')
        T = r['T_bubble_K' if side == 'bubble' else 'T_dew_K']
        p = _stream(database, dwsim_method, r['z_benzene'], T + sign * 2e-3, r['P_bar']).props
        assert str(p.phase) == ('liquid' if side == 'bubble' else 'vapor'), r
        H, S = _offsets(corrections, r['z_benzene'], T)
        assert p.H_total_mix - H == pytest.approx(r[f'H_{key}_kJ_kmol'], abs=3.0), r
        assert p.S_total_mix - S == pytest.approx(r[f'S_{key}_kJ_kmol_K'], abs=0.02), r
        assert p.dens_mix == pytest.approx(r[f'density_{key}_kg_m3'], rel=2e-4), r


# ---------------------------------------------------------------------------------------------------------------------
# Group 2: PT flash through the two-phase window
# ---------------------------------------------------------------------------------------------------------------------

@pytest.fixture(scope='module')
def pt_points(database, dwsim_method):
    out = []
    for r in _flash()['pt_flash_points']:
        s = _stream(database, dwsim_method, r['z_benzene'], r['T_K'], r['P_bar'])
        zf = np.array([r['z_benzene'], 1.0 - r['z_benzene']])
        x, y = np.array(r['x']), np.array(r['y'])
        V = (zf[0] - x[0]) / (y[0] - x[0]) if abs(y[0] - x[0]) > 1e-12 else 0.0
        V = min(max(V, 0.0), 1.0)
        out.append((r, s, zf, x, y, V))
    return out


def _regions():
    """Region of every PT point from the DWSIM bubble and dew temperatures of the same case (liq, two, vap)."""
    tb = {(0.1, 1.01325): 379.354, (0.5, 1.01325): 365.151, (0.9, 1.01325): 354.941, (0.5, 2.0): 390.413}
    td = {(0.1, 1.01325): 381.702, (0.5, 1.01325): 371.742, (0.9, 1.01325): 357.621, (0.5, 2.0): 396.634}
    out = []
    for r in _flash()['pt_flash_points']:
        k = (r['z_benzene'], r['P_bar'])
        out.append('liq' if r['T_K'] < tb[k] - 1e-3 else 'vap' if r['T_K'] > td[k] + 1e-3 else 'two')
    return out


def test_region_counts():
    reg = _regions()
    assert reg.count('liq') == 35 and reg.count('two') == 15 and reg.count('vap') == 30


def test_phase_and_vapor_fraction(pt_points):
    for (r, s, zf, x, y, V), reg in zip(pt_points, _regions()):
        assert str(s.props.phase) == {'liq': 'liquid', 'vap': 'vapor', 'two': 'two_phase'}[reg], r
        assert float(s.props.V_frac) == pytest.approx(V, abs=6e-4), r


def test_two_phase_compositions(pt_points):
    n = 0
    for (r, s, zf, x, y, V), reg in zip(pt_points, _regions()):
        if reg != 'two':
            continue
        n += 1
        assert float(s.props.xi[0]) == pytest.approx(x[0], abs=1e-4), r
        assert float(s.props.yi[0]) == pytest.approx(y[0], abs=1e-4), r
    assert n == 15


def test_stability_trial_compositions_of_single_phase_streams(pt_points):
    """DWSIM prints, for a single-phase stream, the trial phase of its stability test: for a liquid stream the vapor
    y = z K normalised, for a vapor stream the liquid x = z / K normalised (fixed points of the K = phi_L / phi_V rule)."""
    for (r, s, zf, x, y, V), reg in zip(pt_points, _regions()):
        if reg == 'two':
            continue
        sub, _, _ = s._engine._active(zf)
        T, P = r['T_K'], r['P_bar']
        t = zf.copy()
        for _ in range(300):
            if reg == 'liq':
                K = np.exp(sub.k.phi(T, P, zf, 'l')[0] - sub.k.phi_eos(T, P, t, 'v')[0])
                new = zf * K
            else:
                K = np.exp(sub.k.phi(T, P, t, 'l')[0] - sub.k.phi_eos(T, P, zf, 'v')[0])
                new = zf / K
            new = new / new.sum()
            done = np.abs(new - t).max() < 1e-13
            t = new
            if done:
                break
        target = y if reg == 'liq' else x
        assert float(t[0]) == pytest.approx(target[0], abs=3e-5), (r, reg)


def test_pt_fugacity_coefficients(pt_points):
    for (r, s, zf, x, y, V), reg in zip(pt_points, _regions()):
        sub, _, _ = s._engine._active(zf)
        T, P = r['T_K'], r['P_bar']
        assert np.exp(sub.k.phi(T, P, x, 'l')[0]) == pytest.approx(r['phi_liquid'], rel=5e-5), r
        assert np.exp(sub.k.phi_eos(T, P, y, 'v')[0]) == pytest.approx(r['phi_vapor'], rel=5e-6), r


def test_pt_enthalpy_entropy_density(pt_points, corrections):
    """In the two-phase region the mixture enthalpy and entropy are mole-weighted over the phases. The tolerance there is
    wider because H changes by about 30 kJ/mol over a vapor-fraction range of 1, so a 3e-4 difference in V (a 0.5 mK
    difference in the temperature window) is 8 J/mol."""
    for (r, s, zf, x, y, V), reg in zip(pt_points, _regions()):
        H, S = _offsets(corrections, r['z_benzene'], r['T_K'], y=y, x=x, V=V)
        tolH, tolS, tolR = (10.0, 0.03, 5e-3) if reg == 'two' else (3.0, 0.01, 2e-4)
        assert s.props.H_total_mix - H == pytest.approx(r['H_kJ_kmol'], abs=tolH), (r, reg)
        assert s.props.S_total_mix - S == pytest.approx(r['S_kJ_kmol_K'], abs=tolS), (r, reg)
        assert s.props.dens_mix == pytest.approx(r['density_kg_m3'], rel=tolR), (r, reg)


def test_uncorrected_enthalpy_is_below_dwsim_by_the_cp_bias(pt_points, corrections):
    """Without the ideal gas Cp correction thermo-sim is 20 to 100 J/mol below DWSIM in every region (documented bias,
    0.3 % of a latent heat). If this test fails the bias has changed: review before an override of the Cp polynomial."""
    off, _ = corrections
    for (r, s, zf, x, y, V), reg in zip(pt_points, _regions()):
        z = {'benzene': r['z_benzene'], 'toluene': 1.0 - r['z_benzene']}
        d = s.props.H_total_mix - sum(z[n] * off[n][0] for n in NAMES) - r['H_kJ_kmol']
        assert -100.0 < d < -20.0, (r, reg, d)


# ---------------------------------------------------------------------------------------------------------------------
# Group 3: PH and PS flash
# ---------------------------------------------------------------------------------------------------------------------
# DWSIM returns, for a PH or PS flash entered with the H or S of a PT point, the temperature and vapor fraction of that
# point (ph_ps_roundtrip in the JSON). The PT table is therefore also the DWSIM answer to the PH and PS flashes, and the
# thermo-sim temperature solved from the DWSIM enthalpy or entropy shows the effect of the ideal gas Cp difference.

def _case_point(case):
    pts = [q for q in _flash()['pt_flash_points']
           if q['z_benzene'] == case['z_benzene'] and q['P_bar'] == case['P_bar'] and abs(q['T_K'] - case['T_K']) < 2e-3]
    assert len(pts) == 1
    return pts[0]


def _vapor_fraction(r):
    x, y = r['x'], r['y']
    z = r['z_benzene']
    return min(max((z - x[0]) / (y[0] - x[0]), 0.0), 1.0) if abs(y[0] - x[0]) > 1e-12 else 0.0


def test_dwsim_ph_ps_roundtrip_matches_the_pt_table():
    for c in _flash()['ph_ps_roundtrip']['cases']:
        r = _case_point(c)
        mm = c['z_benzene'] * MM['benzene'] + (1.0 - c['z_benzene']) * MM['toluene']
        assert c['dwsim_T_K'] == pytest.approx(r['T_K'], abs=1e-3)
        assert c['dwsim_vapor_fraction'] == pytest.approx(_vapor_fraction(r), abs=1e-4)
        assert c['H_kJ_kg'] == pytest.approx(r['H_kJ_kmol'] / mm, abs=0.011)
        assert c['dwsim_S_kJ_kg_K'] == pytest.approx(r['S_kJ_kmol_K'] / mm, abs=2e-5)


def _solve_T(database, method, z_b, P, T0, target, what):
    from scipy.optimize import brentq

    def f(T):
        p = _stream(database, method, z_b, T, P).props
        return (p.H_total_mix if what == 'H' else p.S_total_mix) - target
    T = brentq(f, T0 - 3.0, T0 + 3.0, xtol=1e-8)
    return T, float(_stream(database, method, z_b, T, P).props.V_frac)


@pytest.mark.parametrize('what', ['H', 'S'])
def test_ph_ps_flash_from_dwsim_state(database, dwsim_method, corrections, what):
    """thermo-sim solves T from the DWSIM H or S. Formation terms and the mixing-entropy rule removed only: T comes out
    0.25 to 0.5 K high in a single phase (the ideal gas Cp polynomial) and the vapor fraction 0.1 to 0.2 % high in the
    two-phase window. With the Cp polynomial removed as well the state is recovered."""
    off, _ = corrections
    for c in _flash()['ph_ps_roundtrip']['cases']:
        r = _case_point(c)
        z_b, P, T0 = r['z_benzene'], r['P_bar'], r['T_K']
        V = _vapor_fraction(r)
        zd = {'benzene': z_b, 'toluene': 1.0 - z_b}
        full_H, full_S = _offsets(corrections, z_b, T0, y=r['y'], x=r['x'], V=V)
        raw_H = sum(zd[n] * off[n][0] for n in NAMES)
        # raw offsets: the same formation and mixing terms without the Cp polynomial
        bias_S = sum(zd[n] * corrections[1](n, T0, True) for n in NAMES)
        raw_S = full_S + bias_S
        value = r['H_kJ_kmol'] if what == 'H' else r['S_kJ_kmol_K']
        raw_target = value + (raw_H if what == 'H' else raw_S)
        full_target = value + (full_H if what == 'H' else full_S)

        T_raw, V_raw = _solve_T(database, dwsim_method, z_b, P, T0, raw_target, what)
        T_full, V_full = _solve_T(database, dwsim_method, z_b, P, T0, full_target, what)
        two_phase = 0.0 < V < 1.0
        if two_phase:
            assert 0.0 < T_raw - T0 < 0.02, (c, T_raw - T0)
            assert 8e-4 < V_raw - V < 3e-3, (c, V_raw - V)
            assert abs(V_full - V) < 6e-4, (c, V_full - V)
        else:
            assert 0.2 < T_raw - T0 < 0.65, (c, T_raw - T0)
            assert abs(T_full - T0) < 0.03, (c, T_full - T0)