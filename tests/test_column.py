"""Rigorous column (MESH, Naphtali-Sandholm) against the DWSIM 10.2.12.0 benzene/toluene tutorial.

Reference data: reference_data/dwsim/rigorous_column_benzene_toluene.json (30 stages, feed at stage 15 preheated to
370 K, total condenser, reflux ratio 3, bottoms 50 mol/s, 1 atm, Peng-Robinson with the DWSIM constants).

What is checked
  - the solution closes its own equations (component and energy balances, equilibrium, sum of fractions) to solver
    precision, independently of DWSIM;
  - the whole DWSIM profile (T, L, V, x, y), the duties and the products;
  - that the small remaining difference to DWSIM is the ideal-gas Cp polynomial: with the DWSIM polynomial applied
    through ``h_ig_correction`` (to the stages and to the feed) the agreement improves by a factor of 2 to 10, down to
    the digits DWSIM prints. This is a diagnostic, not a model option.
  - robustness on other feed states, feed stages, reflux ratios and sizes (internal consistency only, no DWSIM data).
DWSIM prints the reboiler duty as a negative number; thermo-sim reports positive heat added.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from thermo_sim import METHODS, Stream, load_database, methods
from thermo_sim.stage_thermo import StageThermo
from thermo_sim.stream import _cp_ext, _cp_gas_coeffs, _cp_gas_limits
from thermo_sim.unitops.column import RigorousColumn

HERE = Path(__file__).resolve().parent.parent / 'reference_data' / 'dwsim'
REF = HERE / 'rigorous_column_benzene_toluene.json'
PURE = HERE / 'pure_component_properties_benzene_toluene.json'
CONSTANTS = HERE / 'dstwu_benzene_toluene.json'
P0 = 1.01325


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


def _feed(database, method, T, z_b=0.5, fn=100.0):
    return Stream('feed', fm=None, comp_mass=None, fn=fn, comp_mol={'benzene': z_b, 'toluene': 1.0 - z_b},
                  V_frac=None, T=T, P=P0, Tref=298.15, Pref=1.01325, psat_factor=None, gamma=None,
                  component_data=database, kij=None, method=method, emit_warnings=False)


@pytest.fixture(scope='module')
def ref():
    return json.loads(REF.read_text())


@pytest.fixture(scope='module')
def feed(database, dwsim_method, ref):
    return _feed(database, dwsim_method, ref['streams']['preheated']['T_K'])


@pytest.fixture(scope='module')
def cp_correction(feed):
    """Ideal-gas enthalpy difference DWSIM minus thermo-sim per component [J/mol], zero at 280 K."""
    pure = json.loads(PURE.read_text())['ideal_gas_cp_dwsim']['coefficients']
    cd = feed.component_data
    C, lo, hi = _cp_gas_coeffs(cd), *_cp_gas_limits(cd)
    names = [str(n).lower() for n in cd['Component']]
    grid = np.arange(280.0, 460.0, 0.25)
    cp_dw = np.array([[(pure[n]['A'] + np.exp(pure[n]['B'] / t + pure[n]['C'] + pure[n]['D'] * t
                                              + pure[n]['E'] * t ** 2)) / 1000.0 for n in names] for t in grid])
    cp_ts = np.array([_cp_ext(C, lo, hi, t) for t in grid])
    d = cp_dw - cp_ts
    cum = np.vstack([np.zeros(len(names)), np.cumsum(0.5 * (d[1:] + d[:-1]) * 0.25, axis=0)])
    return lambda T: np.array([np.interp(T, grid, cum[:, i]) for i in range(len(names))])


@pytest.fixture(scope='module')
def tutorial(feed):
    col = RigorousColumn(30, 15, P0, reflux_ratio=3.0, bottoms_flow=50.0)
    return col, col.solve(feed)


@pytest.fixture(scope='module')
def tutorial_cp(feed, cp_correction):
    col = RigorousColumn(30, 15, P0, reflux_ratio=3.0, bottoms_flow=50.0, h_ig_correction=cp_correction)
    return col.solve(feed)


# ---------------------------------------------------------------------------------------------------------------------
# stage thermodynamics
# ---------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize('T, z_b, phase', [(340.0, 0.5, 'liquid'), (350.0, 0.9, 'liquid'), (390.0, 0.5, 'vapor'),
                                           (400.0, 0.2, 'vapor')])
def test_stage_enthalpy_is_the_stream_enthalpy(database, dwsim_method, T, z_b, phase):
    s = _feed(database, dwsim_method, T, z_b)
    assert str(s.props.phase) == phase
    th = StageThermo(s)
    x = np.array([z_b, 1.0 - z_b])
    h = th.h_liquid(T, P0, x) if phase == 'liquid' else th.h_vapor(T, P0, x)
    assert h == pytest.approx(s.props.H_total_mix, rel=1e-12)


# ---------------------------------------------------------------------------------------------------------------------
# the solution satisfies its own equations
# ---------------------------------------------------------------------------------------------------------------------

def test_converged_and_specifications_met(tutorial):
    _, r = tutorial
    assert r.converged and r.iterations <= 12 and r.residual < 1e-9
    assert r.D == pytest.approx(50.0, rel=1e-10)
    assert r.B == pytest.approx(50.0, rel=1e-10)
    assert r.L[0] == pytest.approx(3.0 * r.D, rel=1e-10)                 # reflux ratio
    assert r.V[1] == pytest.approx(4.0 * r.D, rel=1e-10)                 # total condenser: all the vapor condensed
    assert r.V[0] == 0.0


def test_overall_and_stage_balances(tutorial, feed):
    col, r = tutorial
    z = np.asarray(feed.z, float)
    xD, xB = r.x[0], r.x[-1]
    assert np.allclose(100.0 * z, r.D * xD + r.B * xB, rtol=0, atol=1e-8)
    assert abs(r.energy_balance_error_kW) < 1e-5
    N = len(r.T)
    for j in range(1, N):                               # every stage below the condenser; the reboiler leaves L = B
        lin = r.L[j - 1] * r.x[j - 1]                   # for j = 1 this is the reflux
        vin = r.V[j + 1] * r.y[j + 1] if j < N - 1 else 0.0
        assert np.allclose(lin + vin + col.f[j], r.L[j] * r.x[j] + r.V[j] * r.y[j], rtol=0, atol=1e-8), j


def test_equilibrium_and_sum_of_fractions(tutorial):
    col, r = tutorial
    for j in range(1, len(r.T)):
        lnK = col.th.ln_K(r.T[j], r.P[j], r.x[j], r.y[j])
        assert np.allclose(np.log(r.y[j]) - np.log(r.x[j]), lnK, atol=1e-8), j
        assert r.x[j].sum() == pytest.approx(1.0, abs=1e-12)
        assert r.y[j].sum() == pytest.approx(1.0, abs=1e-8)


def test_condenser_liquid_is_at_its_bubble_point(tutorial):
    col, r = tutorial
    sat = col.th.eng.bubble_T(r.P[0], r.x[0])
    assert r.T[0] == pytest.approx(float(sat.T), abs=1e-7)
    assert np.allclose(r.x[0], r.y[1], atol=1e-12)                       # total condenser: x1 = y2


def test_temperature_profile_is_monotonic(tutorial):
    _, r = tutorial
    assert np.all(np.diff(r.T) > 0)


# ---------------------------------------------------------------------------------------------------------------------
# DWSIM profile, duties and products
# ---------------------------------------------------------------------------------------------------------------------

def _dev(r, prof):
    T, L, V = np.array(prof['T_K']), np.array(prof['L_mol_s']), np.array(prof['V_mol_s'])
    xb, yb = np.array(prof['x_benzene']), np.array(prof['y_benzene'])
    ib = r.names.index('benzene')
    dx = np.abs(r.x[:, ib] - xb)
    dy = np.abs(r.y[:, ib] - yb)
    mx, my = np.minimum(xb, 1 - xb), np.minimum(yb, 1 - yb)
    return dict(T=np.abs(r.T - T).max(), L=np.abs(r.L / L - 1).max(),
                V=np.abs(r.V[1:] / V[1:] - 1).max(), x=dx.max(), y=dy.max(),
                xrel=(dx / mx).max(), yrel=(dy / my).max())


def test_profile_against_dwsim(tutorial, ref):
    """Observed: T 1.2 mK, L and V 7e-5, x and y 4.5e-5 (4e-4 relative on the minority component, which is where DWSIM's
    1e-5 convergence tolerance and the sharp split show)."""
    _, r = tutorial
    d = _dev(r, ref['profile'])
    assert d['T'] < 0.004
    assert d['L'] < 3e-4 and d['V'] < 3e-4
    assert d['x'] < 2e-4 and d['y'] < 2e-4
    assert d['xrel'] < 2e-3 and d['yrel'] < 2e-3


def test_duties_against_dwsim(tutorial, ref):
    _, r = tutorial
    assert r.Qc_kW == pytest.approx(ref['duties']['condenser_kW'], rel=2e-4)
    assert r.Qr_kW == pytest.approx(ref['duties']['reboiler_kW'], rel=2e-4)
    # DWSIM streams: Qr - Qc = E_out - E_in of the column, sign convention positive for heat added
    s = ref['streams']
    e = s['distillate']['energy_flow_kW'] + s['bottoms']['energy_flow_kW'] - s['preheated']['energy_flow_kW']
    assert r.Qr_kW - r.Qc_kW == pytest.approx(e, abs=1.0)


def test_products_against_dwsim(tutorial, ref):
    _, r = tutorial
    for key, stream, stage in (('distillate', r.distillate, 0), ('bottoms', r.bottoms, -1)):
        s = ref['streams'][key]
        assert stream.T == pytest.approx(s['T_K'], abs=0.004)
        assert stream.fn == pytest.approx(s['molar_flow_mol_s'], rel=1e-9)
        assert r.x[stage, 0] == pytest.approx(s['x_benzene'], abs=2e-6)
        assert stream.props.dens_mix == pytest.approx(s['density_kg_m3'], rel=2e-4)
        assert str(stream.props.phase) == 'liquid'


def test_preheated_feed_against_dwsim(feed, ref):
    s = ref['streams']['preheated']
    assert float(feed.V_frac) == pytest.approx(s['vapor_fraction'], abs=1e-4)
    assert float(feed.props.yi[0]) == pytest.approx(s['y_benzene'], abs=1e-5)
    assert float(feed.props.xi[0]) == pytest.approx(s['x_benzene_liquid'], abs=1e-5)


def test_remaining_difference_is_the_ideal_gas_cp(tutorial, tutorial_cp, ref):
    """With the DWSIM ideal-gas Cp polynomial in the stages and in the feed the agreement is 2 to 10 times better and
    reaches the digits DWSIM prints."""
    plain = _dev(tutorial[1], ref['profile'])
    corr = _dev(tutorial_cp, ref['profile'])
    assert corr['T'] < 0.001
    assert corr['L'] < 3e-5 and corr['V'] < 3e-5
    assert corr['x'] < 2e-5 and corr['y'] < 2e-5
    for k in ('T', 'L', 'V', 'x', 'y'):
        assert corr[k] < plain[k], k
    assert tutorial_cp.Qr_kW == pytest.approx(ref['duties']['reboiler_kW'], rel=2e-4)


# ---------------------------------------------------------------------------------------------------------------------
# robustness and guards (internal consistency, no DWSIM data)
# ---------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize('T_feed, stage, R, N, B', [
    (300.0, 15, 3.0, 30, 50.0),        # subcooled liquid, no preheater
    (365.2, 8, 6.0, 30, 50.0),         # at the bubble point, high reflux, sharp split
    (380.0, 22, 1.5, 30, 40.0),        # superheated vapor, low reflux, unequal split
    (370.0, 15, 3.0, 20, 50.0),        # fewer stages
    (300.0, 15, 6.0, 40, 55.0),        # more stages, bottoms above the toluene content
    (370.0, 8, 3.0, 30, 40.0),         # feed above the optimum
])
def test_other_feeds_and_specifications_converge_and_close(database, dwsim_method, T_feed, stage, R, N, B):
    f = _feed(database, dwsim_method, T_feed)
    col = RigorousColumn(N, stage, P0, reflux_ratio=R, bottoms_flow=B)
    r = col.solve(f)
    assert r.converged and r.residual < 1e-9
    assert r.B == pytest.approx(B, rel=1e-10) and r.D == pytest.approx(100.0 - B, rel=1e-9)
    assert r.L[0] == pytest.approx(R * r.D, rel=1e-9)
    assert abs(r.energy_balance_error_kW) < 1e-4
    assert np.allclose(100.0 * np.asarray(f.z), r.D * r.x[0] + r.B * r.x[-1], atol=1e-7)
    assert np.all(np.diff(r.T) > -1e-9)
    assert r.Qc_kW > 0 and r.Qr_kW > 0


def test_more_reflux_improves_purity_and_costs_energy(database, dwsim_method):
    f = _feed(database, dwsim_method, 370.0)
    lo = RigorousColumn(30, 15, P0, reflux_ratio=2.0, bottoms_flow=50.0).solve(f)
    hi = RigorousColumn(30, 15, P0, reflux_ratio=4.0, bottoms_flow=50.0).solve(f)
    assert hi.x[0, 0] > lo.x[0, 0] and hi.x[-1, 1] > lo.x[-1, 1]
    assert hi.Qr_kW > lo.Qr_kW and hi.Qc_kW > lo.Qc_kW


@pytest.mark.parametrize('kwargs, exc', [
    (dict(reflux_ratio=3.0), NotImplementedError),                                  # needs the bottoms flow
    (dict(reflux_ratio=3.0, bottoms_flow=50.0, condenser='partial'), NotImplementedError),
    (dict(reflux_ratio=3.0, bottoms_flow=50.0, pressure_drop=0.2), NotImplementedError),
    (dict(reflux_ratio=3.0, bottoms_flow=50.0, efficiency=0.7), NotImplementedError),
    (dict(reflux_ratio=3.0, bottoms_flow=50.0, solver='wang_henke'), NotImplementedError),
    (dict(reflux_ratio=-1.0, bottoms_flow=50.0), ValueError),
])
def test_unsupported_and_invalid_options(kwargs, exc):
    with pytest.raises(exc):
        RigorousColumn(30, 15, P0, **kwargs)


@pytest.mark.parametrize('stage', [1, 30, 0, 31])
def test_feed_stage_must_be_a_tray(stage):
    with pytest.raises(ValueError):
        RigorousColumn(30, stage, P0, reflux_ratio=3.0, bottoms_flow=50.0)


def test_bottoms_flow_must_be_smaller_than_the_feed(feed):
    with pytest.raises(ValueError):
        RigorousColumn(30, 15, P0, reflux_ratio=3.0, bottoms_flow=100.0).solve(feed)
