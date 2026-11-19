"""Recycle solver and the DWSIM "Recycle Loops" tutorial (water, Steam Tables, IAPWS-IF97).

Part 1 tests the solver on loops with a known fixed point (no database).
Part 2 reproduces the flowsheet Fresh-Feed + Recycle -> heater (350 K) -> 70/30 split -> Recycle, built only with
the Stream algebra (`+`, `split`) and `Cooler` as heater, and compares with DWSIM 10.2.12.0
(reference_data/dwsim/recycle_loop_water.json).

DWSIM enthalpy has its own reference, so only differences are compared (heater duty). Water uses the gamma-phi
route (k_mode='gamma'); its liquid Cp is about 0.6 % above IAPWS-IF97, which sets the duty tolerance.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from thermo_sim import METHODS, Stream, load_database
from thermo_sim.unitops import Cooler
from thermo_sim.unitops.recycle import Recycle

REF = Path(__file__).resolve().parent.parent / 'reference_data' / 'dwsim' / 'recycle_loop_water.json'
METHOD_NAMES = ['legacy', 'wegstein', 'broyden']


def _ref():
    return json.loads(REF.read_text())


# ------------------------------------------------------------------ part 1: solver, closed form

@pytest.mark.parametrize('method', METHOD_NAMES)
def test_linear_loop_fixed_point(method):
    """x = a + b x has the fixed point a / (1 - b); with a 30 % recycle b = 0.3."""
    x, it, ok = Recycle(method=method, max_iter=200).solve_vector(lambda x: 1.0 + 0.3 * x, [0.0])
    assert ok
    assert x[0] == pytest.approx(1.0 / 0.7, rel=1e-9)


@pytest.mark.parametrize('method', METHOD_NAMES)
def test_coupled_nonlinear_loop(method):
    g = lambda x: np.array([np.cos(x[0]) + 0.1 * x[1], 0.5 * np.sin(x[0]) + 0.4 * x[1] + 1.0])
    x, _, ok = Recycle(method=method, max_iter=300).solve_vector(g, [0.0, 0.0])
    assert ok
    assert np.allclose(g(x), x, atol=1e-9)


def test_accelerations_need_fewer_iterations_for_a_slow_loop():
    """Recycle ratio 0.9: direct substitution is slow, the accelerations are not."""
    g = lambda x: 1.0 + 0.9 * x
    n = {m: Recycle(method=m, max_iter=500).solve_vector(g, [0.0])[1] for m in METHOD_NAMES}
    assert n['wegstein'] < n['legacy'] / 3
    assert n['broyden'] < n['legacy'] / 3


def test_smoothing_damps_without_changing_the_solution():
    g = lambda x: 1.0 + 0.3 * x
    x1, n1, _ = Recycle(smoothing=1.0, max_iter=300).solve_vector(g, [0.0])
    x2, n2, ok = Recycle(smoothing=0.5, max_iter=300).solve_vector(g, [0.0])
    assert ok and x2[0] == pytest.approx(x1[0], rel=1e-8)
    assert n2 > n1


def test_not_converged_is_reported():
    _, it, ok = Recycle(max_iter=3).solve_vector(lambda x: 1.0 + 0.9 * x, [0.0])
    assert not ok and it == 3


@pytest.mark.parametrize('kw', [dict(method='newton'), dict(smoothing=0.0), dict(smoothing=1.5),
                                dict(max_iter=0), dict(tol_mass=0.0)])
def test_invalid_settings(kw):
    with pytest.raises(ValueError):
        Recycle(**kw)


# ------------------------------------------------------------------ part 2: DWSIM tutorial

def _fresh():
    i = _ref()['inputs']
    return Stream('Fresh-Feed', fm=i['fresh_mass_flow_kg_s'], comp_mass={'water': 1.0}, fn=None, comp_mol=None,
                  V_frac=None, T=i['fresh_T_K'], P=i['fresh_P_bar'], Tref=298.15, Pref=1.01325, psat_factor=None,
                  gamma=None, component_data=load_database(), kij=None, method=METHODS['PR'], k_mode='gamma',
                  emit_warnings=False)


def _flowsheet(fresh):
    """MIX-1 (Stream +), H-1 (Cooler as heater), SP-1 (Stream.split). Stores the last pass in `state`."""
    i = _ref()['inputs']
    state = {}

    def loop(rec):
        mixed = fresh if rec is None else fresh + rec            # DWSIM starts with an empty recycle
        heater = Cooler(i['heater_T_out_K'], dP=i['heater_dP_bar'], allow_condensation=True).run(mixed)
        product, recycle = heater.outlet.split([i['splitter_product_fraction']])
        state.update(mixed=mixed, heater=heater, product=product, recycle=recycle)
        return recycle

    return loop, state


@pytest.fixture(scope='module')
def plant():
    """Tight tolerances so that the comparison with the closed form is not limited by the stopping rule."""
    fresh = _fresh()
    loop, state = _flowsheet(fresh)
    result = Recycle(method='legacy', tol_T=1e-6, tol_P=1e-6, tol_mass=1e-7, max_iter=100).solve(loop)
    loop(result.stream)                                          # state with the converged recycle
    return fresh, result, dict(state)


def test_converges_with_dwsim_tolerances():
    t = _ref()['recycle_block']['tolerances']
    loop, _ = _flowsheet(_fresh())
    r = Recycle(tol_T=t['temperature_K'], tol_P=t['pressure_bar'], tol_mass=t['mass_flow_kg_s']).solve(loop)
    assert r.converged
    assert r.errors['T'] <= t['temperature_K'] and r.errors['P'] <= t['pressure_bar']
    assert r.errors['mass_flow'] <= t['mass_flow_kg_s']
    assert r.iterations < 15


def test_dwsim_final_errors_are_below_its_tolerances():
    """Consistency of the stored reference: the DWSIM residuals sit well inside the tolerances."""
    rb = _ref()['recycle_block']
    assert rb['final_errors']['temperature_K'] < rb['tolerances']['temperature_K']
    assert rb['final_errors']['mass_flow_kg_s'] < rb['tolerances']['mass_flow_kg_s']
    assert rb['final_errors']['pressure_bar'] == 0.0


def test_closed_form_flows(plant):
    """70/30 split: the product equals the fresh feed, the mixer carries 1/0.7 and the recycle 0.3/0.7."""
    fresh, result, st = plant
    assert st['product'].fm == pytest.approx(fresh.fm, rel=1e-6)
    assert st['mixed'].fm == pytest.approx(fresh.fm / 0.7, rel=1e-6)
    assert result.stream.fm == pytest.approx(0.3 * fresh.fm / 0.7, rel=1e-6)


def test_flows_against_dwsim(plant):
    _, result, st = plant
    s = _ref()['streams']
    assert st['mixed'].fm == pytest.approx(s['mixed']['mass_flow_kg_s'], abs=1e-4)
    assert st['product'].fm == pytest.approx(s['product']['mass_flow_kg_s'], abs=1e-4)
    assert result.stream.fm == pytest.approx(s['recycle']['mass_flow_kg_s'], abs=1e-4)
    assert st['mixed'].fn * 1000.0 == pytest.approx(s['mixed']['molar_flow_mol_s'], rel=1e-4)
    assert st['product'].fn * 1000.0 == pytest.approx(s['product']['molar_flow_mol_s'], rel=1e-4)


def test_mixed_temperature(plant):
    """Adiabatic mixing of 1 kg/s at 300 K and 0.4286 kg/s at 350 K: an energy balance, nearly independent of Cp."""
    _, _, st = plant
    assert st['mixed'].T == pytest.approx(_ref()['streams']['mixed']['T_K'], abs=0.1)


def test_heater_outlet_and_duty(plant):
    _, _, st = plant
    h = st['heater']
    assert h.T_out == pytest.approx(_ref()['streams']['heated']['T_K'], abs=1e-9)
    assert h.duty_kW == pytest.approx(_ref()['results']['heater_duty_kW'], rel=0.012)


def test_all_streams_are_liquid(plant):
    _, result, st = plant
    for s in (st['mixed'], st['heater'].outlet, st['product'], result.stream):
        assert s.props.phase == 'liquid'


def test_pressure_is_constant(plant):
    _, result, st = plant
    assert st['mixed'].P == pytest.approx(1.01325, abs=1e-9)
    assert result.stream.P == pytest.approx(1.01325, abs=1e-9)
    assert result.errors['P'] == 0.0


def test_overall_energy_balance(plant):
    """Steady state: enthalpy of the fresh feed plus the heater duty leaves with the product."""
    fresh, _, st = plant
    H_in = fresh.fn * fresh.props.H_total_mix + st['heater'].duty_kW
    H_out = st['product'].fn * st['product'].props.H_total_mix
    assert H_out == pytest.approx(H_in, abs=1e-3)


def test_recycle_is_consistent_with_the_flowsheet(plant):
    """The returned recycle is the one the flowsheet calculates from itself."""
    _, result, st = plant
    assert result.stream.fm == pytest.approx(st['recycle'].fm, rel=1e-6)
    assert result.stream.T == pytest.approx(st['recycle'].T, abs=1e-6)
    assert result.errors['mass_flow'] <= result.tolerances['mass_flow']


@pytest.mark.parametrize('method', ['wegstein', 'broyden'])
def test_accelerated_methods_give_the_same_loop(plant, method):
    fresh, result, st = plant
    loop, state = _flowsheet(fresh)
    r = Recycle(method=method, tol_T=1e-6, tol_P=1e-6, tol_mass=1e-7, max_iter=100).solve(loop)
    assert r.converged
    assert r.stream.fm == pytest.approx(result.stream.fm, rel=1e-6)
    assert state['heater'].duty_kW == pytest.approx(st['heater'].duty_kW, rel=1e-6)


def test_history_is_recorded(plant):
    _, result, _ = plant
    assert result.history[0][0] == 1
    assert len(result.history) == result.iterations
    assert result.history[-1][3] == pytest.approx(result.errors['mass_flow'])
    assert result.history[0][3] > result.history[-1][3]
