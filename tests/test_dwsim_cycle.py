"""Propane refrigeration cycle (compressor -> condenser -> valve -> evaporator) against DWSIM 10.2.12.0.

Stream state, phase split, Z and density are compared stream by stream; duties, compressor power
and COP as differences. Peng-Robinson with the DWSIM propane constants, R = 8.314 and DWSIM's own
ideal-gas Cp equation (fitted with a quartic, see _dwsim_cycle.py). The valve is isenthalpic; for a
pure component it solves the vapor fraction inside the dome.
Reference: reference_data/dwsim/propane_cycle_streams.csv and propane_cycle_units.json.
Tolerances: 2 to 5 times the observed differences, which are limited by the digits DWSIM prints.

Known difference (xfail, see reference_data/dwsim/README_cycle_section.md): DWSIM places the
saturation points of a pure component with its vapor-pressure correlation (252.995 K at 2.45 bar),
thermo-sim with the EOS (253.221 K, confirmed by an independent solution of the equal-fugacity
condition). At the (T, P) of every DWSIM state the Z roots agree to 1e-6 (stream 3), so the EOS is
the same; the streams that start or end on the dome inherit the 0.226 K shift. The evaporator is taken
to saturated vapor (DWSIM: 253 K, 0.005 K above its own dew point).
"""
import numpy as np
import pytest

from thermo_sim import METHODS, Stream, load_database
from thermo_sim.unitops import Valve

from _dwsim_cycle import (STREAMS, UNITS, cp_ideal_diffs, cycle, cycle_packaged, enthalpy_diffs,
                          stream_diffs, unit_diffs)


def _tol(name, key):
    if key == 'dT':
        return 2e-3                       # DWSIM prints 6 digits (252.995, 325.138)
    if key == 'dP':
        return 1e-9
    if key == 'dV':
        return 1e-5                       # 0.337217
    if key in ('rel_Zv', 'rel_Zl', 'rel_rho'):
        return 1e-4
    raise KeyError(key)


SAT_REASON = ("DWSIM places the saturation point with its vapor-pressure correlation: 0.226 K from the "
              "EOS one (252.995 vs 253.221 K at 2.45 bar); see README_cycle_section.md")
SAT_AFFECTED = {('1_evap_out', 'dT'), ('1_evap_out', 'rel_Zv'), ('1_evap_out', 'rel_rho'),
                ('2_comp_out', 'dT'), ('2_comp_out', 'rel_Zv'), ('2_comp_out', 'rel_rho'),
                ('4_valve_out', 'dT'), ('4_valve_out', 'dV'), ('4_valve_out', 'rel_Zv'),
                ('4_valve_out', 'rel_Zl'),
                ('5_evap_out', 'dT'), ('5_evap_out', 'rel_Zv'), ('5_evap_out', 'rel_rho')}
xf_sat = pytest.mark.xfail(reason=SAT_REASON, strict=False)

CASES = [pytest.param(n, k, id=f'{n}-{k}', marks=[xf_sat] if (n, k) in SAT_AFFECTED else [])
         for n in STREAMS.index for k in stream_diffs(n)]


@pytest.mark.parametrize('name,key', CASES)
def test_stream_vs_dwsim(name, key):
    assert stream_diffs(name)[key] == pytest.approx(0.0, abs=_tol(name, key))


UNIT_TOL = {                              # relative to the DWSIM value
    'compressor power [kW]': ('compressor', 'power_kW', 3e-3),       # observed 1.1e-3 (0.226 K inlet offset)
    'condenser duty [kW]': ('condenser', 'duty_kW', 3e-3),           # observed 1.3e-3
    'evaporator duty [kW]': ('evaporator', 'duty_kW', 3e-3),         # 0.226 K x Cp / 11.73 kW = 0.15 %
}


@pytest.mark.parametrize('label', list(UNIT_TOL))
def test_duties_vs_dwsim(label):
    unit, key, tol = UNIT_TOL[label]
    assert unit_diffs()[label] / UNITS[unit][key] == pytest.approx(0.0, abs=tol)


@xf_sat
def test_compressor_outlet_temperature():
    assert unit_diffs()['compressor T_out [K]'] == pytest.approx(0.0, abs=2e-3)


def test_compressor_temperature_rise():
    """The inlet offset cancels in T_out - T_in: 72.141 K (thermo-sim) against 72.143 K (DWSIM)."""
    assert unit_diffs()['compressor T rise [K]'] == pytest.approx(0.0, abs=5e-3)


def test_cop_vs_dwsim():
    assert unit_diffs()['COP'] == pytest.approx(0.0, abs=1.5e-3)


def test_enthalpy_differences_vs_dwsim():
    """H - H(1) differs by about Cp x 0.226 K = 18 kJ/kmol through the offset of stream 1."""
    for k, v in enthalpy_diffs().items():
        assert v == pytest.approx(0.0, abs=20.0), k           # kJ/kmol, on ~15000 kJ/kmol


@pytest.mark.parametrize('name', list(cp_ideal_diffs()))
def test_ideal_gas_cp_is_dwsim_equation(name):
    assert cp_ideal_diffs()[name][2] == pytest.approx(0.0, abs=5e-4)


def test_valve_is_isenthalpic():
    v = cycle()['valve']
    assert v.H_out == pytest.approx(v.H_in, abs=1e-4)             # kJ/kmol
    assert unit_diffs()['valve dH [kJ/kmol]'] == pytest.approx(0.0, abs=1e-4)
    assert 0.0 < v.vapor_fraction_out < 1.0
    assert v.T_out < v.T_in


def test_energy_balance_of_the_cycle():
    """W + Q_cond + Q_evap = fn (H5 - H1): the valve neither adds nor removes energy."""
    L = cycle()
    s1, s5 = L['streams']['1_evap_out'], L['streams']['5_evap_out']
    lhs = L['comp'].W_brake_kW + L['cond'].duty_kW + L['evap'].duty_kW
    assert lhs == pytest.approx(s1.fn * (s5.props.H_total_mix - s1.props.H_total_mix), abs=1e-9)


def test_cycle_closure_like_the_tutorial():
    """Tutorial checks: Q_cond = Q_evap + W within 0.5 % and T5 = T1 within 0.5 %."""
    L = cycle()
    assert -L['cond'].duty_kW == pytest.approx(L['evap'].duty_kW + L['comp'].W_brake_kW, rel=5e-3)
    assert L['streams']['5_evap_out'].T == pytest.approx(L['streams']['1_evap_out'].T, rel=5e-3)
    assert 2.0 < unit_diffs()['COP'] + UNITS['cop']['value'] < 7.0


def test_saturation_temperature_is_the_pr_one():
    """Stream 1 is solved as saturated vapor at 2.45 bar. 253.2214 K is the root of the PR
    equal-fugacity condition with the DWSIM constants (independent solution of the cubic)."""
    assert cycle()['streams']['1_evap_out'].T == pytest.approx(253.2214, abs=2e-3)


def test_liquid_root_matches_dwsim_at_the_condenser_outlet():
    """3_cond_out is away from the dome: Z and density agree to the digits DWSIM prints."""
    d = stream_diffs('3_cond_out')
    assert d['rel_Zl'] == pytest.approx(0.0, abs=1e-5)
    assert d['rel_rho'] == pytest.approx(0.0, abs=1e-5)


# Valve on a mixture (T is solved by the PH flash, which is continuous across the dome)
def test_valve_on_a_mixture_conserves_enthalpy():
    db = load_database()
    s = Stream(description='mix', fm=None, comp_mass=None, fn=0.001,
               comp_mol={'propane': 0.95, 'methane': 0.05}, V_frac=None, T=290.0, P=30.0,
               Tref=298.15, Pref=1.01325, psat_factor=None, gamma=None, component_data=db,
               kij=np.zeros((2, 2)), k_mode='cubic', method=METHODS['PR'], emit_warnings=False)
    v = Valve(2.45).run(s)
    assert v.H_out == pytest.approx(v.H_in, abs=1e-3)
    assert v.P_out == 2.45 and v.T_out < v.T_in
    assert 0.0 < v.vapor_fraction_out < 1.0


def test_cooler_with_vapor_fraction_spec():
    from thermo_sim.unitops import Cooler
    L = cycle()
    assert L['evap'].outlet.props.V_frac == pytest.approx(1.0, abs=1e-9)
    assert L['evap'].T_out == pytest.approx(L['streams']['1_evap_out'].T, abs=1e-6)   # closes the cycle
    liq = Cooler(V_frac=0.0).run(L['streams']['2_comp_out'])                        # saturated liquid at 12.5 bar
    assert liq.outlet.props.V_frac == pytest.approx(0.0, abs=1e-9)
    with pytest.raises(ValueError):
        Cooler(T_out=300.0, V_frac=0.0)
    with pytest.raises(ValueError):
        Cooler()


def test_valve_rejects_pressure_increase():
    with pytest.raises(ValueError):
        Valve(20.0).run(cycle()['streams']['3_cond_out'])


# Packaged database (ideal-gas Cp fitted to CoolProp): the duties differ from DWSIM only through Cp.
PACKAGED_TOL = {'compressor power [kW]': 0.03, 'condenser duty [kW]': 0.03}


@pytest.mark.parametrize('label', list(PACKAGED_TOL))
def test_packaged_database_duties_within_cp_difference(label):
    unit, key, _ = UNIT_TOL[label]
    assert unit_diffs(cycle_packaged())[label] / UNITS[unit][key] == pytest.approx(0.0, abs=PACKAGED_TOL[label])