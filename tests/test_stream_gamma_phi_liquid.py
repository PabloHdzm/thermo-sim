"""Liquid phase of Stream in gamma-phi mode: no PR departure in H and Cp, pressure term only in S."""
import pytest

from thermo_sim import Stream, load_database


@pytest.fixture(scope='module')
def water():
    db = load_database()
    return Stream('water', fm=1.0, comp_mass=None, fn=None, comp_mol={'water': 1.0}, V_frac=None,
                  T=300.0, P=3.0, Tref=298.15, Pref=1.01325, psat_factor=None, gamma=None,
                  component_data=db, kij=None, method='PR', k_mode='gamma', emit_warnings=False)


def test_liquid_departure_h_and_cp_are_zero(water):
    p = water.props
    assert p.phase == 'liquid'
    assert p.H_dep_mix_liq == 0.0
    assert p.Cp_dep_mix_liq == 0.0


def test_liquid_enthalpy_increment_is_the_liquid_cp_integral(water):
    # at fixed P, T_dew is fixed, so H(T2) - H(T1) must be exactly the integral of the database liquid Cp
    s1, s2 = water.at(T=300.0, P=3.0), water.at(T=380.0, P=3.0)
    dH = s2.props.H_total_mix - s1.props.H_total_mix                    # kJ/kmol
    cd = water.component_data
    expected = float(Stream._int_cp_liq(cd, 300.0, 380.0).iloc[0])       # J/mol = kJ/kmol
    assert dH == pytest.approx(expected, rel=1e-9)


def test_liquid_enthalpy_is_independent_of_pressure_in_the_ideal_part(water):
    # physical dh from 1 to 5 bar is about +0.4 kJ/kg (v(1 - beta T) dP); the gamma-phi liquid has no
    # pressure term, so it must be 0 and in any case far below the 49 kJ/kg of the T_dew(P) path
    a = water.at(T=300.0, P=1.01325).props.h_total_mix
    b = water.at(T=300.0, P=5.0).props.h_total_mix
    assert abs(b - a) < 1.0


def test_liquid_entropy_is_independent_of_pressure(water):
    a = water.at(T=300.0, P=1.01325).props.s_total_mix
    b = water.at(T=300.0, P=5.0).props.s_total_mix
    assert abs(b - a) < 1e-3