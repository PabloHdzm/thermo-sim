"""Ammonia synthesis loop (compressor -> heater -> reactor -> cooler -> flash) against DWSIM.

Stream state, composition, Z and density are compared stream by stream; duties and compressor
power as relative differences. The reactor is a fixed-conversion unit at the DWSIM equilibrium
conversion (48.8008 % of N2), so the reactor test checks the heat of reaction (H includes the
formation enthalpies), not the chemical equilibrium. Entropy is not compared: DWSIM's two-phase
entropy is inconsistent with its enthalpy (see reference_data/dwsim/README.md).

The ideal-gas Cp of H2, N2 and NH3 is DWSIM's own equation (fitted with a quartic, see _dwsim_loop.py);
with the packaged CoolProp-based Cp the duties differ from DWSIM by about 1 %.
Tolerances are 2 to 5 times the observed differences, which are limited by the digits DWSIM prints.
"""
import pytest

from _dwsim_loop import (STREAMS, UNITS, cp_ideal_diffs, loop, loop_packaged, stream_diffs,
                         unit_diffs)

FLASH_STREAMS = {'cold', 'purge', 'nh3_liquid'}


def _tol(name, key):
    if key == 'dT':
        return 0.05 if name == 'comp_out' else 1e-6          # DWSIM prints 588.76 K; observed 0.019 K
    if key == 'dP':
        return 1e-9
    if key == 'abs_fn':
        return 5e-9                                          # kmol/s, 6 printed digits
    if key == 'dV':
        return 2e-5
    if key.startswith('rel_n_'):
        return 1e-4 if name in FLASH_STREAMS else 1e-5
    if key in ('rel_Zv', 'rel_Zl', 'rel_rho'):
        return 1e-4 if name == 'comp_out' else 2e-5      # comp_out: 0.019 K in T moves rho by 3e-5
    if key.startswith('rel_y_'):
        return 1e-4
    if key.startswith('rel_x_'):
        return 2e-4                                          # x(H2) is 0.0078
    raise KeyError(key)


CASES = [(n, k) for n in STREAMS.index for k in stream_diffs(n)]


@pytest.mark.parametrize('name,key', CASES, ids=[f'{n}-{k}' for n, k in CASES])
def test_stream_vs_dwsim(name, key):
    assert stream_diffs(name)[key] == pytest.approx(0.0, abs=_tol(name, key))


UNIT_TOL = {                                                 # relative to DWSIM; observed 6e-6 to 2.4e-4 (heater)
    'compressor power [kW]': ('compressor', 'power_kW', 1e-3),
    'heater duty [kW]': ('heater', 'duty_kW', 1e-3),
    'reactor duty [kW]': ('reactor', 'duty_kW', 1e-3),
    'cooler duty [kW]': ('cooler', 'duty_kW', 1e-3),
}


@pytest.mark.parametrize('label', list(UNIT_TOL))
def test_unit_duty_vs_dwsim(label):
    unit, key, rel = UNIT_TOL[label]
    assert unit_diffs()[label] == pytest.approx(0.0, abs=rel * abs(UNITS[unit][key]))


def test_compressor_outlet_temperature():
    assert unit_diffs()['compressor T_out [K]'] == pytest.approx(0.0, abs=0.05)


@pytest.mark.parametrize('name', list(cp_ideal_diffs()))
def test_ideal_gas_cp_matches_dwsim(name):
    """The Cp polynomial used in the loop reproduces DWSIM's mixture Cp (observed 2e-4 at worst)."""
    assert cp_ideal_diffs()[name][2] == pytest.approx(0.0, abs=5e-4)


def test_cooler_vapor_fraction():
    assert unit_diffs()['cooler vapor fraction'] == pytest.approx(0.0, abs=1e-6)


def test_mass_and_energy_balance_of_the_loop():
    L = loop()
    assert L['drum'].mass_error < 1e-10
    assert L['drum'].energy_error_kW == pytest.approx(0.0, abs=1e-6)
    # The reactor conserves atoms, not tabulated mass: 3 MM(H2) + MM(N2) differs from
    # 2 MM(NH3) by about 1e-6 (relative) in the database.
    assert L['rx'].outlet.fm == pytest.approx(L['heat'].outlet.fm, rel=1e-5)


# Packaged database (ideal-gas Cp fitted to CoolProp): the Cp differs from DWSIM's equation by
# 0.3 to 0.5 %. Observed: duties 0.15 to 0.9 %, compressor T_out +0.56 K. These tests record that
# known, understood difference; the EOS itself is validated by the tests above.
PACKAGED_TOL = {
    'compressor power [kW]': 0.015,
    'heater duty [kW]': 0.015,
    'reactor duty [kW]': 0.015,
    'cooler duty [kW]': 0.015,
}


@pytest.mark.parametrize('label', list(PACKAGED_TOL))
def test_packaged_database_duties_within_cp_difference(label):
    unit, key, _ = UNIT_TOL[label]
    ref = UNITS[unit][key]
    assert unit_diffs(loop_packaged())[label] / ref == pytest.approx(0.0, abs=PACKAGED_TOL[label])


def test_packaged_database_compressor_outlet_temperature():
    assert unit_diffs(loop_packaged())['compressor T_out [K]'] == pytest.approx(0.0, abs=1.0)


def test_packaged_database_does_not_change_equilibrium():
    """Phase equilibrium does not depend on the ideal-gas Cp, only the temperatures it leads to."""
    d = unit_diffs(loop_packaged())
    assert d['cooler vapor fraction'] == pytest.approx(0.0, abs=1e-6)