"""Entropy and enthalpy at 298.15 K and 1.01325 bar against DWSIM 10.2.12.0 (Peng-Robinson), formation terms excluded.

Pure components check that there is no reference-state offset per component. Mixtures check the empirical mixing
rule of tests/dwsim_conventions.py (HYPOTHESIS fitted to DWSIM output, not verified in its source): thermo-sim minus
the rule offset must reproduce the DWSIM entropy. See docs/validation_dwsim_entropy_mixing.md.

Critical constants and kij of DWSIM come from natgas_phase_envelope.json.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from thermo_sim import METHODS, Stream, load_database, methods

try:                                                          # tests/ may or may not be a package
    from dwsim_conventions import entropy_offset_empirical
except ImportError:
    from tests.dwsim_conventions import entropy_offset_empirical

DATA = Path(__file__).resolve().parent.parent / 'reference_data' / 'dwsim'
REF = json.loads((DATA / 'mixing_entropy_298K.json').read_text())
ENV = json.loads((DATA / 'natgas_phase_envelope.json').read_text())
NAMES = list(ENV['composition'])
MOLAR_MASS = ENV['molar_mass']
T, P = REF['conditions']['T_K'], REF['conditions']['P_bar']
SWEEP = [(float(x), s) for x, s in REF['binary_sweep_methane_ethane']['x_methane_vs_S_dwsim'].items()]


@pytest.fixture(scope='module')
def dwsim_method(tmp_path_factory):
    rows = [dict(table='dwsim_natgas', Component=n, Tc=c['Tc'], Pc=c['Pc'], omega=c['omega'], Vc=c['Vc'],
                 Zra=c['Zra'], fitted_with='', source='DWSIM 10.2.12.0') for n, c in ENV['constants'].items()]
    path = tmp_path_factory.mktemp('ov') / 'overrides.csv'
    pd.DataFrame(rows).to_csv(path, index=False)
    old = methods.OVERRIDES_CSV_PATH
    methods.OVERRIDES_CSV_PATH = path
    methods.reload_overrides()
    yield METHODS['PR'].but(constants='dwsim_natgas')
    methods.OVERRIDES_CSV_PATH = old
    methods.reload_overrides()


@pytest.fixture(scope='module')
def database():
    """Database without formation terms: S and H relative to the ideal gas at Tref, Pref."""
    db = load_database().copy()
    db['hf_298_k'] = 0.0
    db['gf_298_k'] = 0.0
    db['n_gas_el'] = 1.0
    return db


@pytest.fixture(scope='module')
def kij():
    k = pd.DataFrame(np.zeros((5, 5)), index=NAMES, columns=NAMES)
    for pair, v in ENV['kij'].items():
        a, b = pair.split('-')
        k.loc[a, b] = k.loc[b, a] = v
    return k


@pytest.fixture(scope='module')
def props(dwsim_method, database, kij):
    def make(z):
        zz = {n: z.get(n, 0.0) for n in NAMES}
        s = Stream('mix', fm=None, comp_mass=None, fn=1.0, comp_mol=zz, V_frac=None, T=T, P=P, Tref=298.15,
                   Pref=1.01325, psat_factor=None, gamma=None, component_data=database, kij=kij,
                   method=dwsim_method, emit_warnings=False)
        return s.props
    return make


# ------------------------------------------------------------------ pure components

@pytest.mark.parametrize('name', ['methane', 'ethane', 'propane', 'butane'])
def test_pure_gas_entropy_and_enthalpy(props, name):
    p, ref = props({name: 1.0}), REF['pure'][name]
    assert p.S_total_mix == pytest.approx(ref['S'], abs=1e-4)
    assert p.H_total_mix == pytest.approx(ref['H'], abs=0.05)


def test_pure_pentane_liquid(props):
    """Liquid at this state: S and H depend on the liquid Cp and the enthalpy of vaporization (0.006 % in H)."""
    p, ref = props({'pentane': 1.0}), REF['pure']['pentane']
    assert p.S_total_mix == pytest.approx(ref['S'], abs=0.01)
    assert p.H_total_mix == pytest.approx(ref['H'], abs=2.5)


# ------------------------------------------------------------------ mixtures with the empirical rule

def _check(props, z, S_dwsim, tol):
    S = props(z).S_total_mix
    offset = entropy_offset_empirical({n: v for n, v in z.items() if v > 0}, MOLAR_MASS)
    assert S - offset == pytest.approx(S_dwsim, abs=tol)
    if abs(offset) > 0.02:                                  # without the rule the comparison fails
        assert abs(S - S_dwsim) > 0.5 * abs(offset)


@pytest.mark.parametrize('x1,S_dwsim', SWEEP, ids=[f'C1={x:.1f}' for x, _ in SWEEP])
def test_methane_ethane_sweep(props, x1, S_dwsim):
    """Maximum residual observed 3.1e-4 kJ/(kmol K). The rule was chosen with this sweep (it includes the 50/50 and
    90/10 cases), so it is a fit, not a validation; the cases below are."""
    _check(props, {'methane': x1, 'ethane': 1.0 - x1}, S_dwsim, 1e-3)


@pytest.mark.parametrize('case', ['five_components', 'methane_ethane_99_1'])
def test_mixtures_outside_the_sweep(props, case):
    """Cases not used to choose the rule (raw DWSIM export)."""
    c = REF['mixtures'][case]
    _check(props, c['composition'], c['S_dwsim'], 1e-3)


@pytest.mark.parametrize('case', ['five_components', 'methane_ethane_50_50', 'methane_ethane_90_10',
                                  'methane_ethane_99_1'])
def test_mixture_enthalpy(props, case):
    """Enthalpy of the mixtures agrees with DWSIM (observed difference below 0.002 kJ/kmol)."""
    c = REF['mixtures'][case]
    assert props(c['composition']).H_total_mix == pytest.approx(c['H_dwsim'], abs=0.01)