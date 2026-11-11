"""Standard-state treatment of the ideal-gas entropy.

The reaction entropy N2/H2 -> NH3 must not depend on the reference pressure, and the
ammonia ideal entropy shifts by exactly R ln(Pref ratio).
"""

import numpy as np
import pytest
from thermo_sim import METHODS, Stream, load_database
from thermo_sim.constants import R_GAS



def _pure(name, T, P, Pref):
    return Stream(description=name, fm=None, comp_mass=None, fn=1.0, comp_mol={name: 1.0},
                  V_frac=None, T=T, P=P, Tref=298.15, Pref=Pref, gamma=None,
                  component_data=load_database(), kij=None, method=METHODS['PR'],
                  emit_warnings=False)


def _reaction_entropy(Pref, T=400.0, P=10.0):
    s = {n: _pure(n, T, P, Pref).props.S_total_mix for n in ('ammonia', 'nitrogen', 'hydrogen')}
    return s['ammonia'] - 0.5 * s['nitrogen'] - 1.5 * s['hydrogen']


def test_reaction_entropy_is_independent_of_reference_pressure():
    assert _reaction_entropy(1.01325) == pytest.approx(_reaction_entropy(1.0), abs=1e-9)


def test_pure_ammonia_ideal_entropy_shift_equals_one_R_ln_pref():
    a = _pure('ammonia', 400.0, 10.0, 1.01325).props.S_ig_indiv_gas[0]
    b = _pure('ammonia', 400.0, 10.0, 1.0).props.S_ig_indiv_gas[0]
    assert a - b == pytest.approx(R_GAS * np.log(1.01325), rel=1e-4)


def test_elements_have_no_shift():
    for name in ('hydrogen', 'nitrogen'):
        a = _pure(name, 400.0, 10.0, 1.01325).props.S_ig_indiv_gas[0]
        b = _pure(name, 400.0, 10.0, 1.0).props.S_ig_indiv_gas[0]
        assert a == pytest.approx(b, abs=1e-9)
