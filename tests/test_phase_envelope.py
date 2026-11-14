"""Phase envelope (P-T): consistency checks on mixtures of very different volatility (no DWSIM data).

Binary and multicomponent mixtures with default constants and k_ij = 0: the continuation must go through
the critical point on both branches (a trace that steps over it, or collapses before it, was the failure
mode found with asymmetric mixtures), the equilibrium residuals recomputed from ln K must vanish, and
the cricondentherm / cricondenbar must not lie below the critical point.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

from thermo_sim import METHODS, Stream, load_database
from thermo_sim.phase_envelope import PhaseEnvelope

CASES = {
    'methane/butane 50/50': {'methane': 0.5, 'butane': 0.5},
    'methane/butane 95/5': {'methane': 0.95, 'butane': 0.05},
    'methane/pentane 90/10': {'methane': 0.9, 'pentane': 0.1},
    'ethane/propane 50/50': {'ethane': 0.5, 'propane': 0.5},
    'propane/pentane 30/70': {'propane': 0.3, 'pentane': 0.7},
    'N2/CO2/CH4/C2H6/C3H8': {'nitrogen': 0.05, 'carbondioxide': 0.10, 'methane': 0.70, 'ethane': 0.10,
                             'propane': 0.05},
}


def _envelope(comp):
    db = load_database()
    have = set(db['Component'].astype(str).str.lower())
    if not set(comp) <= have:
        pytest.skip(f'missing in the database: {sorted(set(comp) - have)} (run scripts/build_db.py)')
    names = list(comp)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        s = Stream(description='mix', fm=None, comp_mass=None, fn=1e-3, comp_mol=dict(comp), V_frac=None,
                   T=300.0, P=1.01325, Tref=298.15, Pref=1.01325, psat_factor=None, gamma=None,
                   component_data=db[db['Component'].str.lower().isin(names)].copy(),
                   kij=pd.DataFrame(0.0, index=names, columns=names), k_mode='cubic',
                   method=METHODS['PR'], emit_warnings=False)
    pe = PhaseEnvelope.from_stream(s)
    return pe, pe.trace(1.01325), np.asarray(s._eos.Tc, float)


@pytest.fixture(scope='module', params=list(CASES), ids=list(CASES))
def traced(request):
    return _envelope(CASES[request.param])


def test_traced_through_the_critical_point(traced):
    _, r, Tc = traced
    assert r.status == 'ok'
    assert Tc.min() < r.critical_point[0] < Tc.max()


def test_residuals_and_phases(traced):
    _, r, _ = traced
    assert r.residual_max(0.05) < 1e-8
    assert r.sum_error_max(0.05) < 1e-10
    assert min(p.dZ for p in r.path if np.linalg.norm(p.lnK) > 0.05) > 1e-4


def test_extrema_not_below_the_critical_point(traced):
    _, r, _ = traced
    Tcr, Pcr = r.critical_point
    assert r.cricondentherm[0] >= Tcr - 1e-6
    assert r.cricondenbar[1] >= Pcr - 1e-6


def test_branches_start_at_the_1_atm_saturation_points(traced):
    pe, r, _ = traced
    assert r.T_bubble[0] == pytest.approx(pe.engine.bubble_T(1.01325, pe.z).T, abs=1e-4)
    assert r.T_dew[0] == pytest.approx(pe.engine.dew_T(1.01325, pe.z).T, abs=1e-4)
    assert r.T_bubble[0] < r.T_dew[0]
