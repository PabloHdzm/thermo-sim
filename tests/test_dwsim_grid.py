"""thermo-sim against DWSIM 10.2.12.0 sensitivity grids (H2/N2/NH3 feed, DWSIM constants).

pr : Peng-Robinson, 175 points (32 two-phase)           reference_data/dwsim/ammonia_feed_grid.csv
srk: Soave 1972 SRK ('srk72'), kij = 0, 163 points (32 two-phase)  ..._srk.csv

The equation of state, constants, kij and R are DWSIM's, so the comparison isolates the
implementation. Tolerances come from the observed differences (PR: first run of
scripts/compare_dwsim_grid.py); the SRK ones are provisional until its first run.

Energy: X(T,P) - X(T,1 bar) is compared, which removes the ideal-gas Cp, the formation enthalpy
and the (different) reference states of the two programs. DWSIM's two-phase entropy is
inconsistent with its own enthalpy (see reference_data/dwsim/README.md), so it is xfail.
"""
import pytest

from _dwsim_grid import CASES, GRIDS, energy_diffs, energy_rows, equilibrium_diffs, row_id

TOL_V = 5e-6            # vapor fraction, absolute (PR observed 1e-6)
TOL_Z = 1e-5            # Z vapor, absolute (PR observed 5e-6)
TOL_PHI = 1e-5          # fugacity coefficients, relative, 6 printed digits (PR observed 5e-6)
TOL_PHI_3DIG = 6e-3     # PR rows exported with 3 digits (refined block): resolution is 5e-3
TOL_REL_ZL = 1e-5       # PR observed 3e-6
TOL_RHO = 2e-5          # mixture density, relative (observed 7e-6 PR, 6e-6 SRK)
TOL_DH = 3e-5           # |dH| / (R T) (PR observed 1.3e-5)
TOL_DS_1PH = 3e-5       # |dS| / R, single-phase rows (observed 8.5e-6 SRK, 6.4e-6 PR)

ALL = [(c, r) for c, g in GRIDS.items() for r in g.itertuples()]
TWO = [(c, r) for c, r in ALL if r.n_phases == 2]
ONE_RHO = [(c, r) for c, r in ALL if r.n_phases == 1 or CASES[c]['density_two_phase']]
EALL = [(c, r) for c in GRIDS for r in energy_rows(c).itertuples()]
E_ONE = [(c, r) for c, r in EALL if r.n_phases == 1]
E_TWO = [(c, r) for c, r in EALL if r.n_phases == 2]


def _ids(rows):
    return [row_id(r, c) for c, r in rows]


_cache = {}


def _eq(case, r):
    k = (case, r.T_K, r.P_bar)
    if k not in _cache:
        _cache[k] = equilibrium_diffs(GRIDS[case].loc[r.Index], case)
    return _cache[k]


def _tol_phi(r):
    return TOL_PHI if r.phi_digits >= 6 else TOL_PHI_3DIG


@pytest.mark.parametrize('case,r', ALL, ids=_ids(ALL))
def test_phase_split_and_vapor_compressibility(case, r):
    d = _eq(case, r)
    assert d['dV'] == pytest.approx(0.0, abs=TOL_V)
    assert d['dZv'] == pytest.approx(0.0, abs=TOL_Z)


@pytest.mark.parametrize('case,r', ALL, ids=_ids(ALL))
def test_vapor_fugacity_coefficients(case, r):
    d = _eq(case, r)
    for n in ('H2', 'N2', 'NH3'):
        assert d['rel_phi_v_' + n] == pytest.approx(0.0, abs=_tol_phi(r)), n


@pytest.mark.parametrize('case,r', TWO, ids=_ids(TWO))
def test_liquid_phase_two_phase_rows(case, r):
    d = _eq(case, r)
    assert d['rel_Zl'] == pytest.approx(0.0, abs=TOL_REL_ZL)
    for n in ('H2', 'N2', 'NH3'):
        assert d['rel_phi_l_' + n] == pytest.approx(0.0, abs=_tol_phi(r)), n


@pytest.mark.parametrize('case,r', ONE_RHO, ids=_ids(ONE_RHO))
def test_mixture_density(case, r):
    """Single-phase rows always; two-phase rows only where DWSIM used the EOS liquid volume."""
    assert _eq(case, r)['rel_rho'] == pytest.approx(0.0, abs=TOL_RHO)


@pytest.mark.parametrize('case,r', EALL, ids=_ids(EALL))
def test_isothermal_enthalpy_differences(case, r):
    d = energy_diffs(GRIDS[case].loc[r.Index], case)
    assert d['dH_RT'] == pytest.approx(0.0, abs=TOL_DH)


@pytest.mark.parametrize('case,r', E_ONE, ids=_ids(E_ONE))
def test_isothermal_entropy_differences_single_phase(case, r):
    d = energy_diffs(GRIDS[case].loc[r.Index], case)
    assert d['dS_R'] == pytest.approx(0.0, abs=TOL_DS_1PH)


@pytest.mark.xfail(strict=False, reason=(
    "DWSIM's two-phase mixture entropy is inconsistent with its own enthalpy (dH != T dS at "
    "constant P by ~1.8 R per mole of liquid; see reference_data/dwsim/README.md), so a "
    "mismatch here is a DWSIM artifact."))
@pytest.mark.parametrize('case,r', E_TWO, ids=_ids(E_TWO))
def test_isothermal_entropy_differences_two_phase(case, r):
    d = energy_diffs(GRIDS[case].loc[r.Index], case)
    assert d['dS_R'] == pytest.approx(0.0, abs=TOL_DS_1PH)