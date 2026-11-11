"""Thermodynamic self-consistency of thermo-sim (no external reference).

At constant P and in phase equilibrium, dH = T dS, so

    S(T2) - S(T1) = integral dH / T = H2/T2 - H1/T1 + integral H / T^2 dT      (by parts)

The right-hand side is evaluated with Simpson's rule on a fine grid of TP flashes of the
H2/N2/NH3 feed (DWSIM constants, same setup as test_dwsim_grid.py). Two-phase and single-phase
ranges are tested separately. This exposes any entropy term that is inconsistent with the
enthalpy, for instance the per-phase mixing term or the liquid ideal-gas reference. DWSIM fails
this test in its two-phase region (about 1.8 R per mole of liquid), see reference_data/dwsim/README.md.

Tolerance: 1e-6 R (observed 7e-8).
"""
import numpy as np
import pytest

from _dwsim_grid import GRID, R, stream

TOL = 1e-6          # |dS_flash - dS_integral| / R (observed max 7e-8)
N = 16              # Simpson intervals (even)


def _dew(P):
    return float(GRID.loc[GRID.P_bar == P, 'T_dew_K'].iloc[0])


def consistency_error(P, T1, T2, n=N):
    """(S2 - S1 from the flashes) - (integral dH/T), in units of R."""
    T = np.linspace(T1, T2, n + 1)
    pr = [stream(t, P).props for t in T]
    H = np.array([p.H_total_mix for p in pr])
    S = np.array([p.S_total_mix for p in pr])
    h = (T2 - T1) / n
    f = H / T**2
    simpson = h / 3.0 * (f[0] + f[-1] + 4.0 * f[1:-1:2].sum() + 2.0 * f[2:-1:2].sum())
    dS_int = H[-1] / T[-1] - H[0] / T[0] + simpson
    return ((S[-1] - S[0]) - dS_int) / R


@pytest.mark.parametrize('P', [50, 100, 150, 200])
def test_entropy_consistent_with_enthalpy_in_the_two_phase_region(P):
    T1, T2 = 200.0, _dew(P) - 4.0                 # stay below the dew point (kink in H)
    assert consistency_error(P, T1, T2) == pytest.approx(0.0, abs=TOL)


@pytest.mark.parametrize('P', [1, 50, 100, 200])
def test_entropy_consistent_with_enthalpy_in_the_single_phase_region(P):
    T1 = max(_dew(P) + 10.0, 300.0) if P > 1 else 300.0
    assert consistency_error(P, T1, T1 + 100.0) == pytest.approx(0.0, abs=TOL)
