"""Ideal-gas Cp outside the fitted interval: linear continuation of the polynomial.

Inside [Tmin_Cp_Gas, Tmax_Cp_Gas] the degree-6 polynomial is used as is. Outside, Cp is a
straight line with the polynomial's value and slope at the nearest limit, because the
polynomial itself diverges quickly (e.g. NH3 +79 % at 1500 K). H and S are the exact
integrals of that Cp, so they are continuous across the limits.
"""
import warnings

import numpy as np
import pytest

from thermo_sim import load_database
from thermo_sim.stream import (_cp_ext, _cp_gas_coeffs, _cp_gas_limits, _H_ext, _int_ext,
                               _int_over_T_ext, _poly_cp, _poly_int, _poly_int_over_T, _S_ext)

COMPONENTS = ('Hydrogen', 'Nitrogen', 'Ammonia', 'Methane')


@pytest.fixture(scope='module')
def db():
    return load_database().set_index('Component')


@pytest.fixture(scope='module')
def cd(db):
    return db.loc[list(COMPONENTS)].reset_index()


@pytest.fixture(scope='module')
def parts(cd):
    return _cp_gas_coeffs(cd), *_cp_gas_limits(cd)


# ---- inside the interval: identical to the polynomial -----------------------------------------
@pytest.mark.parametrize('T', [200.0, 248.469, 500.0, 999.0])
def test_inside_the_interval_equals_the_polynomial(parts, T):
    C, lo, hi = parts
    assert np.allclose(_cp_ext(C, lo, hi, T), _poly_cp(C, T), rtol=1e-12)
    assert np.allclose(_int_ext(C, lo, hi, 298.15, T), _poly_int(C, 298.15, T), rtol=1e-9, atol=1e-6)
    assert np.allclose(_int_over_T_ext(C, lo, hi, 298.15, T), _poly_int_over_T(C, 298.15, T),
                       rtol=1e-9, atol=1e-9)


# ---- continuity -------------------------------------------------------------------------------
@pytest.mark.parametrize('side', ['lo', 'hi'])
def test_cp_h_and_s_are_continuous_at_the_limits(parts, side):
    C, lo, hi = parts
    T0 = lo if side == 'lo' else hi
    eps = 1e-6
    for f in (_cp_ext, _H_ext, _S_ext):
        assert np.allclose(f(C, lo, hi, T0 - eps), f(C, lo, hi, T0 + eps), rtol=1e-6, atol=1e-4)


@pytest.mark.parametrize('side', ['lo', 'hi'])
def test_slope_of_cp_is_continuous_at_the_limits(parts, side):
    C, lo, hi = parts
    T0, h = (lo if side == 'lo' else hi), 1e-3
    left = (_cp_ext(C, lo, hi, T0 - h) - _cp_ext(C, lo, hi, T0 - 2 * h)) / h
    right = (_cp_ext(C, lo, hi, T0 + 2 * h) - _cp_ext(C, lo, hi, T0 + h)) / h
    assert np.allclose(left, right, atol=1e-4)


# ---- consistency of the integrals -------------------------------------------------------------
@pytest.mark.parametrize('T', [120.0, 199.5, 201.0, 500.0, 999.0, 1001.0, 1500.0, 2500.0])
def test_derivatives_of_h_and_s_give_cp(parts, T):
    C, lo, hi = parts
    h = 1e-3
    dH = (_H_ext(C, lo, hi, T + h) - _H_ext(C, lo, hi, T - h)) / (2 * h)
    dS = (_S_ext(C, lo, hi, T + h) - _S_ext(C, lo, hi, T - h)) / (2 * h)
    cp = _cp_ext(C, lo, hi, T)
    assert np.allclose(dH, cp, rtol=1e-6)
    assert np.allclose(dS, cp / T, rtol=1e-6)


def test_integrals_are_additive_across_the_limits(parts):
    C, lo, hi = parts
    a, b, c = 150.0, 400.0, 1300.0
    for f in (_int_ext, _int_over_T_ext):
        assert np.allclose(f(C, lo, hi, a, c), f(C, lo, hi, a, b) + f(C, lo, hi, b, c), rtol=1e-9, atol=1e-9)


def test_cp_is_positive_and_finite_everywhere(parts):
    C, lo, hi = parts
    for T in np.linspace(50.0, 3000.0, 60):
        cp = _cp_ext(C, lo, hi, T)
        assert np.all(np.isfinite(cp)) and np.all(cp > 0)


# ---- backwards compatibility ------------------------------------------------------------------
def test_without_limit_columns_the_polynomial_is_used_everywhere(cd):
    bare = cd.drop(columns=['Tmin_Cp_Gas', 'Tmax_Cp_Gas'])
    C, (lo, hi) = _cp_gas_coeffs(bare), _cp_gas_limits(bare)
    assert np.all(np.isinf(lo)) and np.all(np.isinf(hi))
    assert np.allclose(_cp_ext(C, lo, hi, 3000.0), _poly_cp(C, 3000.0))


def test_nan_limits_mean_no_limit(cd):
    d = cd.copy()
    d['Tmax_Cp_Gas'] = np.nan
    lo, hi = _cp_gas_limits(d)
    assert np.all(np.isinf(hi)) and np.all(np.isfinite(lo))


# ---- accuracy outside the interval, against CoolProp ------------------------------------------
CP = pytest.importorskip('CoolProp.CoolProp')
cprop = pytest.importorskip('CoolProp')


def _cp0(fluid):
    s = CP.AbstractState('HEOS', fluid)

    def f(T):
        s.update(cprop.DmolarT_INPUTS, 1e-3, T)
        return s.cp0molar()
    return f


@pytest.mark.parametrize('T,tol', [(150.0, 0.03), (1200.0, 0.03), (1500.0, 0.06)])
def test_extrapolated_cp_against_coolprop(db, cd, parts, T, tol):
    """Linear continuation stays within a few percent (the raw polynomial gives -142 %
    for H2 and +79 % for NH3 at 1500 K)."""
    C, lo, hi = parts
    cp = _cp_ext(C, lo, hi, T)
    for i, name in enumerate(COMPONENTS):
        ref = _cp0(db.loc[name, 'CoolProp_fluid'])(T)
        assert abs(cp[i] / ref - 1) < tol, (name, T, cp[i], ref)


def test_the_raw_polynomial_is_worse_than_the_continuation_at_1500_k(db, cd, parts):
    C, lo, hi = parts
    ext, raw = _cp_ext(C, lo, hi, 1500.0), _poly_cp(C, 1500.0)
    for i, name in enumerate(COMPONENTS):
        ref = _cp0(db.loc[name, 'CoolProp_fluid'])(1500.0)
        assert abs(ext[i] / ref - 1) < abs(raw[i] / ref - 1), name


# ---- Stream level -----------------------------------------------------------------------------
def test_stream_outside_the_interval_warns_and_gives_finite_properties(db):
    from thermo_sim import METHODS, Stream

    def make(T, emit):
        return Stream('h2', fm=None, comp_mass=None, fn=1.0, comp_mol={'hydrogen': 1.0},
                      V_frac=None, T=T, P=1.0, Tref=298.15, Pref=1.01325, gamma=None,
                      component_data=db.reset_index(), kij=None, method=METHODS['PR'],
                      k_mode='cubic', psat_factor=None, emit_warnings=emit)

    with pytest.warns(RuntimeWarning, match='ideal-gas Cp'):
        make(1500.0, True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        h1000, h1500 = (make(T, False).props.H_total_mix for T in (1000.0, 1500.0))
    from scipy.integrate import quad
    ref = quad(_cp0('Hydrogen'), 1000.0, 1500.0)[0]
    assert np.isfinite(h1500) and h1500 > h1000
    assert (h1500 - h1000) == pytest.approx(ref, rel=0.02)
