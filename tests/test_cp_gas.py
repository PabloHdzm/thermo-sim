"""Ideal-gas Cp polynomial: evaluator (any degree) and database fit against CoolProp.

The evaluator lives in stream.py (_cp_gas_coeffs, _poly_cp, _poly_int, _poly_int_over_T).
The database (src/thermo_sim/data/properties_db.csv) is built by scripts/build_db.py from
CoolProp's ideal-gas Cp (cp0molar), so CoolProp is an independent reference here.
"""
import numpy as np
import pandas as pd
import pytest

from thermo_sim import load_database
from thermo_sim.stream import _cp_gas_coeffs, _poly_cp, _poly_int, _poly_int_over_T

R_J = 1.0  # J/mol == kJ/kmol numerically


def _table(degree, n=3, seed=0):
    rng = np.random.default_rng(seed)
    scale = [30.0, 1e-2, 1e-5, 1e-8, 1e-11, 1e-14, 1e-17, 1e-20]
    data = {f"{L}_Cp_Gas": rng.uniform(0.5, 1.5, n) * scale[k]
            for k, L in enumerate("ABCDEFGH"[:degree + 1])}
    return pd.DataFrame(data, index=[f"c{i}" for i in range(n)])


# ---- evaluator ----------------------------------------------------------------------------
def test_quartic_matches_the_original_explicit_formulas():
    """Refactor regression: the generic evaluator equals the old A..E expressions."""
    cd = _table(4)
    A, B, C, D, E = (cd[f"{L}_Cp_Gas"].to_numpy() for L in "ABCDE")
    Ta, Tb = 298.15, 253.8
    cp_old = A + B * Tb + C * Tb**2 + D * Tb**3 + E * Tb**4
    int_old = (A * (Tb - Ta) + B / 2 * (Tb**2 - Ta**2) + C / 3 * (Tb**3 - Ta**3)
               + D / 4 * (Tb**4 - Ta**4) + E / 5 * (Tb**5 - Ta**5))
    ovt_old = (A * np.log(Tb / Ta) + B * (Tb - Ta) + C / 2 * (Tb**2 - Ta**2)
               + D / 3 * (Tb**3 - Ta**3) + E / 4 * (Tb**4 - Ta**4))
    Cm = _cp_gas_coeffs(cd)
    assert np.allclose(_poly_cp(Cm, Tb), cp_old, rtol=1e-13)
    assert np.allclose(_poly_int(Cm, Ta, Tb), int_old, rtol=1e-12)
    assert np.allclose(_poly_int_over_T(Cm, Ta, Tb), ovt_old, rtol=1e-12)


@pytest.mark.parametrize('degree', [2, 4, 5, 6, 7])
def test_degree_follows_the_available_columns(degree):
    Cm = _cp_gas_coeffs(_table(degree))
    assert Cm.shape == (3, degree + 1)


@pytest.mark.parametrize('degree', [4, 6])
def test_integrals_are_consistent_with_cp(degree):
    """d/dT of the integral of Cp is Cp, and d/dT of the integral of Cp/T is Cp/T."""
    Cm = _cp_gas_coeffs(_table(degree))
    T, h, Ta = 400.0, 1e-3, 298.15
    d1 = (_poly_int(Cm, Ta, T + h) - _poly_int(Cm, Ta, T - h)) / (2 * h)
    d2 = (_poly_int_over_T(Cm, Ta, T + h) - _poly_int_over_T(Cm, Ta, T - h)) / (2 * h)
    assert np.allclose(d1, _poly_cp(Cm, T), rtol=1e-7)
    assert np.allclose(d2, _poly_cp(Cm, T) / T, rtol=1e-7)


def test_integrals_vanish_on_an_empty_interval_and_are_antisymmetric():
    Cm = _cp_gas_coeffs(_table(6))
    assert np.allclose(_poly_int(Cm, 300.0, 300.0), 0.0)
    assert np.allclose(_poly_int(Cm, 250.0, 350.0), -_poly_int(Cm, 350.0, 250.0))
    assert np.allclose(_poly_int_over_T(Cm, 250.0, 350.0), -_poly_int_over_T(Cm, 350.0, 250.0))


def test_missing_cp_columns_raise():
    with pytest.raises(KeyError):
        _cp_gas_coeffs(pd.DataFrame({'x': [1.0]}))


def test_non_contiguous_cp_columns_raise():
    cd = _table(5).drop(columns='D_Cp_Gas')
    with pytest.raises(ValueError, match='contiguous'):
        _cp_gas_coeffs(cd)


# ---- database fit vs CoolProp --------------------------------------------------------------
CP = pytest.importorskip('CoolProp.CoolProp')
cprop = pytest.importorskip('CoolProp')


@pytest.fixture(scope='module')
def db():
    return load_database().set_index('Component')


def _cp0(fluid):
    s = CP.AbstractState('HEOS', fluid)

    def f(T):
        s.update(cprop.DmolarT_INPUTS, 1e-3, T)
        return s.cp0molar()
    return f


def test_cp_range_covers_cold_service(db):
    """Components whose EOS reaches 200 K must be fitted from 200 K (no extrapolation at 248 K)."""
    for name in ('Hydrogen', 'Nitrogen', 'Ammonia', 'Methane'):
        assert db.loc[name, 'Tmin_Cp_Gas'] <= 200.0 + 1e-9, name


def test_cp_fit_error_against_coolprop(db):
    """Cp within 0.2 % over the whole fitted range, for every component."""
    for name, row in db.iterrows():
        cp0 = _cp0(row['CoolProp_fluid'])
        c = _cp_gas_coeffs(db.loc[[name]])
        T = np.linspace(row['Tmin_Cp_Gas'], row['Tmax_Cp_Gas'], 300)
        ref = np.array([cp0(t) for t in T])
        fit = np.array([_poly_cp(c, t)[0] for t in T])
        assert np.max(np.abs(fit / ref - 1)) < 2e-3, name


def test_ideal_enthalpy_change_against_coolprop(db):
    """Integral of Cp from 298.15 K: within 2 kJ/kmol for |T - 298.15| <= 100 K, else 0.1 %."""
    from scipy.integrate import quad
    for name, row in db.iterrows():
        cp0 = _cp0(row['CoolProp_fluid'])
        c = _cp_gas_coeffs(db.loc[[name]])
        for T in (220.0, 248.469, 350.0, 398.0, 600.0, 900.0):
            if not row['Tmin_Cp_Gas'] <= T <= row['Tmax_Cp_Gas']:
                continue
            ref = quad(cp0, 298.15, T)[0]
            got = _poly_int(c, 298.15, T)[0]
            tol = 2.0 if abs(T - 298.15) <= 100 else 1e-3 * abs(ref)
            assert abs(got - ref) < tol, (name, T, got, ref)


def test_no_extrapolation_warning_at_cold_flash(db):
    """The DWSIM comparison case (248.47 K) must not trigger the Cp range warning."""
    import warnings
    from thermo_sim import METHODS, Stream
    kij = np.array([[0.0, 0.1030, 0.0], [0.1030, 0.0, 0.2193], [0.0, 0.2193, 0.0]])
    with warnings.catch_warnings():
        warnings.filterwarnings('error', message='.*fit range.*')
        Stream('x', fm=None, comp_mass=None, fn=100.0,
               comp_mol={'hydrogen': 0.726, 'nitrogen': 0.242, 'ammonia': 0.032}, V_frac=None,
               T=248.469, P=150.0, Tref=298.15, Pref=1.01325, gamma=None,
               component_data=db.reset_index(), kij=kij, method=METHODS['PR'], k_mode='cubic',
               psat_factor=None)
