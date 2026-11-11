"""The gas constant R is a ThermoMethod parameter (default: thermo_sim.constants.R_GAS).

Dimensionless A and B do not depend on R, so phase equilibrium (T, P, x, y, K, phi) is
independent of R. Only quantities with units of volume, density, entropy and Rackett/Peneloux
volumes move with R.
"""

import numpy as np
import pytest

from thermo_sim import METHODS, Stream, ThermoMethod, build_eos, load_database
from thermo_sim.constants import R_GAS
import pandas as pd

KIJ_PR = np.array([[0.0, 0.1030, 0.0],
                   [0.1030, 0.0, 0.2193],
                   [0.0, 0.2193, 0.0]])          # H2, N2, NH3 (ordered by Tc)
FEED = {'hydrogen': 0.726, 'nitrogen': 0.242, 'ammonia': 0.032}

R_LO, R_HI = 8.1, 8.5


def _table(names=('hydrogen', 'nitrogen', 'ammonia')):
    """Small component table with the columns build_eos needs (no database needed)."""
    rows = {'hydrogen': (33.19, 13.13, -0.216, 64.5), 'nitrogen': (126.192, 33.958, 0.0372, 89.4),
            'ammonia': (405.56, 113.63, 0.2526, 73.014)}
    return pd.DataFrame({'Component': list(names),
                         'Tc (K)': [rows[n][0] for n in names], 'Pc (bar)': [rows[n][1] for n in names],
                         'w': [rows[n][2] for n in names], 'Vc (cm3/mol)': [rows[n][3] for n in names],
                         'frac_mol': [1.0 / len(names)] * len(names)})


@pytest.fixture(scope='module')
def db():
    return load_database()


def _stream(db, method, comp_mol, **over):
    kw = dict(description='x', fm=None, comp_mass=None, fn=100.0, comp_mol=comp_mol,
              V_frac=None, T=300.0, P=1.0, Tref=298.15, Pref=1.01325, psat_factor=None,
              gamma=None, component_data=db, kij=None, k_mode='cubic', method=method,
              emit_warnings=False)
    kw.update(over)
    return Stream(**kw)


# ----------------------------------------------------------------------------------------
# The parameter itself
# ----------------------------------------------------------------------------------------
def test_default_r_is_the_project_constant():
    assert ThermoMethod('pr', 'std').R == R_GAS
    assert METHODS['PR'].R == R_GAS


def test_but_sets_r_on_the_eos():
    eos = build_eos(METHODS['PR'].but(R=8.314), _table(('ammonia',)))
    assert eos.R == 8.314


def test_default_eos_uses_the_class_default():
    assert build_eos('PR', _table(('ammonia',))).R == R_GAS


def test_non_default_r_is_reported_in_the_notes():
    eos = build_eos(METHODS['PR'].but(R=8.314), _table(('ammonia',)))
    assert any('R = 8.314' in n for n in eos.notes)


@pytest.mark.parametrize('bad', [0.0, -8.314, 83.14, 1.987])
def test_invalid_r_is_rejected(bad):
    with pytest.raises(ValueError):
        ThermoMethod('pr', 'std', R=bad)


# ----------------------------------------------------------------------------------------
# What depends on R and what does not
# ----------------------------------------------------------------------------------------
def test_dimensionless_eos_parameters_do_not_depend_on_r():
    d = _table()
    out = []
    for R in (R_LO, R_HI):
        e = build_eos(METHODS['PR'].but(R=R), d)
        x = np.full(len(d), 1.0 / len(d))
        am, bm = e.mixture_params(400.0, x)
        out.append((am * 150e5 / (e.R * 400.0) ** 2, bm * 150e5 / (e.R * 400.0)))
    assert out[0] == pytest.approx(out[1], rel=1e-12)


def test_flash_temperature_does_not_depend_on_r(db):
    kw = dict(V_frac=0.99, T=300.0, P=150.0, kij=KIJ_PR)
    t = [_stream(db, METHODS['PR'].but(R=R), FEED, **kw).T for R in (R_LO, R_HI)]
    assert t[0] == pytest.approx(t[1], abs=1e-8)


def test_gas_density_scales_as_one_over_r(db):
    """Ideal-gas limit: rho = P/(Z R T), so rho ~ 1/R."""
    rho = [_stream(db, METHODS['PR'].but(R=R), {'nitrogen': 1.0}, P=1.0).props.dens_mix_gas
           for R in (R_LO, R_HI)]
    assert rho[0] / rho[1] == pytest.approx(R_HI / R_LO, rel=1e-3)


def test_entropy_pressure_term_uses_the_method_r(db):
    """S(2 bar) - S(1 bar) of a low-pressure gas is -n R ln 2: the ratio between two R is R_HI/R_LO."""
    def s(R, P):
        return _stream(db, METHODS['PR'].but(R=R), {'nitrogen': 1.0}, P=P).props.S_total_mix
    d_lo = s(R_LO, 2.0) - s(R_LO, 1.0)
    d_hi = s(R_HI, 2.0) - s(R_HI, 1.0)
    assert d_lo < 0 and d_hi < 0
    assert d_hi / d_lo == pytest.approx(R_HI / R_LO, rel=2e-2)


def test_default_r_is_codata():
    assert R_GAS == pytest.approx(8.314462618, rel=1e-12)


def test_general_presets_use_the_default_r():
    for name in ('PR', 'STD-SRK', 'RK'):
        assert METHODS[name].R == R_GAS