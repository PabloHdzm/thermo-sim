"""The 'srk72' form: SRK with Soave's original (1972) kappa, as used by DWSIM.

    kappa = 0.48 + 1.574 w - 0.176 w^2,   Omega_a = 0.42748,   Omega_b = 0.08664

The default 'srk' form keeps the refit of Graboski and Daubert (1978):
    kappa = 0.48508 + 1.55171 w - 0.15613 w^2.
Both share the Redlich-Kwong cubic (delta1 = 1, delta2 = 0).
"""
import numpy as np
import pandas as pd
import pytest

from thermo_sim import METHODS, ThermoMethod, build_eos
from thermo_sim.eos import FORMS

def _table(names=('ammonia',)):
    """Small component table with the columns build_eos needs (no database needed)."""
    rows = {'hydrogen': (33.19, 13.13, -0.216, 64.5), 'nitrogen': (126.192, 33.958, 0.0372, 89.4),
            'ammonia': (405.56, 113.63, 0.2526, 73.014)}
    return pd.DataFrame({'Component': list(names),
                         'Tc (K)': [rows[n][0] for n in names], 'Pc (bar)': [rows[n][1] for n in names],
                         'w': [rows[n][2] for n in names], 'Vc (cm3/mol)': [rows[n][3] for n in names],
                         'frac_mol': [1.0 / len(names)] * len(names)})


def test_srk72_constants():
    f = FORMS['srk72']
    assert (f.delta1, f.delta2, f.Omega_a, f.Omega_b) == (1.0, 0.0, 0.42748, 0.08664)


@pytest.mark.parametrize('w', [-0.2160, 0.0, 0.037, 0.256, 0.344])
def test_srk72_kappa(w):
    assert FORMS['srk72'].kappa(np.array([w]))[0] == pytest.approx(0.48 + 1.574 * w - 0.176 * w**2)


def test_srk72_differs_from_srk_only_in_kappa():
    a, b = FORMS['srk72'], FORMS['srk']
    assert (a.delta1, a.delta2, a.Omega_a, a.Omega_b) == (b.delta1, b.delta2, b.Omega_a, b.Omega_b)
    w = np.array([0.256])
    assert a.kappa(w)[0] != pytest.approx(b.kappa(w)[0], abs=1e-4)


def test_method_accepts_srk72_with_the_usual_alphas_and_volume_models():
    for alpha in ('std', 'bm'):
        for vol in ('eos', 'eos_peneloux', 'rackett'):
            assert ThermoMethod('srk72', alpha, liquid_volume=vol).eos == 'srk72'

def _table(names=('ammonia',)):
    """Small component table with the columns build_eos needs (no database needed)."""
    rows = {'hydrogen': (33.19, 13.13, -0.216, 64.5), 'nitrogen': (126.192, 33.958, 0.0372, 89.4),
            'ammonia': (405.56, 113.63, 0.2526, 73.014)}
    return pd.DataFrame({'Component': list(names),
                         'Tc (K)': [rows[n][0] for n in names], 'Pc (bar)': [rows[n][1] for n in names],
                         'w': [rows[n][2] for n in names], 'Vc (cm3/mol)': [rows[n][3] for n in names],
                         'frac_mol': [1.0 / len(names)] * len(names)})


def test_srk72_preset_and_eos_build():
    assert METHODS['SRK72'].eos == 'srk72'
    eos = build_eos(METHODS['SRK72'], _table())
    assert eos.form.name == 'srk72'
