import pandas as pd
import pytest


@pytest.fixture
def h2():
    return pd.DataFrame({'Component': ['Hydrogen'], 'Tc (K)': [33.1844], 'Pc (bar)': [13.1171],
                         'w': [-0.20952], 'frac_mol': [1.0]})


@pytest.fixture
def water(h2):
    return h2.assign(Component='Water', **{'Tc (K)': 647.1, 'Pc (bar)': 220.64, 'w': 0.344})


@pytest.fixture
def h2_n2_nh3():
    """Haber-Bosch feed (literature critical constants)."""
    return pd.DataFrame({'Component': ['hydrogen', 'nitrogen', 'ammonia'],
                         'Tc (K)': [33.19, 126.192, 405.56],
                         'Pc (bar)': [13.13, 33.958, 113.63],
                         'w': [-0.216, 0.0372, 0.2526],
                         'frac_mol': [0.726, 0.242, 0.032]})


@pytest.fixture
def nh3():
    return pd.DataFrame({'Component': ['ammonia'], 'Tc (K)': [405.56], 'Pc (bar)': [113.63],
                         'w': [0.2526], 'frac_mol': [1.0]})


@pytest.fixture
def feed_with_argon(h2_n2_nh3):
    """Same feed plus an argon component at z = 0 (must not change the result)."""
    ar = pd.DataFrame({'Component': ['argon'], 'Tc (K)': [150.687], 'Pc (bar)': [48.63],
                       'w': [-0.002], 'frac_mol': [0.0]})
    return pd.concat([h2_n2_nh3, ar], ignore_index=True)
