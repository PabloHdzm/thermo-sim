"""Pure N2 and H2 against DWSIM 10.2.12.0 (PRSV2 package), with DWSIM constants and R = 8.314.

N2 uses PRSV2 (kappa1-3 of DWSIM); H2 has kappa1-3 = 0, for which DWSIM applies the plain
Peng-Robinson alpha (kappa = 0.37464 + 1.5422 w - 0.26992 w^2), reproduced by alpha_overrides.
Data: reference_data/dwsim/ammonia_pure_prsv2.csv.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

from thermo_sim import METHODS, Stream, load_database
from _dwsim_grid import INP, NAMES, PRSV2_PARAMS, R, REF_DIR

DATA = pd.read_csv(REF_DIR / 'ammonia_pure_prsv2.csv')
VAP = DATA[DATA.phase == 'vapor']
LIQ = DATA[DATA.phase == 'liquid']
TOL = {'nitrogen': 5e-6, 'hydrogen': 1e-5}      # observed 1.7e-6 (N2) and 4.7e-6 (H2)


def _pure(comp, T, P):
    kw = dict(constants_user=INP['constants'], R=R, liquid_volume='eos',
              alpha='prsv2', alpha_overrides={'hydrogen': 'std'}, alpha_params=PRSV2_PARAMS)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message=r'Zra \(user\)')
        s = Stream(description='pure', fm=None, comp_mass=None, fn=100.0, comp_mol={comp: 1.0},
                   V_frac=None, T=float(T), P=float(P), Tref=INP['Tref_K'], Pref=1.01325,
                   psat_factor=None, gamma=None, component_data=load_database(),
                   kij=np.zeros((1, 1)), k_mode='cubic', method=METHODS['PR'].but(**kw),
                   emit_warnings=False)
    return s.props


@pytest.mark.parametrize('r', list(VAP.itertuples()),
                         ids=[f'{r.component}_T{r.T_K:g}_P{r.P_bar:g}' for r in VAP.itertuples()])
def test_pure_vapor_vs_dwsim(r):
    p = _pure(r.component, r.T_K, r.P_bar)
    tol = TOL[r.component]
    assert p.Z_gas == pytest.approx(r.Z, rel=tol)
    assert float(np.ravel(p.phi_v)[0]) == pytest.approx(r.phi, rel=tol)
