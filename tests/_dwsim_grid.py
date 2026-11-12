"""Shared helpers: thermo-sim vs the DWSIM sensitivity grids (H2/N2/NH3 feed, DWSIM constants).

Cases
-----
pr  : Peng-Robinson, DWSIM kij, liquid volume 'poly'         -> ammonia_feed_grid.csv
srk : Soave 1972 SRK ('srk72'), kij = 0, liquid volume 'eos' -> ammonia_feed_grid_srk.csv
prsv2: PRSV2 (H2 with the plain PR alpha), kij = 0, 'eos'   -> ammonia_feed_grid_prsv2.csv

Used by tests/test_dwsim_grid.py, tests/test_isobaric_consistency.py and
scripts/compare_dwsim_grid.py. Reference: reference_data/dwsim/README.md.
"""
import json
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from thermo_sim import METHODS, Stream, load_database

ROOT = Path(__file__).resolve().parents[1]
REF_DIR = ROOT / 'reference_data' / 'dwsim'
META = json.loads((REF_DIR / 'ammonia_loop_feed_150bar.json').read_text(encoding='utf-8'))
INP = META['inputs']
NAMES = META['components']                       # hydrogen, nitrogen, ammonia
R = META['meta']['R_J_per_mol_K']                # 8.314, as DWSIM
SPECIES = ['H2', 'N2', 'NH3']

# PRSV2 parameters shown by DWSIM for N2 and NH3. DWSIM displays kappa1(N2) = 0.02; the value that
# reproduces its numbers is 0.01996 (Stryjek and Vera). H2 has kappa1-3 = 0 and behaves in DWSIM
# like the plain Peng-Robinson alpha, hence the 'std' override.
PRSV2_PARAMS = {'kappa1': {'nitrogen': 0.01996, 'ammonia': 0.001},
                'kappa2': {'nitrogen': 0.3162, 'ammonia': -0.1265},
                'kappa3': {'nitrogen': 0.5350, 'ammonia': 0.51}}

# tol_scale multiplies the tolerances of test_dwsim_grid.py (PRSV2: kappa values given with 3 to 4
# digits, and the N2 alpha is strongly extrapolated above Tr = 1).
CASES = {
    'pr':  dict(csv='ammonia_feed_grid.csv', base='PR', eos=None, liquid_volume='eos',
                kij=np.array(INP['kij']), density_two_phase=True, tol_scale=1.0),
    'srk': dict(csv='ammonia_feed_grid_srk.csv', base='STD-SRK', eos='srk72',
                liquid_volume='eos', kij=np.zeros((3, 3)), density_two_phase=True,
                tol_scale=1.0),
    'prsv2': dict(csv='ammonia_feed_grid_prsv2.csv', base='PR', eos=None, liquid_volume='eos',
                  kij=np.zeros((3, 3)), density_two_phase=True, tol_scale=30.0, energy=False,
                  extra=dict(alpha='prsv2', alpha_overrides={'hydrogen': 'std'},
                             alpha_params=PRSV2_PARAMS)),
}


def _load(case):
    g = pd.read_csv(REF_DIR / CASES[case]['csv'])
    if 'status' in g:                    # keep only the rows an independent EOS code reproduces
        g = g[g.status == 'reproduced']
    return g.reset_index(drop=True)


GRIDS = {k: _load(k) for k in CASES}
GRID = GRIDS['pr']                               # kept for the PR-only consistency test


def _method(case):
    c = CASES[case]
    kw = dict(constants_user=INP['constants'], R=R, liquid_volume=c['liquid_volume'])
    if c['eos']:
        kw['eos'] = c['eos']
    kw.update(c.get('extra', {}))
    return METHODS[c['base']].but(**kw)


@lru_cache(maxsize=None)
def stream(T, P, case='pr'):
    """TP flash of the DWSIM feed with DWSIM constants, kij and R."""
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message=r'Zra \(user\)')
        return Stream(
            description='grid', fm=None, comp_mass=None, fn=100.0,
            comp_mol=dict(zip(NAMES, INP['feed_mole_fractions'])), V_frac=None,
            T=float(T), P=float(P), Tref=INP['Tref_K'], Pref=1.01325, psat_factor=None,
            gamma=None, component_data=load_database(), kij=CASES[case]['kij'],
            k_mode='cubic', method=_method(case), emit_warnings=False)


def row_id(r, case=None):
    s = f"T{r.T_K:g}_P{r.P_bar:g}"
    return f"{case}_{s}" if case else s


def equilibrium_diffs(r, case='pr'):
    """Dict of differences thermo-sim - DWSIM for one grid row (absent where not applicable)."""
    p = stream(r.T_K, r.P_bar, case).props
    two = r.n_phases == 2
    out = {'dV': p.V_frac - r.vapor_fraction,
           'dZv': p.Z_gas - r.Z_vapor}
    phi_v = np.asarray(p.phi_v, float)
    for i, n in enumerate(SPECIES):
        out['rel_phi_v_' + n] = phi_v[i] / r['phi_v_' + n] - 1.0
    if two:
        out['rel_Zl'] = p.Z_liq_eos / r.Z_liquid - 1.0
        phi_l = np.atleast_1d(np.asarray(p.phi_l, float))
        if phi_l.size != len(SPECIES):         # thermo-sim found a single phase: no liquid values
            phi_l = np.full(len(SPECIES), np.nan)
        for i, n in enumerate(SPECIES):
            out['rel_phi_l_' + n] = phi_l[i] / r['phi_l_' + n] - 1.0
    if (not two) or CASES[case]['density_two_phase']:
        out['rel_rho'] = p.dens_mix / r.rho_mix_kg_m3 - 1.0
    return out


def energy_rows(case='pr'):
    """Rows with a 1-bar row at the same T (needed for the isothermal differences).

    Empty for cases whose DWSIM energies are not derived consistently from the EOS (PRSV2)."""
    g = GRIDS[case]
    if not CASES[case].get('energy', True):
        return g.iloc[0:0]
    has1 = set(g.loc[g.P_bar == 1, 'T_K'])
    return g[g.T_K.isin(has1) & (g.P_bar > 1)]


def energy_diffs(r, case='pr'):
    """dH/(R T) and dS/R of  X(T,P) - X(T,1 bar)  between thermo-sim and DWSIM.

    The ideal-gas Cp, the formation enthalpy and the reference states cancel in the difference.
    """
    g = GRIDS[case]
    ref = g[(g.T_K == r.T_K) & (g.P_bar == 1)].iloc[0]
    a, b = stream(r.T_K, r.P_bar, case).props, stream(r.T_K, 1.0, case).props
    dH_ts = a.H_total_mix - b.H_total_mix
    dS_ts = a.S_total_mix - b.S_total_mix
    dH_dw = r.H_kJ_kmol - ref.H_kJ_kmol
    dS_dw = r.S_kJ_kmol_K - ref.S_kJ_kmol_K
    rg = R                                        # 8.314 J/(mol K) = 8.314 kJ/(kmol K)
    return {'dH_RT': (dH_ts - dH_dw) / (rg * r.T_K), 'dS_R': (dS_ts - dS_dw) / rg}