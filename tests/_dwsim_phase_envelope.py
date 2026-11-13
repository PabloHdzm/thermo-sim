"""P-T phase envelope of a natural-gas-like mixture against DWSIM 10.2.12.0 (Peng-Robinson).

0.80 CH4 / 0.10 C2H6 / 0.05 C3H8 / 0.03 n-C4H10 / 0.02 n-C5H12, k_ij and constants of DWSIM, R = 8.314.
Reference: reference_data/dwsim/natgas_phase_envelope.csv and natgas_phase_envelope.json.
Shared by tests/test_dwsim_phase_envelope.py and scripts/compare_dwsim_phase_envelope.py.

Distances between curves are measured in the plane (T [K], 100 ln P): 0.01 is 0.01 K or 1e-4 in
relative pressure, whichever is larger.
"""
import json
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from thermo_sim import METHODS, Stream, load_database
from thermo_sim.phase_envelope import PhaseEnvelope

ROOT = Path(__file__).resolve().parents[1]
REF_DIR = ROOT / 'reference_data' / 'dwsim'
REF = pd.read_csv(REF_DIR / 'natgas_phase_envelope.csv')
CFG = json.loads((REF_DIR / 'natgas_phase_envelope.json').read_text(encoding='utf-8'))
NAMES = list(CFG['composition'])
P_START = 1.01325                       # bar, initial pressure of the DWSIM custom-initialization run
SCALE = 100.0


def missing_components(db=None):
    have = set((load_database() if db is None else db)['Component'].astype(str).str.lower())
    return [n for n in NAMES if n not in have]


def points(section):
    d = REF[REF['section'] == section]
    return d['T_K'].to_numpy(), d['P_bar'].to_numpy()


def kij_frame():
    m = pd.DataFrame(0.0, index=NAMES, columns=NAMES)
    for key, v in CFG['kij'].items():
        a, b = key.split('-')
        m.loc[a, b] = m.loc[b, a] = v
    return m


@lru_cache(maxsize=None)
def feed():
    """Mixture at the DWSIM operating point (300 K, 1 atm)."""
    op = CFG['operating_point']
    db = load_database()
    comp = CFG['composition']
    method = METHODS['PR'].but(constants_user=CFG['constants'], R=CFG['R'], liquid_volume='eos')
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message=r'Zra \(user\)')
        return Stream(
            description='natgas', fm=None, comp_mass=None, fn=1e-3, comp_mol=dict(comp), V_frac=None,
            T=op['T_K'], P=op['P_bar'], Tref=298.15, Pref=1.01325, psat_factor=None, gamma=None,
            component_data=db[db['Component'].str.lower().isin(NAMES)].copy(), kij=kij_frame(),
            k_mode='cubic', method=method, emit_warnings=False)


@lru_cache(maxsize=None)
def envelope():
    pe = PhaseEnvelope.from_stream(feed())
    return pe, pe.trace(P_START, ds_max=0.05, max_points=1500)


def distance(Tc, Pc, T, P):
    """Distance of each point (T, P) to the polyline (Tc, Pc), in (K, 100 ln P)."""
    c = np.column_stack([Tc, SCALE * np.log(Pc)])
    a, ab = c[:-1], np.diff(c, axis=0)
    out = []
    for t_, p_ in zip(T, P):
        p = np.array([t_, SCALE * np.log(p_)])
        u = np.clip(np.einsum('ij,ij->i', p - a, ab) / (np.einsum('ij,ij->i', ab, ab) + 1e-30), 0, 1)
        out.append(float(np.min(np.linalg.norm(p - (a + u[:, None] * ab), axis=1))))
    return np.array(out)


def branch_distances(section, branch, init=None):
    _, r = envelope()
    T, P = points(section)
    keep = P >= P_START - 1e-6                       # DWSIM default init starts below 1 atm
    Tc, Pc = (r.T_bubble, r.P_bubble) if branch == 'bubble' else (r.T_dew, r.P_dew)
    return T[keep], P[keep], distance(Tc, Pc, T[keep], P[keep])


def special_points():
    """thermo-sim minus DWSIM for the critical point, cricondentherm and cricondenbar."""
    _, r = envelope()
    Tc, Pc = points('critical')
    Td, Pd = points('dew')
    i_t, i_p = np.argmax(Td), np.argmax(Pd)
    return {
        'critical dT [K]': r.critical_point[0] - Tc[0],
        'critical dP [bar]': r.critical_point[1] - Pc[0],
        'cricondentherm dT [K]': r.cricondentherm[0] - Td[i_t],
        'cricondenbar dP [bar]': r.cricondenbar[1] - Pd[i_p],
        'cricondenbar dT [K] (DWSIM grid 1 K)': r.cricondenbar[0] - Td[i_p],
    }


def pip_diffs():
    """Relative difference of the phase identification curve (DWSIM P > 1 bar only: 1E-5 is an artifact)."""
    pe, _ = envelope()
    T, P = points('pip')
    keep = P > 1.0
    mine = pe.phase_identification_curve(T[keep])
    return T[keep], P[keep], mine / P[keep] - 1.0
