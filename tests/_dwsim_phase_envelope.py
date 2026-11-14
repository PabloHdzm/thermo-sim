"""P-T phase envelopes against DWSIM 10.2.12.0 (Peng-Robinson), one entry in CASES per mixture.

natgas: 0.80 CH4 / 0.10 C2H6 / 0.05 C3H8 / 0.03 n-C4H10 / 0.02 n-C5H12
n2co2 : 0.05 N2 / 0.10 CO2 / 0.70 CH4 / 0.10 C2H6 / 0.05 C3H8
Constants, k_ij and R = 8.314 of DWSIM. Reference: reference_data/dwsim/<case>_phase_envelope.csv / .json.
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
CASES = ('natgas', 'n2co2')
P_START = 1.01325                       # bar, initial pressure of the DWSIM custom-initialization run
SCALE = 100.0


@lru_cache(maxsize=None)
def config(case):
    return json.loads((REF_DIR / f'{case}_phase_envelope.json').read_text(encoding='utf-8'))


@lru_cache(maxsize=None)
def _ref(case):
    return pd.read_csv(REF_DIR / f'{case}_phase_envelope.csv')


def names(case):
    return list(config(case)['composition'])


def missing_components(case, db=None):
    have = set((load_database() if db is None else db)['Component'].astype(str).str.lower())
    return [n for n in names(case) if n not in have]


def sections(case):
    return sorted(set(_ref(case)['section']))


def points(case, section):
    d = _ref(case)
    d = d[d['section'] == section]
    return d['T_K'].to_numpy(), d['P_bar'].to_numpy()


def kij_frame(case):
    nm = names(case)
    m = pd.DataFrame(0.0, index=nm, columns=nm)
    for key, v in config(case)['kij'].items():
        a, b = key.split('-')
        m.loc[a, b] = m.loc[b, a] = v
    return m


@lru_cache(maxsize=None)
def feed(case):
    """Mixture at the DWSIM operating point."""
    cfg, op = config(case), config(case)['operating_point']
    db = load_database()
    method = METHODS['PR'].but(constants_user=cfg['constants'], R=cfg['R'], liquid_volume='eos')
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message=r'Zra \(user\)')
        return Stream(
            description=case, fm=None, comp_mass=None, fn=1e-3, comp_mol=dict(cfg['composition']), V_frac=None,
            T=op['T_K'], P=op['P_bar'], Tref=298.15, Pref=1.01325, psat_factor=None, gamma=None,
            component_data=db[db['Component'].str.lower().isin(names(case))].copy(), kij=kij_frame(case),
            k_mode='cubic', method=method, emit_warnings=False)


@lru_cache(maxsize=None)
def envelope(case):
    pe = PhaseEnvelope.from_stream(feed(case))
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


def branch_distances(case, section, branch):
    _, r = envelope(case)
    T, P = points(case, section)
    keep = P >= P_START - 1e-6                       # DWSIM default init starts below 1 atm
    Tc, Pc = (r.T_bubble, r.P_bubble) if branch == 'bubble' else (r.T_dew, r.P_dew)
    return T[keep], P[keep], distance(Tc, Pc, T[keep], P[keep])


def special_points(case):
    """thermo-sim minus DWSIM for the critical point, cricondentherm and cricondenbar.

    DWSIM only reports its own grid (about 1 K / 0.25 bar steps), so the extrema it prints are lower
    bounds of the true ones: the cricondenbar difference is >= 0 by up to the grid sampling."""
    _, r = envelope(case)
    Tc, Pc = points(case, 'critical')
    Td, Pd = points(case, 'dew')
    i_t, i_p = np.argmax(Td), np.argmax(Pd)
    plateau = Pd[Td >= Td.max() - 1e-9]
    return {
        'critical dT [K]': r.critical_point[0] - Tc[0],
        'critical dP [bar]': r.critical_point[1] - Pc[0],
        'cricondentherm dT [K]': r.cricondentherm[0] - Td[i_t],
        'cricondentherm P - nearest DWSIM plateau [bar]': float(
            r.cricondentherm[1] - np.clip(r.cricondentherm[1], plateau.min(), plateau.max())),
        'cricondenbar dP [bar]': r.cricondenbar[1] - Pd[i_p],
        'cricondenbar dT [K] (DWSIM grid 1 K)': r.cricondenbar[0] - Td[i_p],
    }


def pip_diffs(case):
    """Relative difference of the phase identification curve (DWSIM P > 1 bar only: 1E-5 is an artifact)."""
    pe, _ = envelope(case)
    T, P = points(case, 'pip')
    keep = P > 1.0
    mine = pe.phase_identification_curve(T[keep])
    return T[keep], P[keep], mine / P[keep] - 1.0