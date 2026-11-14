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
PH_CASES = tuple(c for c in CASES if (REF_DIR / f'{c}_phase_envelope_ph.csv').exists())
PS_CASES = tuple(c for c in CASES if (REF_DIR / f'{c}_phase_envelope_ps.csv').exists())
TH_CASES = tuple(c for c in CASES if (REF_DIR / f'{c}_phase_envelope_th.csv').exists())
TS_CASES = tuple(c for c in CASES if (REF_DIR / f'{c}_phase_envelope_ts.csv').exists())
VP_CASES = tuple(c for c in CASES if (REF_DIR / f'{c}_phase_envelope_vp.csv').exists())
VT_CASES = tuple(c for c in CASES if (REF_DIR / f'{c}_phase_envelope_vt.csv').exists())
TREF = 298.15                           # K, reference of the ideal-gas enthalpy (same as DWSIM)
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


BRANCH_SECTIONS = (('bubble_custom_init', 'bubble'), ('bubble_default_init', 'bubble'),
                   ('dew', 'dew'), ('dew_default_init', 'dew'))


def branch_sections(case):
    """(section, branch) of the P-T export of `case`: custom and default initialization of both branches, those
    that exist (DWSIM's default-initialization runs only change where each branch starts)."""
    have = set(_ref(case)['section'])
    return [(s_, b_) for s_, b_ in BRANCH_SECTIONS if s_ in have]


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


# ---------------------------------------------------------------------------------------------------
# Enthalpy along the envelope (PH diagram)
#
# DWSIM prints the mixture enthalpy [kJ/kg] with the ideal gas at 298.15 K as reference and no formation
# enthalpy; thermo-sim H = ideal gas (database Cp, with its linear extension outside the fitted range)
# + EOS departure, formation enthalpy left out. The database Cp differs from the DWSIM Cp equation
# (1 to 3 % at 300 K, more below the 200 K limit of the fit), so two comparisons are made:
#   raw       thermo-sim with the database Cp against DWSIM;
#   corrected thermo-sim minus  z . integral(Cp_database - Cp_DWSIM) dT, i.e. what the EOS part alone gives.
# ---------------------------------------------------------------------------------------------------
_GL_X, _GL_W = np.polynomial.legendre.leggauss(64)


@lru_cache(maxsize=None)
def _ph_ref(case):
    return pd.read_csv(REF_DIR / f'{case}_phase_envelope_ph.csv')


def ph_points(case, section):
    d = _ph_ref(case)
    d = d[d['section'] == section]
    return d['value'].to_numpy(), d['P_bar'].to_numpy()


def _stream_z(case):
    return np.asarray(feed(case).z, float)


def _stream_names(case):
    return [str(c).lower() for c in feed(case).component_data['Component']]


def _int_cp_database(case, T):
    """Per component, integral of the database ideal-gas Cp from TREF to T [J/mol]."""
    from thermo_sim.stream import Stream as _S
    return _S._int_cp_gas(feed(case).component_data, TREF, T).to_numpy(dtype=float)


def _int_cp_dwsim(case, T):
    """Per component, integral of the DWSIM ideal-gas Cp from TREF to T [J/mol] (Gauss-Legendre)."""
    cp = config(case)['cp_ideal_gas']
    t = 0.5 * (T - TREF) * _GL_X + 0.5 * (T + TREF)
    out = []
    for n in _stream_names(case):
        c = cp[n]
        f = (c['A'] + np.exp(c['B'] / t + c['C'] + c['D'] * t + c['E'] * t ** 2)) / 1000.0     # J/(mol K)
        out.append(0.5 * (T - TREF) * float(_GL_W @ f))
    return np.array(out)


def _eos_residual(case, T, P, phase):
    s = feed(case)
    return float(s._eos.H_residual(T, P, _stream_z(case), phase))


def mixture_molar_mass(case, source='dwsim'):
    """Molar mass of the mixture [g/mol]: DWSIM's (config `molar_mass`, if it covers every component) or the
    database's."""
    z, nm = _stream_z(case), _stream_names(case)
    mm = config(case).get('molar_mass', {})
    if source == 'dwsim' and all(n in mm for n in nm):
        return float(z @ np.array([mm[n] for n in nm]))
    return float(z @ np.asarray(feed(case).MM, float))         # no DWSIM molar masses for this case


def cp_enthalpy_difference(case, T):
    """z . integral(Cp_database - Cp_DWSIM) dT from TREF to T [J/mol]."""
    z = _stream_z(case)
    return float(z @ (_int_cp_database(case, T) - _int_cp_dwsim(case, T)))


def saturation_enthalpy(case):
    """Enthalpy [kJ/kg] along the traced envelope: dict with arrays P, H, dH_cp for the bubble branch
    (saturated liquid) and the dew branch (saturated vapor), both ending at the critical point."""
    pe, r = envelope(case)
    z, MM = _stream_z(case), mixture_molar_mass(case)
    out = {}
    for branch, phase in (('bubble', 'l'), ('dew', 'v')):
        T = [p.T for p in r.path if p.branch == branch] + [r.critical_point[0]]
        P = [p.P for p in r.path if p.branch == branch] + [r.critical_point[1]]
        H = [(float(z @ _int_cp_database(case, t)) + _eos_residual(case, t, p, phase)) / MM for t, p in zip(T, P)]
        dC = [cp_enthalpy_difference(case, t) / MM for t in T]
        out[branch] = dict(T=np.array(T), P=np.array(P), H=np.array(H), dH_cp=np.array(dC))
    return out


def ph_distances(case, section, branch, corrected):
    """Distance of the DWSIM points (H, P) to our curve, in (kJ/kg, 100 ln P)."""
    c = saturation_enthalpy(case)[branch]
    H = c['H'] - c['dH_cp'] if corrected else c['H']
    h, P = ph_points(case, section)
    keep = P >= P_START - 1e-6
    return h[keep], P[keep], distance(H, c['P'], h[keep], P[keep])


def operating_point_enthalpy(case):
    """thermo-sim minus DWSIM enthalpy at the operating point (vapor) [kJ/kg]: raw and corrected."""
    op, z, MM = config(case)['operating_point'], _stream_z(case), mixture_molar_mass(case)
    h_dw = ph_points(case, 'operating')[0][0]
    h = (float(z @ _int_cp_database(case, op['T_K'])) + _eos_residual(case, op['T_K'], op['P_bar'], 'v')) / MM
    return h - h_dw, h - cp_enthalpy_difference(case, op['T_K']) / MM - h_dw


def critical_enthalpy(case):
    """thermo-sim minus DWSIM enthalpy at the critical point [kJ/kg], raw and corrected; DWSIM's value is
    the last point of its bubble and dew branches."""
    c = saturation_enthalpy(case)
    h_dw = ph_points(case, 'bubble_custom_init')[0][-1]
    return c['bubble']['H'][-1] - h_dw, c['bubble']['H'][-1] - c['bubble']['dH_cp'][-1] - h_dw


# ---------------------------------------------------------------------------------------------------
# Entropy along the envelope (PS diagram)
#
# thermo-sim S = ideal gas (database Cp, integral of Cp/T from TREF) + ideal mixing (-R sum z ln z)
# + EOS departure - R ln(P / Pref), formation entropy left out (as in the Stream, minus the formation term).
# DWSIM prints S in kJ/(kg K). Its values differ from thermo-sim's by a constant, 0.0821 kJ/(kg K)
# (1.74 J/(mol K)) that does not depend on T or P (it varies 5e-4 along the whole envelope): a
# difference in the entropy reference of DWSIM that cancels in differences. The curves are therefore
# compared after taking out that offset, measured at the operating point (vapor, 300 K, 1 atm); the
# offset itself is reported, and as in the PH diagram the comparison is made raw (database Cp) and
# corrected for the integral of (Cp_database - Cp_DWSIM)/T. Distances are in (J/(kg K), 100 ln P).
# ---------------------------------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _ps_ref(case):
    return pd.read_csv(REF_DIR / f'{case}_phase_envelope_ps.csv')


def ps_points(case, section):
    d = _ps_ref(case)
    d = d[d['section'] == section]
    return d['value'].to_numpy(), d['P_bar'].to_numpy()


def _int_cp_over_T_database(case, T):
    """Per component, integral of the database ideal-gas Cp / T from TREF to T [J/(mol K)]."""
    from thermo_sim.stream import Stream as _S
    return _S._int_cp_gas_over_T(feed(case).component_data, TREF, T).to_numpy(dtype=float)


def _int_cp_over_T_dwsim(case, T):
    cp = config(case)['cp_ideal_gas']
    t = 0.5 * (T - TREF) * _GL_X + 0.5 * (T + TREF)
    out = []
    for n in _stream_names(case):
        c = cp[n]
        f = (c['A'] + np.exp(c['B'] / t + c['C'] + c['D'] * t + c['E'] * t ** 2)) / 1000.0 / t
        out.append(0.5 * (T - TREF) * float(_GL_W @ f))
    return np.array(out)


def _eos_entropy_terms(case, T, P, phase):
    """EOS departure and pressure term [J/(mol K)]."""
    s = feed(case)
    eos = s._eos
    return float(eos.S_residual(T, P, _stream_z(case), phase)) - float(eos.R) * np.log(P / s.Pref)


def _mixing_entropy(case):
    z = _stream_z(case)
    return -float(feed(case)._eos.R) * float(np.sum(z[z > 0] * np.log(z[z > 0])))


def cp_entropy_difference(case, T):
    """z . integral((Cp_database - Cp_DWSIM) / T) dT from TREF to T [J/(mol K)]."""
    return float(_stream_z(case) @ (_int_cp_over_T_database(case, T) - _int_cp_over_T_dwsim(case, T)))


def _entropy(case, T, P, phase):
    """thermo-sim entropy [kJ/(kg K)] with the database Cp, DWSIM molar mass, no formation entropy."""
    return (float(_stream_z(case) @ _int_cp_over_T_database(case, T)) + _mixing_entropy(case)
            + _eos_entropy_terms(case, T, P, phase)) / mixture_molar_mass(case)


def saturation_entropy(case):
    """Entropy [kJ/(kg K)] along the traced envelope, same layout as saturation_enthalpy (dS_cp is the
    Cp-difference term, in kJ/(kg K))."""
    pe, r = envelope(case)
    MM = mixture_molar_mass(case)
    out = {}
    for branch, phase in (('bubble', 'l'), ('dew', 'v')):
        T = [p.T for p in r.path if p.branch == branch] + [r.critical_point[0]]
        P = [p.P for p in r.path if p.branch == branch] + [r.critical_point[1]]
        S = [_entropy(case, t, p, phase) for t, p in zip(T, P)]
        dC = [cp_entropy_difference(case, t) / MM for t in T]
        out[branch] = dict(T=np.array(T), P=np.array(P), S=np.array(S), dS_cp=np.array(dC))
    return out


def entropy_offset(case, corrected=False):
    """thermo-sim minus DWSIM entropy at the operating point (vapor) [kJ/(kg K)]."""
    op = config(case)['operating_point']
    s_dw = ps_points(case, 'operating')[0][0]
    s = _entropy(case, op['T_K'], op['P_bar'], 'v')
    if corrected:
        s -= cp_entropy_difference(case, op['T_K']) / mixture_molar_mass(case)
    return s - s_dw


def ps_distances(case, section, branch, corrected):
    """Distance of the DWSIM points (S, P) to our curve once the offset is taken out, in
    (J/(kg K), 100 ln P)."""
    c = saturation_entropy(case)[branch]
    S = (c['S'] - c['dS_cp'] if corrected else c['S']) - entropy_offset(case, corrected)
    s, P = ps_points(case, section)
    keep = P >= P_START - 1e-6
    Sc = np.column_stack([1000.0 * S, SCALE * np.log(c['P'])])
    a, ab = Sc[:-1], np.diff(Sc, axis=0)
    out = []
    for s_, p_ in zip(s[keep], P[keep]):
        p = np.array([1000.0 * s_, SCALE * np.log(p_)])
        u = np.clip(np.einsum('ij,ij->i', p - a, ab) / (np.einsum('ij,ij->i', ab, ab) + 1e-30), 0, 1)
        out.append(float(np.min(np.linalg.norm(p - (a + u[:, None] * ab), axis=1))))
    return s[keep], P[keep], np.array(out)


def critical_entropy(case):
    """thermo-sim minus DWSIM entropy at the critical point, offset taken out [kJ/(kg K)]: raw, corrected."""
    c = saturation_entropy(case)['bubble']
    s_dw = ps_points(case, 'bubble_custom_init')[0][-1]
    return (c['S'][-1] - entropy_offset(case) - s_dw,
            c['S'][-1] - c['dS_cp'][-1] - entropy_offset(case, True) - s_dw)


# ---------------------------------------------------------------------------------------------------
# Point-by-point tables and curves for the PH and PS diagrams (compare and plot scripts)
# ---------------------------------------------------------------------------------------------------
def _curve(case, kind, branch, corrected=False):
    """(P, T, value) along our branch; value is H [kJ/kg] or S [kJ/(kg K)] with the offset taken out."""
    if kind == 'PH':
        c = saturation_enthalpy(case)[branch]
        v = c['H'] - c['dH_cp'] if corrected else c['H']
    else:
        c = saturation_entropy(case)[branch]
        v = (c['S'] - c['dS_cp'] if corrected else c['S']) - entropy_offset(case, corrected)
    return c['P'], c['T'], v


def comparison_table(case, kind, corrected=False, rows=12):
    """Rows (branch, P, T, DWSIM value, thermo-sim value, difference) at DWSIM's own pressures, on the part
    of each branch where P increases (up to the cricondenbar on the dew branch). Our value is
    interpolated in ln P along our curve."""
    ref = ph_points if kind == 'PH' else ps_points
    out = []
    for section, branch in (('bubble_custom_init', 'bubble'), ('dew', 'dew')):
        x_dw, P_dw = ref(case, section)
        keep = P_dw >= P_START - 1e-6
        x_dw, P_dw = x_dw[keep], P_dw[keep]
        n_dw = int(np.argmax(P_dw)) + 1 if branch == 'dew' else len(P_dw) - 1
        P, T, v = _curve(case, kind, branch, corrected)
        n = int(np.argmax(P)) + 1 if branch == 'dew' else len(P) - 1
        idx = np.unique(np.linspace(0, n_dw - 1, rows).round().astype(int))
        for i in idx:
            if not (P[0] <= P_dw[i] <= P[:n].max()):
                continue
            lnp = np.log(P[:n])
            mine = float(np.interp(np.log(P_dw[i]), lnp, v[:n]))
            t = float(np.interp(np.log(P_dw[i]), lnp, T[:n]))
            out.append((branch, float(P_dw[i]), t, float(x_dw[i]), mine, mine - float(x_dw[i])))
    return out


# ---------------------------------------------------------------------------------------------------
# TH and TS diagrams (temperature as ordinate)
#
# They are the same saturation curves as PH and PS, so H(T, P) and S(T, P) are the ones already validated;
# what these diagrams add is the geometry near the critical point and the cricondentherm, where T is not
# monotonic. Distances are measured in (kJ/kg, K) for TH and (J/(kg K), K) for TS. DWSIM does not
# compute the TH diagram of natgas (calculation failed, index out of range) with or without custom
# initialization, and for n2co2 only without it, so the TH case lists depend on the files present.
# The DWSIM operating point of the TS diagram need not be at the same T as that of the PS diagram (natgas:
# 300 K and 298.15 K), so the entropy offset is measured at the T of each file.
# ---------------------------------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _t_ref(case, kind):
    return pd.read_csv(REF_DIR / f'{case}_phase_envelope_{kind.lower()}.csv')


def t_points(case, kind, section):
    """(value, T_K) of a section of the TH or TS export (H in kJ/kg, S in kJ/(kg K))."""
    d = _t_ref(case, kind)
    d = d[d['section'] == section]
    return d['value'].to_numpy(), d['T_K'].to_numpy()


def bubble_section(case, kind):
    """Name of the bubble section of the TH/TS export: custom or default initialization."""
    names_ = set(_t_ref(case, kind)['section'])
    return 'bubble_custom_init' if 'bubble_custom_init' in names_ else 'bubble_default_init'


def _polyline_distance(cx, cy, px, py):
    c = np.column_stack([cx, cy])
    a, ab = c[:-1], np.diff(c, axis=0)
    out = []
    for x_, y_ in zip(px, py):
        p = np.array([x_, y_])
        u = np.clip(np.einsum('ij,ij->i', p - a, ab) / (np.einsum('ij,ij->i', ab, ab) + 1e-30), 0, 1)
        out.append(float(np.min(np.linalg.norm(p - (a + u[:, None] * ab), axis=1))))
    return np.array(out)


def ts_offset(case, corrected=False):
    """thermo-sim minus DWSIM entropy at the operating point of the TS export (vapor) [kJ/(kg K)]."""
    s_dw, T_op = t_points(case, 'TS', 'operating')
    P_op = config(case)['operating_point']['P_bar']
    s = _entropy(case, float(T_op[0]), P_op, 'v')
    if corrected:
        s -= cp_entropy_difference(case, float(T_op[0])) / mixture_molar_mass(case)
    return s - float(s_dw[0])


def th_distances(case, section, branch, corrected):
    """Distance of the DWSIM points (H, T) to our curve in (kJ/kg, K), for the part of DWSIM's branch above
    the temperature of our first point (1 atm)."""
    c = saturation_enthalpy(case)[branch]
    H = c['H'] - c['dH_cp'] if corrected else c['H']
    h, T = t_points(case, 'TH', section)
    keep = T >= c['T'][0] - 1e-6
    return h[keep], T[keep], _polyline_distance(H, c['T'], h[keep], T[keep])


def ts_distances(case, section, branch, corrected):
    """Distance of the DWSIM points (S, T) to our curve once the offset is taken out, in (J/(kg K), K)."""
    c = saturation_entropy(case)[branch]
    S = (c['S'] - c['dS_cp'] if corrected else c['S']) - ts_offset(case, corrected)
    s, T = t_points(case, 'TS', section)
    keep = T >= c['T'][0] - 1e-6
    return s[keep], T[keep], _polyline_distance(1000.0 * S, c['T'], 1000.0 * s[keep], T[keep])


def operating_point_th(case):
    """thermo-sim minus DWSIM enthalpy at the TH operating point (vapor) [kJ/kg]: raw and corrected."""
    h_dw, T_op = t_points(case, 'TH', 'operating')
    T, P = float(T_op[0]), config(case)['operating_point']['P_bar']
    z, MM = _stream_z(case), mixture_molar_mass(case)
    h = (float(z @ _int_cp_database(case, T)) + _eos_residual(case, T, P, 'v')) / MM
    return h - float(h_dw[0]), h - cp_enthalpy_difference(case, T) / MM - float(h_dw[0])


def _curve_t(case, kind, branch, corrected=False):
    """(T, value) along our branch for the TH or TS diagram; value is H [kJ/kg] or S [kJ/(kg K)] (offset out)."""
    if kind == 'TH':
        c = saturation_enthalpy(case)[branch]
        v = c['H'] - c['dH_cp'] if corrected else c['H']
    else:
        c = saturation_entropy(case)[branch]
        v = (c['S'] - c['dS_cp'] if corrected else c['S']) - ts_offset(case, corrected)
    return c['T'], v


def comparison_table_t(case, kind, corrected=False, rows=14):
    """Rows (branch, T, DWSIM value, thermo-sim value, difference) at DWSIM's own temperatures (above the
    1 atm point of our branch). The dew branch is split at the cricondentherm: points of DWSIM before its
    maximum T are compared with the rising part of our curve, the later ones with the falling part. Our value
    is interpolated in T."""
    out = []
    for branch in ('bubble', 'dew'):
        section = bubble_section(case, kind) if branch == 'bubble' else 'dew'
        x_dw, T_dw = t_points(case, kind, section)
        T, v = _curve_t(case, kind, branch, corrected)
        keep = T_dw >= T[0] - 1e-6
        x_dw, T_dw = x_dw[keep], T_dw[keep]
        if branch == 'dew':
            k_dw, k = int(np.argmax(T_dw)), int(np.argmax(T))
            parts = [(np.arange(0, k_dw + 1), T[:k + 1], v[:k + 1]),
                     (np.arange(k_dw + 1, len(T_dw)), T[k:][::-1], v[k:][::-1])]
        else:
            parts = [(np.arange(len(T_dw)), T, v)]
        for idx, Tp, vp in parts:
            if len(idx) == 0:
                continue
            for i in np.unique(np.linspace(idx[0], idx[-1], max(2, rows // len(parts))).round().astype(int)):
                if not (Tp.min() <= T_dw[i] <= Tp.max()):
                    continue
                mine = float(np.interp(T_dw[i], Tp, vp))
                out.append((branch, float(T_dw[i]), float(x_dw[i]), mine, mine - float(x_dw[i])))
    return out


# ---------------------------------------------------------------------------------------------------
# VP and VT diagrams: molar volume [m3/kmol] of the saturated liquid (bubble) and vapor (dew)
#
# DWSIM prints the volume of each saturated phase as Z R T / P of the cubic EOS (no volume translation:
# the agreement with thermo-sim's EOS roots is 1e-6 relative). Distances are measured in
# (100 ln V, 100 ln P) for VP and (100 ln V, K) for VT, i.e. in percent of volume. Its critical-point volume
# is printed in m3/mol (1000 times smaller than the other points, 8.9e-5 against 8.9e-2 m3/kmol).
# ---------------------------------------------------------------------------------------------------
DWSIM_CRITICAL_VOLUME_FACTOR = 1000.0         # DWSIM prints the critical volume in m3/mol, the rest in m3/kmol


@lru_cache(maxsize=None)
def _v_ref(case, kind):
    return pd.read_csv(REF_DIR / f'{case}_phase_envelope_{kind.lower()}.csv')


def v_points(case, kind, section):
    """(volume [m3/kmol], P [bar] or T [K]) of a section of the VP or VT export."""
    d = _v_ref(case, kind)
    d = d[d['section'] == section]
    return d['value'].to_numpy(), d[d.columns[2]].to_numpy()


def molar_volume(case, T, P, phase):
    """EOS molar volume of the feed composition [m3/kmol] ('l': liquid root, 'v': vapor root)."""
    eos, z = feed(case)._eos, _stream_z(case)
    R = float(eos.R)
    am, bm = eos.mixture_params(T, z)
    A, B = am * P * 1e5 / (R * T) ** 2, bm * P * 1e5 / (R * T)
    Z = eos.Z_roots(A, B)[0] if phase == 'v' else eos.liquid_Z_root(A, B)
    return float(Z * R * T / (P * 1e5) * 1000.0)


def saturation_volume(case):
    """Volume along the traced envelope: dict branch -> T, P, V arrays, both ending at the critical point."""
    pe, r = envelope(case)
    out = {}
    for branch, phase in (('bubble', 'l'), ('dew', 'v')):
        T = [p.T for p in r.path if p.branch == branch] + [r.critical_point[0]]
        P = [p.P for p in r.path if p.branch == branch] + [r.critical_point[1]]
        V = [molar_volume(case, t, p, phase) for t, p in zip(T, P)]
        out[branch] = dict(T=np.array(T), P=np.array(P), V=np.array(V))
    return out


def vp_distances(case, section, branch):
    """Distance of the DWSIM points (V, P) to our curve in (100 ln V, 100 ln P)."""
    c = saturation_volume(case)[branch]
    v, P = v_points(case, 'VP', section)
    keep = P >= P_START - 1e-6
    return v[keep], P[keep], _polyline_distance(SCALE * np.log(c['V']), SCALE * np.log(c['P']),
                                                SCALE * np.log(v[keep]), SCALE * np.log(P[keep]))


def vt_distances(case, section, branch):
    """Distance of the DWSIM points (V, T) to our curve in (100 ln V, K)."""
    c = saturation_volume(case)[branch]
    v, T = v_points(case, 'VT', section)
    keep = T >= c['T'][0] - 1e-6
    return v[keep], T[keep], _polyline_distance(SCALE * np.log(c['V']), c['T'], SCALE * np.log(v[keep]), T[keep])


def critical_volume(case):
    """thermo-sim minus DWSIM critical volume, relative: (liquid root, vapor root) at our critical point."""
    c = saturation_volume(case)
    v_dw = v_points(case, 'VP', 'critical')[0][0] * DWSIM_CRITICAL_VOLUME_FACTOR
    return c['bubble']['V'][-1] / v_dw - 1.0, c['dew']['V'][-1] / v_dw - 1.0


def comparison_table_v(case, kind, rows=14):
    """Rows (branch, P or T, DWSIM V, thermo-sim V, relative difference [%]) at DWSIM's own abscissas, above the
    1 atm point of our branch. The dew branch is split at its maximum (cricondenbar for VP, cricondentherm for
    VT): DWSIM points before its maximum are compared with the rising part of our curve, later ones with the
    falling part. ln V is interpolated in ln P (VP) or T (VT)."""
    key, tr = ('P', np.log) if kind == 'VP' else ('T', lambda x: x)
    out = []
    for branch in ('bubble', 'dew'):
        section = 'bubble_custom_init' if branch == 'bubble' else 'dew'
        v_dw, x_dw = v_points(case, kind, section)
        c = saturation_volume(case)[branch]
        x, lv = tr(c[key]), np.log(c['V'])
        keep = (x_dw >= (P_START if key == 'P' else c['T'][0]) - 1e-6)
        v_dw, x_dw = v_dw[keep], x_dw[keep]
        if branch == 'dew':
            k_dw, k = int(np.argmax(x_dw)), int(np.argmax(x))
            parts = [(np.arange(0, k_dw + 1), x[:k + 1], lv[:k + 1]),
                     (np.arange(k_dw + 1, len(x_dw)), x[k:][::-1], lv[k:][::-1])]
        else:
            parts = [(np.arange(len(x_dw)), x, lv)]
        for idx, xp, lp in parts:
            if len(idx) == 0:
                continue
            for i in np.unique(np.linspace(idx[0], idx[-1], max(2, rows // len(parts))).round().astype(int)):
                xi = tr(x_dw[i])
                if not (xp.min() <= xi <= xp.max()):
                    continue
                mine = float(np.exp(np.interp(xi, xp, lp)))
                out.append((branch, float(x_dw[i]), float(v_dw[i]), mine, 100.0 * (mine / float(v_dw[i]) - 1.0)))
    return out