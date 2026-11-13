"""Propane refrigeration cycle (DWSIM tutorial flowsheet) run with thermo-sim units.

1_evap_out (saturated vapor, 2.45 bar, 1 mol/s) -> compressor (12.5 bar, eta_is 0.8) ->
condenser (305 K) -> isenthalpic valve (2.45 bar) -> evaporator (to saturated vapor, DWSIM: 253 K)
-> 5_evap_out.
DWSIM: Peng-Robinson, propane constants and ideal-gas Cp of the DWSIM compound, R = 8.314.
Reference: reference_data/dwsim/propane_cycle_streams.csv and propane_cycle_units.json.
Shared by tests/test_dwsim_cycle.py and scripts/compare_dwsim_cycle.py.

Saturation: DWSIM locates the saturation points of a pure component with its vapor-pressure
correlation (see propane_cycle_units.json), thermo-sim with the EOS, 0.226 K apart at 2.45 bar.
Streams on the dome are therefore compared with that offset in mind (see the tests).

Formation enthalpy: H_total of thermo-sim includes the formation enthalpy of the database
(-104.7 kJ/mol), DWSIM's Energy Flow its own (-104.68 kJ/mol), so absolute enthalpies are only
compared as differences with respect to the first stream.
"""
import json
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.polynomial import Polynomial

from thermo_sim import METHODS, Stream, load_database
from thermo_sim.unitops import Compressor, Cooler, Valve

ROOT = Path(__file__).resolve().parents[1]
REF_DIR = ROOT / 'reference_data' / 'dwsim'
STREAMS = pd.read_csv(REF_DIR / 'propane_cycle_streams.csv').set_index('stream')
UNITS = json.loads((REF_DIR / 'propane_cycle_units.json').read_text(encoding='utf-8'))
R = 8.314
TREF, PREF = 298.15, 1.01325
NAME = 'propane'
CP_FIT_RANGE = (230.0, 340.0)          # K, covers the cycle (253 to 325 K)
_CP_LETTERS = 'ABCDEFGH'


def dwsim_cp(T):
    """DWSIM ideal-gas Cp of propane [kJ/(kmol K)] = [J/(mol K)]."""
    d = UNITS['cp']
    return (d['A'] + np.exp(d['B'] / T + d['C'] + d['D'] * T + d['E'] * T**2)) / 1000.0


@lru_cache(maxsize=None)
def database():
    """Packaged database with the ideal-gas Cp of propane replaced by DWSIM's equation.

    The database stores a polynomial in T (columns A_Cp_Gas ... G_Cp_Gas); DWSIM's equation is
    fitted with a quartic over CP_FIT_RANGE and all the other Cp columns are set to zero."""
    db = load_database().copy(deep=True)
    row = (db['Component'].astype(str).str.lower() == NAME).to_numpy()
    if not row.any():
        raise KeyError("propane is not in the packaged database: run scripts/build_db.py")
    Tfit = np.linspace(*CP_FIT_RANGE, 200)
    coef = Polynomial.fit(Tfit, dwsim_cp(Tfit), 4).convert().coef
    for letter in _CP_LETTERS:
        if f'{letter}_Cp_Gas' in db.columns:
            db.loc[row, f'{letter}_Cp_Gas'] = 0.0
    for k, letter in enumerate('ABCDE'):
        db.loc[row, f'{letter}_Cp_Gas'] = coef[k]
    db.loc[row, 'Tmin_Cp_Gas'], db.loc[row, 'Tmax_Cp_Gas'] = CP_FIT_RANGE
    return db


def _method():
    return METHODS['PR'].but(constants_user=UNITS['constants'], R=R, liquid_volume='eos')


def _feed(db=None):
    f = UNITS['feed']
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message=r'Zra \(user\)')
        return Stream(
            description='1_evap_out', fm=None, comp_mass=None, fn=f['n_mol_s'] / 1000.0,
            comp_mol={NAME: 1.0}, V_frac=f['vapor_fraction'], T=253.0, P=f['P_bar'],  # T: first guess
            Tref=TREF, Pref=PREF, psat_factor=None, gamma=None,
            component_data=database() if db is None else db, kij=np.zeros((1, 1)),
            k_mode='cubic', method=_method(), emit_warnings=False)


@lru_cache(maxsize=None)
def cycle():
    """Run the flowsheet with DWSIM's Cp; returns the unit results and the streams by DWSIM name."""
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message=r'Zra \(user\)')
        return _run_cycle()


@lru_cache(maxsize=None)
def cycle_packaged():
    """Same flowsheet with the packaged database (CoolProp-based ideal-gas Cp)."""
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message=r'Zra \(user\)')
        return _run_cycle(load_database())


def _run_cycle(db=None):
    u = UNITS
    s1 = _feed(db)
    comp = Compressor(u['compressor']['P_out_bar'], eta=u['compressor']['eta_adiabatic']).run(s1)
    cond = Cooler(u['condenser']['T_out_K'], dP=u['condenser']['dP_bar'],
                  allow_condensation=True).run(comp.outlet)
    valve = Valve(u['valve']['P_out_bar']).run(cond.outlet)
    # DWSIM heats to 253 K, 0.005 K above its own (correlation) dew point; in the EOS that point is
    # 253.221 K, so the outlet is specified as saturated vapor, the state DWSIM is describing.
    evap = Cooler(V_frac=1.0, dP=u['evaporator']['dP_bar']).run(valve.outlet)
    streams = {'1_evap_out': s1, '2_comp_out': comp.outlet, '3_cond_out': cond.outlet,
               '4_valve_out': valve.outlet, '5_evap_out': evap.outlet}
    return dict(comp=comp, cond=cond, valve=valve, evap=evap, streams=streams)


def cop(L=None):
    L = cycle() if L is None else L
    return L['evap'].duty_kW / L['comp'].W_brake_kW


def stream_diffs(name, L=None):
    """thermo-sim - DWSIM for one stream (relative where noted)."""
    s, ref = (cycle() if L is None else L)['streams'][name], STREAMS.loc[name]
    p = s.props
    out = {'dT': s.T - ref.T_K, 'dP': s.P - ref.P_bar,
           'dV': float(p.V_frac) - ref.vapor_fraction}
    if ref.vapor_fraction > 0:
        out['rel_Zv'] = p.Z_gas / ref.Z_vapor - 1.0
    if ref.vapor_fraction < 1:
        out['rel_Zl'] = p.Z_liq_eos / ref.Z_liquid - 1.0
    if ref.vapor_fraction in (0.0, 1.0):               # single phase: mixture density
        out['rel_rho'] = p.dens_mix / ref.rho_mix_kg_m3 - 1.0
    return out


def unit_diffs(L=None):
    """Duties and power: thermo-sim - DWSIM (kW, heat added to the stream)."""
    L, u = (cycle() if L is None else L), UNITS
    return {
        'compressor T_out [K]': L['comp'].T_out - u['compressor']['T_out_K'],
        'compressor T rise [K]': (L['comp'].T_out - L['comp'].T_in)
                                 - (u['compressor']['T_out_K'] - STREAMS.loc['1_evap_out', 'T_K']),
        'compressor power [kW]': L['comp'].W_brake_kW - u['compressor']['power_kW'],
        'condenser duty [kW]': L['cond'].duty_kW - u['condenser']['duty_kW'],
        'evaporator duty [kW]': L['evap'].duty_kW - u['evaporator']['duty_kW'],
        'valve outlet vapor fraction': float(L['valve'].outlet.props.V_frac)
                                       - STREAMS.loc['4_valve_out', 'vapor_fraction'],
        'valve dH [kJ/kmol]': L['valve'].H_out - L['valve'].H_in,
        'COP': cop(L) - u['cop']['value'],
    }


def enthalpy_diffs(L=None):
    """H_total (with formation enthalpy) relative to stream 1, thermo-sim - DWSIM [kJ/kmol]."""
    L = cycle() if L is None else L
    h0_ts = L['streams']['1_evap_out'].props.H_total_mix
    h0_dw = STREAMS.loc['1_evap_out', 'energy_flow_per_kmol_kJ']
    return {name + ' H - H(1)': (s.props.H_total_mix - h0_ts) - (STREAMS.loc[name, 'energy_flow_per_kmol_kJ'] - h0_dw)
            for name, s in L['streams'].items()}


def cp_ideal_diffs(db=None):
    """Ideal-gas Cp of the database polynomial vs DWSIM per vapor stream: (database, DWSIM, relative)."""
    db = (database() if db is None else db).copy()
    db['Component'] = db['Component'].astype(str).str.lower()
    row = db.set_index('Component').loc[NAME]
    out = {}
    for name, ref in STREAMS.iterrows():
        if not np.isfinite(ref.Cp_ideal_vapor_kJ_kmol_K):
            continue
        T = float(ref.T_K)
        cp = sum(row[f'{k}_Cp_Gas'] * T**j for j, k in enumerate(_CP_LETTERS) if f'{k}_Cp_Gas' in row.index)
        out[name] = (cp, ref.Cp_ideal_vapor_kJ_kmol_K, cp / ref.Cp_ideal_vapor_kJ_kmol_K - 1.0)
    return out