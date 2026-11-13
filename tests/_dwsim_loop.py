"""Ammonia synthesis loop (DWSIM tutorial flowsheet) run with thermo-sim units.

feed (300 K, 30 bar, 4 mol/s, H2 0.75 / N2 0.25) -> compressor (200 bar, eta 0.75) -> heater
(700 K) -> equilibrium reactor (isothermal, ln K = 11000/T - 25, as in the DWSIM reaction set) ->
cooler (250 K) -> flash drum (250 K).  DWSIM: PR, inside-out flash, liquid volume from the EOS,
kij and critical constants of the grid runs, R = 8.314.
Reference: reference_data/dwsim/ammonia_loop_streams.csv and ammonia_loop_units.json.
Shared by tests/test_dwsim_loop.py and scripts/compare_dwsim_loop.py.
"""
import json
import warnings
from functools import lru_cache

import numpy as np
from numpy.polynomial import Polynomial
import pandas as pd

from thermo_sim import METHODS, Stream, load_database
from thermo_sim.unitops import Compressor, Cooler, EquilibriumReactor, FlashDrum, StoichReactor

from _dwsim_grid import INP, NAMES, R, REF_DIR

# kij as a DataFrame indexed by component name: each stream takes the sub-matrix of its active
# components (the feed has no NH3), so no special matrix is needed per stream.
KIJ = pd.DataFrame(np.array(INP['kij'], float), index=NAMES, columns=NAMES)

STREAMS = pd.read_csv(REF_DIR / 'ammonia_loop_streams.csv').set_index('stream')
UNITS = json.loads((REF_DIR / 'ammonia_loop_units.json').read_text(encoding='utf-8'))
CP_DW = json.loads((REF_DIR / 'ammonia_loop_cp.json').read_text(encoding='utf-8'))
SPECIES = ['H2', 'N2', 'NH3']                      # order of NAMES: hydrogen, nitrogen, ammonia
SYNTHESIS = {'hydrogen': -3.0, 'nitrogen': -1.0, 'ammonia': 2.0}


CP_FIT_RANGE = (240.0, 720.0)          # K, covers the loop (250 to 700 K)


def dwsim_cp(i, T):
    """DWSIM ideal-gas Cp of component i (CP_DW order) [kJ/(kmol K)] = [J/(mol K)]."""
    d = CP_DW
    return (d['A'][i] + np.exp(d['B'][i] / T + d['C'][i] + d['D'][i] * T + d['E'][i] * T**2)) / 1000.0


_CP_LETTERS = 'ABCDEFGH'


@lru_cache(maxsize=None)
def database():
    """Database with the ideal-gas Cp of H2, N2 and NH3 replaced by DWSIM's.

    The database stores a polynomial in T (columns A_Cp_Gas ... up to G_Cp_Gas, degree 6);
    DWSIM's A + exp(B/T + C + D T + E T^2) is fitted by least squares with a quartic over
    CP_FIT_RANGE and ALL the other Cp columns (F, G, ...) are set to zero, otherwise the
    packaged high-order terms would still be added. Cp is within 7e-4 at worst (H2), against
    0.3 to 0.5 % for the CoolProp-based polynomial of the packaged database.
    """
    db = load_database().copy(deep=True)
    key = db['Component'].astype(str).str.lower()
    Tfit = np.linspace(*CP_FIT_RANGE, 500)
    for i, name in enumerate(CP_DW['components']):
        coef = Polynomial.fit(Tfit, dwsim_cp(i, Tfit), 4).convert().coef
        row = (key == name).to_numpy()
        for letter in _CP_LETTERS:
            if f'{letter}_Cp_Gas' in db.columns:
                db.loc[row, f'{letter}_Cp_Gas'] = 0.0
        for k, letter in enumerate('ABCDE'):
            db.loc[row, f'{letter}_Cp_Gas'] = coef[k]
        db.loc[row, 'Tmin_Cp_Gas'], db.loc[row, 'Tmax_Cp_Gas'] = CP_FIT_RANGE
    return db


def _feed(db=None):
    method = METHODS['PR'].but(constants_user=INP['constants'], R=R, liquid_volume='eos')
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message=r'Zra \(user\)')
        return Stream(
            description='feed', fm=None, comp_mass=None, fn=0.004,        # 4 mol/s in kmol/s
            comp_mol={'hydrogen': 0.75, 'nitrogen': 0.25, 'ammonia': 0.0}, V_frac=None,
            T=300.0, P=30.0, Tref=INP['Tref_K'], Pref=1.01325, psat_factor=None, gamma=None,
            component_data=database() if db is None else db, kij=KIJ, k_mode='cubic',
            method=method, emit_warnings=False)


@lru_cache(maxsize=None)
def loop():
    """Run the flowsheet; returns a dict with the unit results and the streams by DWSIM name."""
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message=r'Zra \(user\)')   # NH3 Zra = 0 is not used (no Peneloux)
        return _run_loop()


@lru_cache(maxsize=None)
def loop_packaged():
    """Same flowsheet with the packaged database (CoolProp-based ideal-gas Cp)."""
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message=r'Zra \(user\)')
        return _run_loop(load_database())


P0_DWSIM = 1.01325   # bar


def _run_loop(db=None):
    u = UNITS
    feed = _feed(db)
    comp = Compressor(u['compressor']['P_out_bar'], eta=u['compressor']['eta_adiabatic']).run(feed)
    heat = Cooler(u['heater']['T_out_K'], dP=u['heater']['dP_bar']).run(comp.outlet)
    # DWSIM conventions (see reference_data/dwsim/README.md): quotient referred to P0 = 1 atm and a
    # user ln K(T) expression evaluated at T + 1 K.
    rx = EquilibriumReactor(SYNTHESIS, dP=u['reactor']['dP_bar'], P_ref=P0_DWSIM,
                            lnK=lambda T: 11000.0 / (T + 1.0) - 25.0).run(heat.outlet)
    cool = Cooler(u['cooler']['T_out_K'], dP=u['cooler']['dP_bar'],
                  allow_condensation=True).run(rx.outlet)
    drum = FlashDrum(T=u['flash']['T_K']).run(cool.outlet)
    streams = {'feed': feed, 'comp_out': comp.outlet, 'hot': heat.outlet, 'rx_out': rx.outlet,
               'cold': cool.outlet, 'purge': drum.vapor, 'nh3_liquid': drum.liquid}
    return dict(comp=comp, heat=heat, rx=rx, cool=cool, drum=drum, streams=streams)


@lru_cache(maxsize=None)
def fixed_conversion_reactor():
    """StoichReactor at DWSIM's converged conversion on the heater outlet (same inlet as R-1)."""
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message=r'Zra \(user\)')
        return StoichReactor(SYNTHESIS, conversion=UNITS['reactor']['conversion_N2'], key='nitrogen',
                             dP=UNITS['reactor']['dP_bar']).run(loop()['streams']['hot'])


def stream_diffs(name):
    """Dict of thermo-sim - DWSIM differences for one stream (relative where noted)."""
    s, ref = loop()['streams'][name], STREAMS.loc[name]
    p = s.props
    out = {}
    out['dT'] = s.T - ref.T_K
    out['dP'] = s.P - ref.P_bar
    n_dw = np.array([ref.n_H2_mol_s, ref.n_N2_mol_s, ref.n_NH3_mol_s]) / 1000.0     # kmol/s
    n_ts = s.fn * np.asarray(s.z, float)
    for i, c in enumerate(SPECIES):
        if n_dw[i] > 0:
            out['rel_n_' + c] = n_ts[i] / n_dw[i] - 1.0
    out['abs_fn'] = s.fn - n_dw.sum()
    if name == 'cold':
        out['dV'] = float(p.V_frac) - ref.vapor_fraction
    if ref.vapor_fraction > 0:                       # vapor phase present
        out['rel_Zv'] = p.Z_gas / ref.Z_vapor - 1.0
        yi = np.asarray(p.yi, float) if name == 'cold' else np.asarray(s.z, float)
        for i, c in enumerate(SPECIES):
            yd = ref['y_' + c]
            if yd > 1e-6:
                out['rel_y_' + c] = yi[i] / yd - 1.0
    if ref.vapor_fraction < 1:                       # liquid phase present
        out['rel_Zl'] = p.Z_liq_eos / ref.Z_liquid - 1.0
        xi = np.asarray(p.xi, float) if name == 'cold' else np.asarray(s.z, float)
        for i, c in enumerate(SPECIES):
            out['rel_x_' + c] = xi[i] / ref['x_' + c] - 1.0
    if ref.vapor_fraction in (0.0, 1.0):             # single-phase streams: mixture density
        out['rel_rho'] = p.dens_mix / ref.rho_mix_kg_m3 - 1.0
    return out


def unit_diffs(L=None):
    """Duties and powers: thermo-sim - DWSIM (kW, signed as heat added to the stream).
    L: result of loop() (default) or loop_packaged()."""
    L, u = (loop() if L is None else L), UNITS
    return {
        'compressor T_out [K]': L['comp'].T_out - u['compressor']['T_out_K'],
        'compressor power [kW]': L['comp'].W_brake_kW - u['compressor']['power_kW'],
        'heater duty [kW]': L['heat'].duty_kW - u['heater']['duty_kW'],
        'reactor duty [kW]': L['rx'].duty_kW - u['reactor']['duty_kW'],
        'cooler duty [kW]': L['cool'].duty_kW - u['cooler']['duty_kW'],
        'cooler vapor fraction': float(L['cool'].outlet.props.V_frac) - u['cooler']['vapor_fraction'],
    }


def enthalpy_diffs():
    """Total molar enthalpy (with formation enthalpy), thermo-sim - DWSIM [kJ/kmol].

    DWSIM's displayed 'Molar Enthalpy' leaves out the formation enthalpy, while its 'Energy
    Flow' (used in the unit balances) includes it; the reference is Energy Flow / molar flow.
    Reported absolute and relative to the feed, which cancels any reference-state offset.
    """
    out = {}
    L = loop()
    h0_ts = L['streams']['feed'].props.H_total_mix
    h0_dw = STREAMS.loc['feed', 'energy_flow_per_kmol_kJ']
    for name, s in L['streams'].items():
        h_ts, h_dw = s.props.H_total_mix, STREAMS.loc[name, 'energy_flow_per_kmol_kJ']
        out[name + ' H abs'] = h_ts - h_dw
        out[name + ' H - H(feed)'] = (h_ts - h0_ts) - (h_dw - h0_dw)
    return out


def cp_ideal_diffs(db=None):
    """Ideal-gas mixture Cp of the database polynomial minus DWSIM's, per stream: dict of
    (database, DWSIM, relative difference) [kJ/(kmol K)], at the stream T and vapor composition.
    Pass `load_database()` to see the packaged (CoolProp-based) polynomial."""
    db = (database() if db is None else db).copy()
    db['Component'] = db['Component'].astype(str).str.lower()
    db = db.set_index('Component')
    out = {}
    for name, ref in STREAMS.iterrows():
        if not np.isfinite(ref.Cp_ideal_vapor_kJ_kmol_K):
            continue
        T = float(ref.T_K)
        cp = 0.0
        for c, sp in zip(NAMES, SPECIES):
            y = ref['y_' + sp]
            if y > 0:
                row = db.loc[c]
                cp += y * sum(row[f'{k}_Cp_Gas'] * T**j for j, k in enumerate(_CP_LETTERS)
                              if f'{k}_Cp_Gas' in row.index)
        out[name] = (cp, ref.Cp_ideal_vapor_kJ_kmol_K, cp / ref.Cp_ideal_vapor_kJ_kmol_K - 1.0)
    return out
