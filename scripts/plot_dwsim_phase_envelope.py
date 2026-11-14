"""Plot the phase envelope of thermo-sim over the DWSIM points (same layout as the DWSIM charts).

Run from the repository root:
    python scripts/plot_dwsim_phase_envelope.py [natgas|n2co2] [PT|PH|PS|TH|TS|VP|VT] [output.png]
Writes docs/figures/<case>_phase_envelope[_<kind>].png by default (needs matplotlib).

PT: pressure against temperature, with the phase identification curve, critical point, cricondentherm,
    cricondenbar and the operating point.
PH, PS: pressure against specific enthalpy [kJ/kg] or entropy [kJ/(kg K)] of the saturated liquid
    (bubble) and vapor (dew), database Cp, entropy offset to DWSIM taken out (see the README section).
VP, VT: pressure or temperature against the molar volume [m3/kmol] (log axis) of the saturated phases.
TH, TS: temperature against the same enthalpy or entropy (DWSIM does not export TH for natgas).
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests'))

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from _dwsim_phase_envelope import (CASES, PH_CASES, PS_CASES, TH_CASES, TS_CASES, VP_CASES, VT_CASES, _curve,  # noqa: E402
                                   _curve_t, bubble_section, config, envelope, missing_components, ph_points, points,
                                   ps_points, saturation_volume, sections, t_points, v_points)

args = sys.argv[1:]
case = args[0] if args else 'natgas'
KINDS = ('PT', 'PH', 'PS', 'TH', 'TS', 'VP', 'VT')
kind = args[1].upper() if len(args) > 1 and args[1].upper() in KINDS else 'PT'
out_arg = [a for a in args[1:] if a.upper() not in KINDS]
if case not in CASES:
    sys.exit(f'case must be one of {CASES}')
if case not in {'PH': PH_CASES, 'PS': PS_CASES, 'TH': TH_CASES, 'TS': TS_CASES, 'VP': VP_CASES, 'VT': VT_CASES, 'PT': CASES}[kind]:
    sys.exit(f'no {kind} export for {case}')
if missing_components(case):
    sys.exit(f'Missing in the database: {missing_components(case)}. Run scripts/build_db.py first.')

suffix = '' if kind == 'PT' else f'_{kind.lower()}'
out = Path(out_arg[0]) if out_arg else ROOT / 'docs' / 'figures' / f'{case}_phase_envelope{suffix}.png'
out.parent.mkdir(parents=True, exist_ok=True)

BLUE, RED, GREEN = '#1f77b4', '#d62728', '#2ca02c'
fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)
pe, r = envelope(case)

if kind == 'PT':
    Tp0 = points(case, 'pip')[0]
    T_pip = np.linspace(min(Tp0) - 5.0, max(Tp0) + 5.0, 161)
    ax.plot(r.T_bubble, r.P_bubble, '-', color=BLUE, lw=1.8, label='thermo-sim bubble')
    ax.plot(r.T_dew, r.P_dew, '-', color=RED, lw=1.8, label='thermo-sim dew')
    ax.plot(T_pip, pe.phase_identification_curve(T_pip), '-', color=GREEN, lw=1.4,
            label='thermo-sim phase identification')
    Tb, Pb = points(case, 'bubble_default_init' if 'bubble_default_init' in sections(case) else 'bubble_custom_init')
    Td, Pd = points(case, 'dew_default_init' if 'dew_default_init' in sections(case) else 'dew')
    Tp, Pp = points(case, 'pip')
    keep = Pp > 1.0
    ax.plot(Tb, Pb, 'o', mfc='none', color=BLUE, ms=3.5, mew=0.7, label='DWSIM bubble')
    ax.plot(Td, Pd, 'o', mfc='none', color=RED, ms=3.5, mew=0.7, label='DWSIM dew')
    ax.plot(Tp[keep], Pp[keep], 's', mfc='none', color=GREEN, ms=4, mew=0.7, label='DWSIM phase identification')
    Tc, Pc = r.critical_point
    ax.plot(Tc, Pc, '*', color='k', ms=11, label=f'critical point ({Tc:.2f} K, {Pc:.2f} bar)')
    ax.plot(*r.cricondentherm, '^', color='k', ms=7, label=f'cricondentherm ({r.cricondentherm[0]:.2f} K)')
    ax.plot(*r.cricondenbar, 'v', color='k', ms=7, label=f'cricondenbar ({r.cricondenbar[1]:.2f} bar)')
    op = config(case)['operating_point']
    ax.plot(op['T_K'], op['P_bar'], 'D', color='#ff7f0e', ms=6, label='operating point')
    ax.set_xlim(max(0.0, 0.9 * min(Tb.min(), Td.min())), 1.1 * max(Td.max(), Tp.max()))
    ax.set_ylim(0, 1.1 * max(Pd.max(), Pp.max()))
    ax.set_xlabel('Temperature [K]')
    ax.set_title(f"P-T phase envelope, {config(case)['title']} (PR)")
elif kind in ('VP', 'VT'):
    key = 'P' if kind == 'VP' else 'T'
    c = saturation_volume(case)
    ax.plot(c['bubble']['V'], c['bubble'][key], '-', color=BLUE, lw=1.8, label='thermo-sim bubble')
    ax.plot(c['dew']['V'], c['dew'][key], '-', color=RED, lw=1.8, label='thermo-sim dew')
    vb, xb = v_points(case, kind, 'bubble_custom_init')
    vd, xd = v_points(case, kind, 'dew')
    ax.plot(vb, xb, 'o', mfc='none', color=BLUE, ms=3.5, mew=0.7, label='DWSIM bubble')
    ax.plot(vd, xd, 'o', mfc='none', color=RED, ms=3.5, mew=0.7, label='DWSIM dew')
    ax.plot(c['bubble']['V'][-1], c['bubble'][key][-1], '*', color='k', ms=11,
            label=f"critical point ({c['bubble'][key][-1]:.2f} {'bar' if key == 'P' else 'K'})")
    ax.set_xscale('log')
    ax.set_xlabel('Molar volume [m3/kmol]')
    ax.set_title(f"{key}-V phase envelope, {config(case)['title']} (PR)")
elif kind in ('TH', 'TS'):
    unit, name = ('kJ/kg', 'Enthalpy') if kind == 'TH' else ('kJ/(kg K)', 'Entropy')
    Tb_, vb = _curve_t(case, kind, 'bubble')
    Td_, vd = _curve_t(case, kind, 'dew')
    ax.plot(vb, Tb_, '-', color=BLUE, lw=1.8, label='thermo-sim bubble')
    ax.plot(vd, Td_, '-', color=RED, lw=1.8, label='thermo-sim dew')
    xb, Tb = t_points(case, kind, bubble_section(case, kind))
    xd, Td = t_points(case, kind, 'dew')
    ax.plot(xb, Tb, 'o', mfc='none', color=BLUE, ms=3.5, mew=0.7, label='DWSIM bubble')
    ax.plot(xd, Td, 'o', mfc='none', color=RED, ms=3.5, mew=0.7, label='DWSIM dew')
    ax.plot(vb[-1], Tb_[-1], '*', color='k', ms=11, label=f'critical point ({Tb_[-1]:.2f} K)')
    k = int(np.argmax(Td_))
    ax.plot(vd[k], Td_[k], '^', color='k', ms=7, label=f'cricondentherm ({Td_[k]:.2f} K)')
    xo, To = t_points(case, kind, 'operating')
    ax.plot(xo, To, 'D', color='#ff7f0e', ms=6, label='operating point (DWSIM)')
    ax.set_ylim(0.9 * min(Td.min(), Tb.min()), 1.08 * max(Td.max(), Tb.max()))
    ax.set_xlabel(f'{name} [{unit}]')
    ax.set_title(f"T-{kind[1]} phase envelope, {config(case)['title']} (PR)")
else:
    ref = ph_points if kind == 'PH' else ps_points
    unit, name = ('kJ/kg', 'Enthalpy') if kind == 'PH' else ('kJ/(kg K)', 'Entropy')
    Pb_, _, vb = _curve(case, kind, 'bubble')
    Pd_, _, vd = _curve(case, kind, 'dew')
    ax.plot(vb, Pb_, '-', color=BLUE, lw=1.8, label='thermo-sim bubble')
    ax.plot(vd, Pd_, '-', color=RED, lw=1.8, label='thermo-sim dew')
    xb, Pb = ref(case, 'bubble_custom_init')
    xd, Pd = ref(case, 'dew')
    ax.plot(xb, Pb, 'o', mfc='none', color=BLUE, ms=3.5, mew=0.7, label='DWSIM bubble')
    ax.plot(xd, Pd, 'o', mfc='none', color=RED, ms=3.5, mew=0.7, label='DWSIM dew')
    ax.plot(vb[-1], Pb_[-1], '*', color='k', ms=11, label=f'critical point ({Pb_[-1]:.2f} bar)')
    xo, Po = ref(case, 'operating')
    ax.plot(xo, Po, 'D', color='#ff7f0e', ms=6, label='operating point (DWSIM)')
    ax.set_ylim(0, 1.1 * max(Pd.max(), Pb.max()))
    ax.set_xlabel(f'{name} [{unit}]')
    ax.set_title(f"P-{kind[1]} phase envelope, {config(case)['title']} (PR)")

ax.set_ylabel('Temperature [K]' if kind in ('TH', 'TS', 'VT') else 'Pressure [bar]')
ax.title.set_fontsize(9)
ax.grid(alpha=0.3)
ax.legend(fontsize=7, loc='best', framealpha=0.9)
fig.tight_layout()
fig.savefig(out)
print(f'saved {out}')