"""Plot the P-T phase envelope of thermo-sim over the DWSIM points (same layout as the DWSIM chart).

Run from the repository root:  python scripts/plot_dwsim_phase_envelope.py [output.png]
Writes docs/figures/natgas_phase_envelope.png by default (needs matplotlib).
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests'))

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from _dwsim_phase_envelope import CFG, envelope, missing_components, points  # noqa: E402

if missing_components():
    sys.exit(f'Missing in the database: {missing_components()}. Run scripts/build_db.py first.')

out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'docs' / 'figures' / 'natgas_phase_envelope.png'
out.parent.mkdir(parents=True, exist_ok=True)

pe, r = envelope()
T_pip = np.linspace(180.0, 340.0, 161)
P_pip = pe.phase_identification_curve(T_pip)

fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)
ax.plot(r.T_bubble, r.P_bubble, '-', color='#1f77b4', lw=1.8, label='thermo-sim bubble')
ax.plot(r.T_dew, r.P_dew, '-', color='#d62728', lw=1.8, label='thermo-sim dew')
ax.plot(T_pip, P_pip, '-', color='#2ca02c', lw=1.4, label='thermo-sim phase identification')

Tb, Pb = points('bubble_default_init')
Td, Pd = points('dew')
Tp, Pp = points('pip')
keep = Pp > 1.0
ax.plot(Tb, Pb, 'o', mfc='none', color='#1f77b4', ms=3.5, mew=0.7, label='DWSIM bubble')
ax.plot(Td, Pd, 'o', mfc='none', color='#d62728', ms=3.5, mew=0.7, label='DWSIM dew')
ax.plot(Tp[keep], Pp[keep], 's', mfc='none', color='#2ca02c', ms=4, mew=0.7, label='DWSIM phase identification')

Tc, Pc = r.critical_point
ax.plot(Tc, Pc, '*', color='k', ms=11, label=f'critical point ({Tc:.2f} K, {Pc:.2f} bar)')
ax.plot(*r.cricondentherm, '^', color='k', ms=7, label=f'cricondentherm ({r.cricondentherm[0]:.2f} K)')
ax.plot(*r.cricondenbar[::1], 'v', color='k', ms=7, label=f'cricondenbar ({r.cricondenbar[1]:.2f} bar)')
op = CFG['operating_point']
ax.plot(op['T_K'], op['P_bar'], 'D', color='#ff7f0e', ms=6, label='operating point')

ax.set_xlim(80, 340)
ax.set_ylim(0, 150)
ax.set_xlabel('Temperature [K]')
ax.set_ylabel('Pressure [bar]')
ax.set_title('P-T phase envelope, 0.80 CH4 / 0.10 C2H6 / 0.05 C3H8 / 0.03 C4H10 / 0.02 C5H12 (PR)')
ax.title.set_fontsize(9)
ax.grid(alpha=0.3)
ax.legend(fontsize=7, loc='upper left', framealpha=0.9)
fig.tight_layout()
fig.savefig(out)
print(f'saved {out}')
