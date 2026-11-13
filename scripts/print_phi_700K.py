"""Fugacity coefficients of H2, N2 and NH3 at 700 K and 200 bar: reactor inlet, reactor outlets
(fixed conversion and equilibrium) and pure components. For comparison with DWSIM's Master
Property Table (DWSIM prints the fugacity coefficient divided by 1e5 in some versions: check
the units of the column before comparing).

Run from the repository root:  python scripts/print_phi_700K.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests'))

from _dwsim_loop import SYNTHESIS, loop  # noqa: E402
from thermo_sim.unitops import EquilibriumReactor  # noqa: E402

warnings.filterwarnings('ignore', message=r'Zra \(user\)')

L = loop()
hot = L['streams']['hot']
eq = EquilibriumReactor(SYNTHESIS).run(hot).outlet
cases = {
    'inlet (3 H2 : 1 N2)': hot,
    'inlet + 1e-9 NH3 (infinite dilution)': hot.at(comp_mol={'hydrogen': 0.75, 'nitrogen': 0.25,
                                                             'ammonia': 1e-9}),
    'outlet, conversion 48.8008 % (rx_out)': L['streams']['rx_out'],
    'outlet, equilibrium (EquilibriumReactor)': eq,
    'pure hydrogen': hot.at(comp_mol={'hydrogen': 1.0}),
    'pure nitrogen': hot.at(comp_mol={'nitrogen': 1.0}),
    'pure ammonia': hot.at(comp_mol={'ammonia': 1.0}),
}
print(f"T = {hot.T:.2f} K, P = {hot.P:.2f} bar")
for title, s in cases.items():
    names = s.component_data['Component'].astype(str).str.lower().tolist()
    phi = np.atleast_1d(np.asarray(s.props.phi_v, float))
    print(f"\n{title}   (Z = {s.props.Z_gas:.6f}, phase {s.props.phase})")
    print(f"  {'component':10s} {'y':>10s} {'phi':>12s} {'ln phi':>12s}")
    for n, y, f in zip(names, np.atleast_1d(list(s.comp_mol.values())), phi):
        print(f"  {n:10s} {y:10.6f} {f:12.6f} {np.log(f):+12.6f}")
    if title.startswith('outlet'):
        lp = dict(zip(names, np.log(phi)))
        print(f"  sum(nu ln phi) = {sum(nu * lp[c] for c, nu in SYNTHESIS.items()):+.6f}")
