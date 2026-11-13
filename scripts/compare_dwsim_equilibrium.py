"""Conversions of the equilibrium reactor of thermo-sim vs DWSIM (see DWSIM_EQ in
tests/test_equilibrium_reactor.py).

Run from the repository root:  python scripts/compare_dwsim_equilibrium.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests'))

import warnings  # noqa: E402
warnings.filterwarnings('ignore', message=r'Zra \(user\)')

from test_equilibrium_reactor import DWSIM_EQ, _dwsim_case  # noqa: E402

print(f"{'T':>5} {'P':>6} {'mode':>9} {'base':>9} {'thermo-sim':>11} {'DWSIM':>9} {'diff':>9}")
for T, P, mode, ref, base, feed in DWSIM_EQ:
    got = _dwsim_case(T, P, mode, base, feed)
    print(f"{T:5.0f} {P:6g} {mode:>9} {base:>9} {got:11.6f} {ref:9.6f} {got - ref:+9.5f}")
