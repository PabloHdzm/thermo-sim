"""Print thermo-sim minus DWSIM for the propane refrigeration cycle (units, streams, enthalpies).

Run from the repository root:  python scripts/compare_dwsim_cycle.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests'))

from _dwsim_cycle import (STREAMS, UNITS, cop, cp_ideal_diffs, cycle, cycle_packaged,  # noqa: E402
                          enthalpy_diffs, stream_diffs, unit_diffs)
from thermo_sim import load_database  # noqa: E402

print('UNITS (thermo-sim - DWSIM)')
for k, v in unit_diffs().items():
    print(f'  {k:30s} {v:+.4e}')
print(f"\nCOP thermo-sim {cop():.4f}   DWSIM {UNITS['cop']['value']:.4f}")
print('\nSTREAMS (thermo-sim - DWSIM; rel_* are relative)')
for name in STREAMS.index:
    print(f'  {name}')
    for k, v in stream_diffs(name).items():
        print(f'    {k:10s} {v:+.3e}')
print('\nENTHALPY relative to stream 1 [kJ/kmol]')
for k, v in enthalpy_diffs().items():
    print(f'  {k:28s} {v:+.3f}')
for label, db in (('DWSIM Cp (fitted quartic, used in the cycle)', None),
                  ('packaged database (CoolProp fit)', load_database())):
    print(f'\nIDEAL-GAS Cp [kJ/(kmol K)] - {label}: database, DWSIM, relative')
    for k, (a, b, r) in cp_ideal_diffs(db).items():
        print(f'  {k:12s} {a:9.4f} {b:9.4f} {r:+.2e}')
L = cycle()
print('\n' + L['comp'].report())
print(L['valve'].report())
print('\nPackaged database, differences in duties and COP:')
for k, v in unit_diffs(cycle_packaged()).items():
    print(f'  {k:30s} {v:+.4e}')
