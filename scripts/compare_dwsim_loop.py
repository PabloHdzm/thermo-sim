"""Print thermo-sim minus DWSIM for the ammonia synthesis loop (units, streams, enthalpies).

Run from the repository root:  python scripts/compare_dwsim_loop.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests'))

from _dwsim_loop import (STREAMS, cp_ideal_diffs, enthalpy_diffs, loop, stream_diffs,  # noqa: E402
                         unit_diffs)

print('UNITS (thermo-sim - DWSIM)')
for k, v in unit_diffs().items():
    print(f'  {k:28s} {v:+.4e}')
print('\nSTREAMS (thermo-sim - DWSIM; rel_* are relative)')
for name in STREAMS.index:
    print(f'  {name}')
    for k, v in stream_diffs(name).items():
        print(f'    {k:12s} {v:+.3e}')
print('\nENTHALPY [kJ/kmol]')
for k, v in enthalpy_diffs().items():
    print(f'  {k:26s} {v:+.3f}')
from thermo_sim import load_database  # noqa: E402
for label, db in (('DWSIM Cp (fitted quartic, used in the loop)', None),
                  ('packaged database (CoolProp fit)', load_database())):
    print(f'\nIDEAL-GAS Cp of the mixture [kJ/(kmol K)] - {label}: database, DWSIM, relative')
    for k, (a, b, r) in cp_ideal_diffs(db).items():
        print(f'  {k:12s} {a:9.4f} {b:9.4f} {r:+.2e}')
L = loop()
print('\n' + L['comp'].report())
print(L['rx'].report())
print(L['drum'].report())