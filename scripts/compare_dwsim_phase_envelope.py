"""Print thermo-sim minus DWSIM for the P-T phase envelopes (natgas and n2co2 cases).

Run from the repository root:  python scripts/compare_dwsim_phase_envelope.py [case ...]
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests'))

from _dwsim_phase_envelope import (CASES, branch_distances, config, envelope, feed,  # noqa: E402
                                   missing_components, pip_diffs, sections, special_points)

for case in sys.argv[1:] or CASES:
    print('=' * 70)
    print(f"{case}: {config(case)['title']}")
    print('=' * 70)
    if missing_components(case):
        print(f'Missing in the database: {missing_components(case)}. Run scripts/build_db.py first.\n')
        continue
    pe, r = envelope(case)
    print(r.report())
    print('\nDistance to the DWSIM points, in (K, 100 ln P)')
    for sec, br in (('bubble_custom_init', 'bubble'), ('bubble_default_init', 'bubble'), ('dew', 'dew')):
        if sec not in sections(case):
            continue
        T, P, d = branch_distances(case, sec, br)
        print(f'  {sec:20s} n = {len(d):3d}   max {d.max():.2e} at {T[d.argmax()]:.3f} K, {P[d.argmax()]:.4f} bar'
              f'   median {sorted(d)[len(d) // 2]:.2e}')
    print('\nSpecial points (thermo-sim - DWSIM)')
    for k, v in special_points(case).items():
        print(f'  {k:50s} {v:+.4e}')
    print('\nPhase identification curve, relative difference')
    for T, P, e in zip(*pip_diffs(case)):
        print(f'  {T:8.2f} K  {P:9.4f} bar  {e:+.2e}')
    print(f'\nOperating point: vapor fraction {feed(case).V_frac:.4f}\n')