"""Print thermo-sim minus DWSIM for the natural-gas P-T phase envelope.

Run from the repository root:  python scripts/compare_dwsim_phase_envelope.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests'))

from _dwsim_phase_envelope import (branch_distances, envelope, feed, missing_components,  # noqa: E402
                                   pip_diffs, special_points)

if missing_components():
    sys.exit(f'Missing in the database: {missing_components()}. Run scripts/build_db.py first.')

pe, r = envelope()
print(r.report())
print('\nDistance to the DWSIM points, in (K, 100 ln P)')
for sec, br in (('bubble_custom_init', 'bubble'), ('bubble_default_init', 'bubble'), ('dew', 'dew')):
    T, P, d = branch_distances(sec, br)
    print(f'  {sec:20s} n = {len(d):3d}   max {d.max():.2e} at {T[d.argmax()]:.3f} K, {P[d.argmax()]:.4f} bar'
          f'   median {sorted(d)[len(d) // 2]:.2e}')
print('\nSpecial points (thermo-sim - DWSIM)')
for k, v in special_points().items():
    print(f'  {k:40s} {v:+.4e}')
print('\nPhase identification curve, relative difference')
for T, P, e in zip(*pip_diffs()):
    print(f'  {T:8.2f} K  {P:9.4f} bar  {e:+.2e}')
print(f'\nOperating point: vapor fraction {feed().V_frac:.4f}')
