"""Print thermo-sim - DWSIM differences over the DWSIM grids (max / median per quantity).

    python scripts/compare_dwsim_grid.py                 # both cases (pr, srk)
    python scripts/compare_dwsim_grid.py srk             # one case
    python scripts/compare_dwsim_grid.py srk diffs.csv   # also write every difference
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tests'))
from _dwsim_grid import GRIDS, energy_diffs, energy_rows, equilibrium_diffs, row_id  # noqa: E402


def run_case(case, out=None):
    recs = []
    for _, r in GRIDS[case].iterrows():
        d = {'id': row_id(r), 'T_K': r.T_K, 'P_bar': r.P_bar, 'n_phases': r.n_phases,
             'phi_digits': r.phi_digits}
        try:
            d.update(equilibrium_diffs(r, case))
        except Exception as e:                      # keep going, report at the end
            d['error'] = f'{type(e).__name__}: {e}'
        recs.append(d)
    eq = pd.DataFrame(recs).set_index('id')
    en = []
    for _, r in energy_rows(case).iterrows():
        d = {'id': row_id(r)}
        try:
            d.update(energy_diffs(r, case))
        except Exception as e:
            d['error'] = f'{type(e).__name__}: {e}'
        en.append(d)
    en = pd.DataFrame(en).set_index('id')
    allr = eq.join(en, rsuffix='_en')

    pd.set_option('display.width', 160)
    cols = [c for c in allr.columns if c.startswith(('d', 'rel_'))]
    print(f'\n=== {case}: {len(eq)} rows ({int((eq.n_phases == 2).sum())} two-phase), '
          f'{len(en)} with energy ===')
    summ = pd.DataFrame({c: [allr[c].abs().max(), allr[c].abs().median(), allr[c].abs().idxmax()]
                         for c in cols}, index=['max|diff|', 'median|diff|', 'worst row']).T
    print(summ.to_string())
    one = allr[allr.n_phases == 1]
    if 'dS_R' in one:
        print(f"dS_R single-phase rows: max |diff| = {one.dS_R.abs().max():.2e}")
    for col in ('error', 'error_en'):
        if col in allr and allr[col].notna().any():
            print(f'\nFAILED ROWS ({col}):')
            print(allr.loc[allr[col].notna(), col].to_string())
    if out:
        allr.to_csv(out)
        print('written', out)


if __name__ == '__main__':
    args = sys.argv[1:]
    cases = [args.pop(0)] if args and args[0] in GRIDS else list(GRIDS)
    for c in cases:
        run_case(c, args[0] if args else None)