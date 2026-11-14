"""Print thermo-sim minus DWSIM for the P-T phase envelopes (natgas and n2co2 cases).

Run from the repository root:  python scripts/compare_dwsim_phase_envelope.py [case ...]
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests'))

from _dwsim_phase_envelope import (CASES, PH_CASES, PS_CASES, TH_CASES, TS_CASES, branch_distances,  # noqa: E402
                                   bubble_section, comparison_table, comparison_table_t, config, critical_enthalpy,
                                   critical_entropy, entropy_offset, envelope, feed, missing_components,
                                   operating_point_enthalpy, operating_point_th, ph_distances, pip_diffs,
                                   ps_distances, sections, special_points, th_distances, ts_distances, ts_offset)

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
    print(f'\nOperating point: vapor fraction {feed(case).V_frac:.4f}')
    if case in PH_CASES:
        print('\nPH diagram: distance to the DWSIM points in (kJ/kg, 100 ln P)')
        print('  raw = database Cp; corrected = without the integral of (Cp database - Cp DWSIM)')
        for sec, br in (('bubble_custom_init', 'bubble'), ('dew', 'dew')):
            for corr in (False, True):
                h, P, d = ph_distances(case, sec, br, corr)
                print(f"  {br:6s} {'corrected' if corr else 'raw      '} n = {len(d):3d}   max {d.max():.2e} at "
                      f'H = {h[d.argmax()]:.3f} kJ/kg, {P[d.argmax()]:.4f} bar   median {sorted(d)[len(d) // 2]:.2e}')
        raw, cor = operating_point_enthalpy(case)
        print(f'  operating point  thermo-sim - DWSIM: raw {raw:+.4f}   corrected {cor:+.2e} kJ/kg')
        raw, cor = critical_enthalpy(case)
        print(f'  critical point   thermo-sim - DWSIM: raw {raw:+.4f}   corrected {cor:+.2e} kJ/kg')
        print('\nPH diagram, point by point at DWSIM pressures (thermo-sim H interpolated in ln P)')
        print('  branch   P [bar]    T [K]   H DWSIM    H raw   raw - DWSIM   H corrected  corr - DWSIM  [kJ/kg]')
        for (b, P, T, dw, raw_, d_raw), (_, _, _, _, cor_, d_cor) in zip(comparison_table(case, 'PH'),
                                                                         comparison_table(case, 'PH', True)):
            print(f'  {b:6s} {P:9.4f} {T:8.3f} {dw:10.3f} {raw_:10.3f} {d_raw:+10.3f} {cor_:13.3f} {d_cor:+12.4f}')
    if case in PS_CASES:
        print('\nPS diagram: distance to the DWSIM points in (J/(kg K), 100 ln P), constant offset taken out')
        for corr in (False, True):
            print(f"  offset thermo-sim - DWSIM at the operating point ({'corrected' if corr else 'raw'}): "
                  f'{entropy_offset(case, corr):+.5f} kJ/(kg K)')
        for sec, br in (('bubble_custom_init', 'bubble'), ('dew', 'dew')):
            for corr in (False, True):
                s, P, d = ps_distances(case, sec, br, corr)
                print(f"  {br:6s} {'corrected' if corr else 'raw      '} n = {len(d):3d}   max {d.max():.2e} at "
                      f'S = {s[d.argmax()]:.5f} kJ/(kg K), {P[d.argmax()]:.4f} bar   median {sorted(d)[len(d) // 2]:.2e}')
        raw, cor = critical_entropy(case)
        print(f'  critical point   thermo-sim - DWSIM: raw {raw:+.3e}   corrected {cor:+.2e} kJ/(kg K)')
        print('\nPS diagram, point by point at DWSIM pressures (offset taken out)')
        print('  branch   P [bar]    T [K]   S DWSIM    S raw   raw - DWSIM   S corrected  corr - DWSIM  [kJ/(kg K)]')
        for (b, P, T, dw, raw_, d_raw), (_, _, _, _, cor_, d_cor) in zip(comparison_table(case, 'PS'),
                                                                         comparison_table(case, 'PS', True)):
            print(f'  {b:6s} {P:9.4f} {T:8.3f} {dw:10.5f} {raw_:10.5f} {d_raw:+10.5f} {cor_:13.5f} {d_cor:+12.6f}')
    if case in TH_CASES:
        print('\nTH diagram: distance to the DWSIM points in (kJ/kg, K)')
        for br in ('bubble', 'dew'):
            sec = bubble_section(case, 'TH') if br == 'bubble' else 'dew'
            for corr in (False, True):
                h, T, d = th_distances(case, sec, br, corr)
                print(f"  {br:6s} {'corrected' if corr else 'raw      '} n = {len(d):3d}   max {d.max():.2e} at "
                      f'H = {h[d.argmax()]:.3f} kJ/kg, {T[d.argmax()]:.3f} K   median {sorted(d)[len(d) // 2]:.2e}')
        raw, cor = operating_point_th(case)
        print(f'  operating point  thermo-sim - DWSIM: raw {raw:+.2e}   corrected {cor:+.2e} kJ/kg')
        print('\nTH diagram, point by point at DWSIM temperatures (dew: rising part, then falling part)')
        print('  branch    T [K]   H DWSIM    H raw   raw - DWSIM   H corrected  corr - DWSIM  [kJ/kg]')
        for (b, T, dw, raw_, d_raw), (_, _, _, cor_, d_cor) in zip(comparison_table_t(case, 'TH'),
                                                                    comparison_table_t(case, 'TH', True)):
            print(f'  {b:6s} {T:8.3f} {dw:10.3f} {raw_:10.3f} {d_raw:+10.3f} {cor_:13.3f} {d_cor:+12.4f}')
    elif case in CASES:
        print('\nTH diagram: no DWSIM export (DWSIM 10.2.12.0 fails this diagram for this mixture)')
    if case in TS_CASES:
        print('\nTS diagram: distance to the DWSIM points in (J/(kg K), K), constant offset taken out')
        for corr in (False, True):
            print(f"  offset thermo-sim - DWSIM at the operating point ({'corrected' if corr else 'raw'}): "
                  f'{ts_offset(case, corr):+.5f} kJ/(kg K)')
        for br in ('bubble', 'dew'):
            sec = bubble_section(case, 'TS') if br == 'bubble' else 'dew'
            for corr in (False, True):
                s_, T, d = ts_distances(case, sec, br, corr)
                print(f"  {br:6s} {'corrected' if corr else 'raw      '} n = {len(d):3d}   max {d.max():.2e} at "
                      f'S = {s_[d.argmax()]:.5f} kJ/(kg K), {T[d.argmax()]:.3f} K   median {sorted(d)[len(d) // 2]:.2e}')
        print('\nTS diagram, point by point at DWSIM temperatures (offset taken out)')
        print('  branch    T [K]   S DWSIM    S raw   raw - DWSIM   S corrected  corr - DWSIM  [kJ/(kg K)]')
        for (b, T, dw, raw_, d_raw), (_, _, _, cor_, d_cor) in zip(comparison_table_t(case, 'TS'),
                                                                    comparison_table_t(case, 'TS', True)):
            print(f'  {b:6s} {T:8.3f} {dw:10.5f} {raw_:10.5f} {d_raw:+10.5f} {cor_:13.5f} {d_cor:+12.6f}')
    print()