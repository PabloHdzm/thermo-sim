"""Convert the DWSIM phase envelope text exports (tab separated sections) to
reference_data/dwsim/natgas_phase_envelope.csv.

Usage: python scripts/import_dwsim_phase_envelope.py DEFAULT_INIT.txt CUSTOM_INIT.txt

The first file is the export with DWSIM's default initialization (the bubble branch starts at the
lowest temperature DWSIM reaches), the second one with custom initialization (initial P = 1.01325 bar,
the same start as thermo-sim). Dew, critical, phase identification and operating points are the same in
both exports; they are taken from the second one.
"""
import csv
import sys
from pathlib import Path

SECTIONS = {'Bubble Points': 'bubble', 'Dew Points': 'dew', 'Critical Point': 'critical',
            'Phase Identification Parameter': 'pip', 'Operating Point': 'operating'}


def parse(path):
    out, cur = {}, None
    for ln in Path(path).read_text(encoding='utf-8-sig').splitlines():
        ln = ln.strip()
        if not ln:
            cur = None
            continue
        if ln in SECTIONS:
            cur = SECTIONS[ln]
            out[cur] = []
        elif ln.startswith('Temperature'):
            continue
        elif cur:
            T, P = ln.split()
            out[cur].append((float(T), float(P)))
    return out


def main(default_file, custom_file, dest):
    a, b = parse(default_file), parse(custom_file)
    rows = [('bubble_default_init', T, P) for T, P in a['bubble']]
    rows += [('bubble_custom_init', T, P) for T, P in b['bubble']]
    for sec in ('dew', 'critical', 'pip', 'operating'):
        rows += [(sec, T, P) for T, P in b[sec]]
    with open(dest, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['section', 'T_K', 'P_bar'])
        w.writerows(rows)
    print(f'{len(rows)} rows -> {dest}')


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    main(sys.argv[1], sys.argv[2], root / 'reference_data' / 'dwsim' / 'natgas_phase_envelope.csv')
