"""Convert DWSIM phase envelope text exports (tab separated sections) to
reference_data/dwsim/<case>_phase_envelope.csv.

Usage:
    python scripts/import_dwsim_phase_envelope.py CASE EXPORT.txt [EXPORT_DEFAULT_INIT.txt]

EXPORT.txt is the export with custom initialization (initial P = 1.01325 bar, the same start as
`PhaseEnvelope.trace`). The optional second file is an export with DWSIM's default initialization (its
bubble branch starts at the lowest temperature DWSIM reaches); only its bubble points are used.
Dew, critical, phase identification and operating points are taken from the first file.
Sections written: bubble_custom_init, [bubble_default_init], dew, critical, pip, operating.
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


def main(case, custom_file, default_file, dest):
    b = parse(custom_file)
    rows = []
    if default_file:
        rows += [('bubble_default_init', T, P) for T, P in parse(default_file)['bubble']]
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
    case = sys.argv[1]
    main(case, sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None,
         root / 'reference_data' / 'dwsim' / f'{case}_phase_envelope.csv')