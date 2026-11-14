"""Convert DWSIM phase envelope text exports (tab separated sections) to
reference_data/dwsim/<case>_phase_envelope[_<kind>].csv.

Usage:
    python scripts/import_dwsim_phase_envelope.py CASE EXPORT.txt [EXPORT_DEFAULT_INIT.txt] [--kind PT|PH|PS|TH|TS|VP|VT] [--default-init]

EXPORT.txt is the export with custom initialization (initial P = 1.01325 bar, the same start as
`PhaseEnvelope.trace`). The optional second file is an export with DWSIM's default initialization (its
bubble branch starts at the lowest temperature DWSIM reaches); only its bubble points are used.
Dew, critical, phase identification and operating points are taken from the first file.
Sections written: bubble_custom_init, [bubble_default_init], dew, critical, pip, operating.

The CSV has the columns section, <first column>, P_bar. The first column is T_K for PT; for the other
kinds it is called `value` and the unit is the one DWSIM prints in the header (PH: kJ/kg), which is
echoed on screen. The second column is P_bar, except for TH and TS (the diagrams with T as ordinate),
where it is T_K.

--default-init: EXPORT.txt was run without custom initialization (e.g. DWSIM fails the TH diagram of some
mixtures with it); its bubble points are then written as bubble_default_init.
"""
import csv
import sys
from pathlib import Path

SECTIONS = {'Bubble Points': 'bubble', 'Dew Points': 'dew', 'Critical Point': 'critical',
            'Phase Identification Parameter': 'pip', 'Operating Point': 'operating'}


def parse(path):
    """({section: [(x, P), ...]}, header of the first column)."""
    out, cur, head = {}, None, None
    for ln in Path(path).read_text(encoding='utf-8-sig').splitlines():
        ln = ln.strip()
        if not ln:
            cur = None
            continue
        if ln in SECTIONS:
            cur = SECTIONS[ln]
            out[cur] = []
        elif cur and not ln[0].isdigit() and ln[0] not in '+-.':
            head = head or ln.split('\t')[0]
        elif cur:
            x, P = ln.split()
            out[cur].append((float(x), float(P)))
    return out, head


def main(case, custom_file, default_file, kind, dest, default_only=False):
    b, head = parse(custom_file)
    rows = []
    if default_file:
        rows += [('bubble_default_init', x, P) for x, P in parse(default_file)[0]['bubble']]
    rows += [('bubble_default_init' if default_only else 'bubble_custom_init', x, P) for x, P in b['bubble']]
    for sec in ('dew', 'critical', 'pip', 'operating'):
        rows += [(sec, x, P) for x, P in b.get(sec, [])]
    with open(dest, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['section', 'T_K' if kind == 'PT' else 'value', 'T_K' if kind in ('TH', 'TS') else 'P_bar'])
        w.writerows(rows)
    print(f'{len(rows)} rows -> {dest}   (first column: {head})')


if __name__ == '__main__':
    args = sys.argv[1:]
    kind = 'PT'
    default_only = '--default-init' in args
    if default_only:
        args.remove('--default-init')
    if '--kind' in args:
        i = args.index('--kind')
        kind = args[i + 1].upper()
        del args[i:i + 2]
    root = Path(__file__).resolve().parents[1]
    suffix = '' if kind == 'PT' else f'_{kind.lower()}'
    main(args[0], args[1], args[2] if len(args) > 2 else None, kind,
         root / 'reference_data' / 'dwsim' / f'{args[0]}_phase_envelope{suffix}.csv', default_only)