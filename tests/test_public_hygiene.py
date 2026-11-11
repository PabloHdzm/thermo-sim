
"""Hygiene of the public repository: nothing from the private reference simulator,
and R_GAS only where it is a default."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src' / 'thermo_sim'

_FORBIDDEN = re.compile('as' + 'pen', re.I)    # built from pieces so this file is clean too
EXCLUDED_DIRS = {'.git', '.venv', 'venv', '__pycache__', '.pytest_cache', '.mypy_cache',
                 'build', 'dist', 'node_modules'}
SUFFIXES = {'.py', '.md', '.toml', '.txt', '.csv', '.yml', '.yaml', '.cfg', '.ini'}

# Temporary: shrinks to empty as these files are cleaned or moved to the private repo.
PENDING = ()


def _skip(p: Path) -> bool:
    parts = p.relative_to(ROOT).parts
    return any(part in EXCLUDED_DIRS or part.endswith('.egg-info') for part in parts)


def test_no_private_reference_mentions_in_public_files():
    bad = {}
    for p in sorted(ROOT.rglob('*')):
        if not p.is_file() or p.suffix.lower() not in SUFFIXES or _skip(p):
            continue
        rel = p.relative_to(ROOT).as_posix()
        if rel.startswith(PENDING):
            continue
        hits = [i for i, ln in enumerate(p.read_text(encoding='utf-8', errors='ignore')
                                         .splitlines(), 1) if _FORBIDDEN.search(ln)]
        if hits:
            bad[rel] = hits
    assert not bad, f"private-reference mentions (file: lines): {bad}"


def test_r_gas_is_only_referenced_where_it_is_a_default():
    allowed = {'constants.py', 'methods.py', 'eos.py', 'build_db.py'}
    files = [*SRC.rglob('*.py'), *(ROOT / 'scripts').rglob('*.py')]
    offenders = [p.relative_to(ROOT).as_posix() for p in sorted(files)
                 if p.name not in allowed and not _skip(p)
                 and re.search(r'\bR_GAS\b', p.read_text(encoding='utf-8', errors='ignore'))]
    assert not offenders, f"R_GAS referenced outside default files: {offenders}"