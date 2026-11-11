"""Access to the packaged property database."""
from __future__ import annotations

from importlib.resources import files

import pandas as pd


def load_database() -> pd.DataFrame:
    """Load the packaged property database (built by scripts/build_db.py)."""
    with (files("thermo_sim") / "data" / "properties_db.csv").open("rb") as f:
        return pd.read_csv(f, encoding="utf-8")
