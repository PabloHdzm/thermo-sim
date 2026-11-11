"""ThermoMethod: which thermodynamics to use.

Layers: EOS form | alpha (global + per component) | critical constants |
liquid volume model | liquid phase model.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field, replace
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from .alpha import get_alpha
from .constants import R_GAS
from .eos import FORMS, CubicEOS
from .volume import VOLUME_MODELS, get_volume_model



# Optional path to a custom overrides CSV instead of the packaged one.
# Call reload_overrides() after changing it.
OVERRIDES_CSV_PATH: Path | None = None
LIQUID_MODEL = {'eos'}  # to be extended with 'nrtl', 'unifac', 'wilson'


@lru_cache(maxsize=1)
def _overrides() -> pd.DataFrame:
    if OVERRIDES_CSV_PATH is not None:
        df = pd.read_csv(OVERRIDES_CSV_PATH, encoding='utf-8')
    else:
        with (files('thermo_sim') / 'data' / 'critical_overrides.csv').open('rb') as f:
            df = pd.read_csv(f, encoding='utf-8')
    df['Component'] = df['Component'].str.lower()
    return df


def reload_overrides() -> None:
    _overrides.cache_clear()


@dataclass(frozen=True)
class ThermoMethod:
    eos: str = 'srk'
    alpha: str = 'std'
    alpha_overrides: Mapping[str, str] = field(default_factory=dict)               # {'hydrogen': 'gd'}
    alpha_params: Mapping[str, Mapping[str, float]] = field(default_factory=dict)  # {'kappa1': {'water': -0.06}}
    constants: str = 'db'                                                          # 'db' or a table name in the overrides CSV
    constants_user: Mapping[str, Mapping[str, float]] = field(default_factory=dict)  # {'hydrogen': {'Tc': ...}}
    liquid_volume: str = 'eos'
    liquid_model: str = 'eos'
    zra_default: str = 'db'      # 'db': user > database Z_RA > Zc ; 'zc': user > Zc (Zc = Pc*Vc/(R*Tc))
    R: float = R_GAS  # J/(mol K)
    name: str = field(default='', compare=False)  # label only, ignored in equality

    def __post_init__(self):
        if self.eos.lower() not in FORMS:
            raise ValueError(f"unknown eos '{self.eos}'. Options: {list(FORMS)}")
        if self.liquid_volume not in VOLUME_MODELS:
            raise ValueError(f"liquid_volume '{self.liquid_volume}': use one of {sorted(VOLUME_MODELS)}")
        if self.eos.lower() not in get_volume_model(self.liquid_volume).allowed_forms:
            raise ValueError(f"liquid_volume '{self.liquid_volume}' is not defined for form '{self.eos}'")
        for a in (self.alpha, *self.alpha_overrides.values()):
            get_alpha(a)
        if self.constants != 'db' and self.constants not in set(_overrides()['table']):
            raise ValueError(f"constants table '{self.constants}' does not exist. "
                             f"Options: ['db', {sorted(set(_overrides()['table']))}]")
        if self.liquid_model not in LIQUID_MODEL:
            raise ValueError(f"liquid_model '{self.liquid_model}': use one of {sorted(LIQUID_MODEL)}")
        if self.zra_default not in ('db', 'zc'):
            raise ValueError(f"zra_default '{self.zra_default}': use 'db' or 'zc'")
        if not 8.0 < self.R < 8.7:
            raise ValueError(f"R = {self.R}: expected J/(mol K), between 8.0 and 8.7")

    @property
    def label(self) -> str:
        return self.name or f"{self.eos.upper()}/{self.alpha}"

    def but(self, **kw) -> 'ThermoMethod':
        """Copy with changes. If `name` is not given, the label gets a '*' to mark a modified method."""
        kw.setdefault('name', self.name + '*' if self.name else '')
        return replace(self, **kw)


# Presets
METHODS: dict[str, ThermoMethod] = {k: replace(v, name=k) for k, v in {
    'STD-SRK':   ThermoMethod('srk', 'std'),
    'SRK72':     ThermoMethod('srk72', 'std'),
    'SRK-BM':    ThermoMethod('srk', 'bm'),
    'PR':        ThermoMethod('pr', 'std'),
    'PR-BM':     ThermoMethod('pr', 'bm'),
    'PR-MC':     ThermoMethod('pr', 'mc'),            # needs alpha_params {'c1': {...}} for ALL components
    'PRSV':      ThermoMethod('pr', 'prsv'),          # needs alpha_params {'kappa1': {...}} or a kappa1 column in the DB
    'IPRSV':     ThermoMethod('pr', 'iprsv'),
    'RK':        ThermoMethod('rk', 'rk'),
}.items()}


def get_method(m):
    if not isinstance(m, str):
        return m
    try:
        return METHODS[m.upper()]
    except KeyError:
        raise ValueError(f"unknown method '{m}'. Presets: {list(METHODS)}") from None


def _col(df, *names):
    for n in names:
        if n in df.columns:
            return df[n].to_numpy(float)
    raise KeyError(f"none of the columns {names} were found")


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({'Component': df['Component'].values,
                        'Tc': _col(df, 'Tc (K)', 'Tc'),
                        'Pc': _col(df, 'Pc (bar)', 'Pc'),
                        'omega': _col(df, 'w', 'omega'),
                        'frac_mol': _col(df, 'frac_mol')})
    vc = next((c for c in ('Vc (cm3/mol)', 'Vc (L/m)', 'Vc') if c in df.columns), None)  # cm3/mol
    out['Vc'] = df[vc].to_numpy(float) if vc else np.nan
    zdb = next((c for c in ('Z_RA', 'Zra') if c in df.columns), None)
    out['Zra_db'] = df[zdb].to_numpy(float) if zdb else np.nan   # from the database
    out['Zra'] = np.nan                                          # user override (constants_user)
    for c in ('kappa1', 'c1', 'c2', 'c3'):  # optional alpha parameters stored in the DB
        if c in df.columns:
            out[c] = df[c].to_numpy(float)
    return out.reset_index(drop=True)


def _alphas_per_component(method: ThermoMethod, names: list[str], strict: bool = True) -> list[str]:
    ov = {k.lower(): v.lower() for k, v in method.alpha_overrides.items()}
    unknown = set(ov) - set(names)
    if unknown and strict:
        raise ValueError(f"alpha_overrides for components not present: {sorted(unknown)}")
    return [ov.get(n, method.alpha.lower()) for n in names]


def _apply_constants(df, method, names, alphas, notes):
    layers = []
    if method.constants != 'db':
        t = _overrides()
        t = t[t['table'] == method.constants]
        layers.append((method.constants, {r.Component: r for r in t.itertuples()}))
    if method.constants_user:
        layers.append(('user', {k.lower(): v for k, v in method.constants_user.items()}))
    for label, mp in layers:
        for i, nm in enumerate(names):
            if nm not in mp:
                continue
            row = mp[nm]
            get = (lambda k: getattr(row, k, None)) if not isinstance(row, Mapping) else row.get
            fw = get('fitted_with')
            if isinstance(fw, str) and fw not in ('*', alphas[i]):
                warnings.warn(f"[{label}] constants for '{nm}' were fitted with alpha '{fw}', "
                              f"but alpha '{alphas[i]}' is in use", UserWarning, stacklevel=3)
            for key, col in (('Tc', 'Tc'), ('Pc', 'Pc'), ('omega', 'omega'), ('Vc', 'Vc'), ('Zra', 'Zra')):
                v = get(key)
                if v is not None and np.isfinite(float(v)):
                    notes.append(f"{nm}: {col} {df.at[i, col]:.6g} -> {float(v):.6g} [{label}]")
                    df.at[i, col] = float(v)


def _params_alpha(method, df, names, alphas):
    needed = set()
    for a in set(alphas):
        m = get_alpha(a)
        needed |= set(m.required) | set(m.optional)
    out = {}
    for k in needed:
        arr = df[k].to_numpy(float).copy() if k in df.columns else np.full(len(df), np.nan)
        for comp, v in method.alpha_params.get(k, {}).items():
            if comp.lower() in names:
                arr[names.index(comp.lower())] = v
        out[k] = arr
    return out


def _valid(z):
    return np.isfinite(z) & (z > 0.1) & (z < 1.0)

def _volume_data(d, R, zra_default='db'):
    """Vc [m3/mol], Zc, Zra and its source. Priority: user > db > Zc.
    An invalid value (outside (0.1, 1)) is skipped with a warning and the next source is used."""
    Vc = d['Vc'].to_numpy(float) * 1e-6
    Zc = d['Pc'].to_numpy(float) * 1e5 * Vc / (R * d['Tc'].to_numpy(float))
    levels = [('user', d['Zra'].to_numpy(float))]
    if zra_default == 'db':
        levels.append(('db', d['Zra_db'].to_numpy(float)))
    levels.append(('Zc', Zc))
    Zra = np.full(len(d), np.nan)
    source = np.full(len(d), 'no data', dtype=object)
    for label, z in levels:
        bad = np.isfinite(z) & ~_valid(z)
        if bad.any():
            warnings.warn(f"Zra ({label}) outside (0.1, 1) for {list(d['Component'][bad])}; "
                          "using the next source.", UserWarning)
        take = np.isnan(Zra) & _valid(z)
        Zra[take], source[take] = z[take], label
    return Vc, Zc, Zra, source


def build_eos(method: str | ThermoMethod, df: pd.DataFrame,
              kij: np.ndarray | None = None, *, strict: bool = True) -> CubicEOS:
    """Build the CubicEOS described by `method` for the components in `df`.

    strict=True:  alpha_overrides naming components that are not in df raise ValueError
                  (catches typos).
    strict=False: those overrides are ignored (the stream builder uses this because it
                  only works with the active components).
    """
    method = get_method(method)
    d = _normalize(df)
    names = [str(c).lower() for c in d['Component']]
    alphas = _alphas_per_component(method, names, strict)
    notes: list[str] = []
    _apply_constants(d, method, names, alphas, notes)
    eos = CubicEOS(method.eos, d, alphas, _params_alpha(method, d, names, alphas), kij, R=method.R)
    if method.R != R_GAS:
        notes.append(f"R = {method.R:.8g} J/(mol K) (default {R_GAS:.8g})")
    eos.Vc, eos.Zc, eos.Zra, eos.zra_source = _volume_data(d, eos.R, method.zra_default)
    eos.method, eos.notes, eos.label = method, notes, method.label
    return eos
