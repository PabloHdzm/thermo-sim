"""Liquid molar-volume models: EOS root, Peneloux, Rackett and DIPPR-105."""
from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import numpy as np


_NAN = np.nan


@dataclass(frozen=True)
class VolumeData:
    """Per-component data, aligned with the EOS. Tc, Pc, Vc and Zra already include the overrides."""
    MM: np.ndarray          # g/mol
    Tc: np.ndarray          # K
    Pc: np.ndarray          # Pa
    Vc: np.ndarray          # m3/mol (nan if missing)
    Zra: np.ndarray         # Rackett Z: Zc by default, or user override
    rho_A: np.ndarray       # refitted DIPPR-105, kg/m3: rho = A * B^-(1-Tr)^n
    rho_B: np.ndarray
    rho_n: np.ndarray

    @classmethod
    def from_stream(cls, eos, component_data) -> 'VolumeData':
        """component_data: per-component property table of the stream (MM, A_Rho_liq, ...)."""
        n = eos.n
        col = lambda c: component_data[c].to_numpy(float) if c in component_data.columns else np.full(n, np.nan)
        return cls(MM=component_data['MM'].to_numpy(float), Tc=np.asarray(eos.Tc, float),
                   Pc=np.asarray(eos.Pc, float) * 1e5, Vc=np.asarray(eos.Vc, float),
                   Zra=np.asarray(eos.Zra, float),
                   rho_A=col('A_Rho_liq'), rho_B=col('B_Rho_liq'), rho_n=col('n_Rho_liq'))


def _z_eos(eos, T, P_Pa, x, phase):
    am, bm = eos.mixture_params(T, x)
    A = am * P_Pa / (eos.R * T) ** 2
    B = bm * P_Pa / (eos.R * T)
    return eos.Z_roots(A, B)[0] if phase == 'v' else eos.liquid_Z_root(A, B)


class LiquidVolumeModel(ABC):
    name: ClassVar[str]
    allowed_forms: ClassVar[frozenset] = frozenset({'pr', 'srk', 'rk'})

    @abstractmethod
    def molar_volume(self, eos, d: VolumeData, T: float, P_Pa: float, x) -> float: ...


class EOSLiquid(LiquidVolumeModel):
    """Liquid root of the EOS (default)."""
    name = 'eos'

    def molar_volume(self, eos, d, T, P_Pa, x):
        return float(_z_eos(eos, T, P_Pa, np.asarray(x, float), 'l') * eos.R * T / P_Pa)


# c_i = a (R Tc/Pc)(b - Zra_i);  V = V_EOS - sum(x_i c_i)
PENELOUX = {'srk': (0.40768, 0.29441),     # Peneloux et al. 1982
            'pr':  (0.50033, 0.25969)}     # Jhaveri-Youngren 1988


class EOSPeneloux(EOSLiquid):
    """EOS liquid root with volume translation.

    Only the volume is shifted; H, S and fugacities are left untouched.
    """
    name = 'eos_peneloux'
    allowed_forms = frozenset(PENELOUX)

    def shift(self, eos, d, x):
        a, b = PENELOUX[eos.form.name]
        x = np.asarray(x, float)
        c = a * (eos.R * eos.Tc / (eos.Pc * 1e5)) * (b - d.Zra)
        missing = (x > 1e-12) & ~np.isfinite(c)
        if missing.any():
            warnings.warn("Peneloux: Zra not available for some component; using c = 0 for it.",
                          RuntimeWarning, stacklevel=3)
        return float(x @ np.where(np.isfinite(c), c, 0.0))

    def molar_volume(self, eos, d, T, P_Pa, x):
        return super().molar_volume(eos, d, T, P_Pa, x) - self.shift(eos, d, x)


class Rackett(LiquidVolumeModel):
    """Modified Rackett (Spencer-Danner) with mixing rules.

    The mixture Tr is capped at 1.
    """
    name = 'rackett'

    def molar_volume(self, eos, d, T, P_Pa, x):
        x = np.asarray(x, float)
        m = x > 1e-12
        if not m.any():
            return _NAN
        xs = x[m] / x[m].sum()
        zra, Vc, Tc, Pc = d.Zra[m], d.Vc[m], d.Tc[m], d.Pc[m]
        if not (np.all(np.isfinite(zra)) and np.all(np.isfinite(Vc))):
            return _NAN
        cb = Vc ** (1.0 / 3.0)
        # Vc mixing rule: kij = 1 - 8 sqrt(Vci Vcj) / (Vci^(1/3) + Vcj^(1/3))^3
        kij = 1.0 - 8.0 * np.sqrt(np.outer(Vc, Vc)) / (cb[:, None] + cb[None, :]) ** 3
        Vcm = xs @ Vc
        Tcm = (xs[:, None] * xs[None, :] * np.outer(Vc, Vc)
            * np.sqrt(np.outer(Tc, Tc)) * (1.0 - kij)).sum() / Vcm ** 2
        Pcm = Tcm / (xs @ (Tc / Pc))
        Zm = xs @ zra
        Trm = min(T / Tcm, 1.0)
        return float(eos.R * Tcm / Pcm * Zm ** (1.0 + (1.0 - Trm) ** (2.0 / 7.0)))


class PolyDIPPR(LiquidVolumeModel):
    """rho_i = A * B^-(1-Tr)^n for Tr < 1, with ideal mixing of volumes.

    A component with Tr >= 1 or without coefficients falls back to pure-component
    Rackett (Tr capped at 1).
    """
    name = 'poly'

    def molar_volume(self, eos, d, T, P_Pa, x):
        x = np.asarray(x, float)
        m = x > 1e-12
        if not m.any():
            return _NAN
        V = 0.0
        for i in np.where(m)[0]:
            Tr = T / d.Tc[i]
            with np.errstate(all='ignore'):
                rho = d.rho_A[i] * d.rho_B[i] ** (-(1.0 - Tr) ** d.rho_n[i])
            if np.isfinite(rho) and rho > 0 and Tr < 1:
                Vi = d.MM[i] * 1e-3 / rho
            else:
                e = np.zeros(len(x)); e[i] = 1.0
                Vi = _RACKETT.molar_volume(eos, d, T, P_Pa, e)
            if not np.isfinite(Vi):
                return _NAN
            V += x[i] / x[m].sum() * Vi
        return float(V)


_RACKETT = Rackett()
VOLUME_MODELS: dict[str, LiquidVolumeModel] = {m.name: m for m in (
    EOSLiquid(), EOSPeneloux(), _RACKETT, PolyDIPPR())}


def get_volume_model(name: str) -> LiquidVolumeModel:
    try:
        return VOLUME_MODELS[name.lower()]
    except KeyError:
        raise ValueError(f"unknown liquid_volume '{name}'. Options: {sorted(VOLUME_MODELS)}") from None


class PhaseVolume:
    """Volume calculator used by the Stream."""

    def __init__(self, eos, data: VolumeData, model: LiquidVolumeModel):
        self.eos, self.data, self.model = eos, data, model

    def gas(self, T, P_Pa, x) -> float:
        """Always the EOS (largest root), no volume translation."""
        return float(_z_eos(self.eos, T, P_Pa, np.asarray(x, float), 'v') * self.eos.R * T / P_Pa)

    def liquid_eos(self, T, P_Pa, x) -> float:
        return VOLUME_MODELS['eos'].molar_volume(self.eos, self.data, T, P_Pa, x)

    def liquid(self, T, P_Pa, x) -> tuple[float, str]:
        """Return (V [m3/mol], method used). If the model fails, falls back to the EOS root
        and the method is reported as '<model>->eos'."""
        V = self.model.molar_volume(self.eos, self.data, T, P_Pa, x)
        if np.isfinite(V) and V > 0:
            return V, self.model.name
        return self.liquid_eos(T, P_Pa, x), f"{self.model.name}->eos"


def build_phase_volume(method, eos, component_data) -> PhaseVolume:
    """method: ThermoMethod; eos: CubicEOS from build_eos (with Vc/Zra);
    component_data: per-component property table of the stream."""
    return PhaseVolume(eos, VolumeData.from_stream(eos, component_data), get_volume_model(method.liquid_volume))
