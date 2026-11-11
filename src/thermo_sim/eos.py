"""Cubic equations of state (PR, SRK, RK) with per-component alpha functions,
closed-form cubic roots and residual (departure) properties.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .alpha import get_alpha
from .constants import R_GAS


def cubic_real_roots(c2: float, c1: float, c0: float) -> np.ndarray:
    """Real roots (ascending) of Z^3 + c2 Z^2 + c1 Z + c0 = 0."""
    p = c1 - c2 * c2 / 3.0
    q = 2.0 * c2 ** 3 / 27.0 - c2 * c1 / 3.0 + c0
    sh = -c2 / 3.0
    D = 0.25 * q * q + p ** 3 / 27.0
    sc = 0.25 * q * q + abs(p) ** 3 / 27.0
    if sc == 0.0:
        r = np.array([sh])
    elif D > 1e-12 * sc:                              # one real root
        s = np.sqrt(D)
        r = np.array([np.cbrt(-0.5 * q + s) + np.cbrt(-0.5 * q - s) + sh])
    elif p >= 0.0:
        r = np.array([sh])
    else:                                             # three real roots (possibly repeated)
        m = 2.0 * np.sqrt(-p / 3.0)
        arg = np.clip(1.5 * q / p * np.sqrt(-3.0 / p), -1.0, 1.0)
        th = np.arccos(arg) / 3.0
        r = np.sort(m * np.cos(th - 2.0 * np.pi * np.arange(3) / 3.0) + sh)
    for _ in range(2):                                # Newton polish
        f = ((r + c2) * r + c1) * r + c0
        fp = (3.0 * r + 2.0 * c2) * r + c1
        ok = np.abs(fp) > 1e-14
        r = np.where(ok, r - np.where(ok, f / np.where(ok, fp, 1.0), 0.0), r)
    return r


def _cubic_fast(c2, c1, c0):
    """Same as cubic_real_roots but on plain Python floats (much less overhead). Returns a list."""
    p = c1 - c2 * c2 / 3.0
    q = 2.0 * c2 ** 3 / 27.0 - c2 * c1 / 3.0 + c0
    sh = -c2 / 3.0
    D = 0.25 * q * q + p ** 3 / 27.0
    sc = 0.25 * q * q + abs(p) ** 3 / 27.0
    if sc == 0.0:
        r = [sh]
    elif D > 1e-12 * sc:
        sq = math.sqrt(D)
        u, v = -0.5 * q + sq, -0.5 * q - sq
        r = [math.copysign(abs(u) ** (1 / 3), u) + math.copysign(abs(v) ** (1 / 3), v) + sh]
    elif p >= 0.0:
        r = [sh]
    else:
        m = 2.0 * math.sqrt(-p / 3.0)
        arg = max(-1.0, min(1.0, 1.5 * q / p * math.sqrt(-3.0 / p)))
        th = math.acos(arg) / 3.0
        r = sorted(m * math.cos(th - 2.0 * math.pi * k / 3.0) + sh for k in range(3))
    out = []
    for z in r:
        for _ in range(2):
            fp = (3.0 * z + 2.0 * c2) * z + c1
            if abs(fp) > 1e-14:
                z -= (((z + c2) * z + c1) * z + c0) / fp
        out.append(z)
    return out


def physical_roots(A, B, d1, d2):
    """Real roots Z > B of the cubic EOS (delta1, delta2 form), ascending.

    Single definition of the cubic coefficients and of the physical-root filter: used by
    CubicEOS and by the equilibrium kernel. If no root exists above B (not expected for
    physical A, B) it warns and returns an artificial root, so the problem is visible."""
    c2 = (d1 + d2 - 1.0) * B - 1.0
    c1 = A + (d1 * d2 - d1 - d2) * B * B - (d1 + d2) * B
    c0 = -(A * B + d1 * d2 * B * B * (1.0 + B))
    r = [z for z in _cubic_fast(c2, c1, c0) if z > B * (1.0 + 1e-12)]
    if not r:
        warnings.warn(f"EOS: no root Z > B (A={A:.3g}, B={B:.3g}); using artificial root B(1+1e-4)",
                      RuntimeWarning, stacklevel=3)
        return [B * (1.0 + 1e-4)]
    return r


@dataclass(frozen=True)
class EOSForm:
    """Constants of a cubic EOS form (no alpha)."""
    name: str
    delta1: float
    delta2: float
    Omega_a: float
    Omega_b: float
    kappa: Callable[[np.ndarray], np.ndarray]


FORMS: dict[str, EOSForm] = {
    'pr':  EOSForm('pr', 1 + np.sqrt(2), 1 - np.sqrt(2), 0.45724, 0.07780,
                   lambda w: 0.37464 + 1.54226 * w - 0.26992 * w**2),
    'srk': EOSForm('srk', 1.0, 0.0, 0.42748, 0.08664,
                   lambda w: 0.48508 + 1.55171 * w - 0.15613 * w**2),
    'rk':  EOSForm('rk', 1.0, 0.0, 0.42748025, 0.08664035,
                   lambda w: np.zeros_like(w)),
}


class ResidualPropertiesMixin:
    """Residual properties for any CubicEOS, using finite differences of a(T).

    For a general cubic EOS (delta1 != delta2):

        H_res  = R T (Z - 1) + (T da/dT - a) / ((d1 - d2) b) * ln((Z + d1 B) / (Z + d2 B))
        S_res  = R ln(Z - B) + da/dT / ((d1 - d2) b) * ln((Z + d1 B) / (Z + d2 B))
        Cp_res = Cv_res - T (dP/dT)_V^2 / (dP/dV)_T - R

    where a and b are the mixture parameters. For SRK/RK (delta2 = 0) the log
    argument reduces to (Z + B) / Z, which is fine as long as Z > 0. The factor
    (d1 - d2) is never zero: 2*sqrt(2) for PR, 1 for SRK/RK.
    """

    def _dam_dT_num(self, T: float, x: np.ndarray, h_rel: float = 1e-5) -> float:
        """da/dT by central differences."""
        h = T * h_rel
        am_fwd, _ = self.mixture_params(T + h, x)
        am_bwd, _ = self.mixture_params(T - h, x)
        return (am_fwd - am_bwd) / (2 * h)

    def _d2am_dT2_num(self, T: float, x: np.ndarray, h_rel: float = 1e-5) -> float:
        h = T * h_rel
        return (self._dam_dT_num(T + h, x) - self._dam_dT_num(T - h, x)) / (2 * h)

    def _phase_Z_base(self, T: float, P_bar: float, x: np.ndarray, phase: str):
        """Return (Z, B, am, bm) for the requested phase ('v' or 'l'). One cubic solve."""
        P_Pa = P_bar * 1e5
        am, bm = self.mixture_params(T, x)
        A = am * P_Pa / (self.R * T) ** 2
        B = bm * P_Pa / (self.R * T)
        if phase == 'v':
            Z = self.Z_roots(A, B)[0]
        else:
            Z = self.liquid_Z_root(A, B)
        return Z, B, am, bm

    # Pure formulas: they take the state (Z, B, am, bm, derivatives) and solve nothing, so that
    # residual_properties() computes the cubic root and each derivative exactly once.
    def _ln_term(self, Z, B):
        d1, d2 = self.delta1, self.delta2
        return np.log(max((Z + d1 * B) / max(Z + d2 * B, 1e-12), 1e-12))

    def _H_from(self, T, Z, B, am, bm, dam) -> float:
        d1, d2 = self.delta1, self.delta2
        return float(self.R * T * (Z - 1.0)
                     + (T * dam - am) / ((d1 - d2) * bm) * self._ln_term(Z, B))

    def _S_from(self, Z, B, bm, dam) -> float:
        d1, d2 = self.delta1, self.delta2
        return float(self.R * np.log(max(Z - B, 1e-12))
                     + dam / ((d1 - d2) * bm) * self._ln_term(Z, B))

    def _Cp_from(self, T, P_bar, Z, B, am, bm, dam, d2am) -> float:
        d1, d2 = self.delta1, self.delta2
        R = self.R
        P_Pa = P_bar * 1e5
        Cv_res = T * d2am / ((d1 - d2) * bm) * self._ln_term(Z, B)
        V = Z * R * T / P_Pa
        dP_dT_V = R / (V - bm) - dam / ((V + d1 * bm) * (V + d2 * bm))
        h = V * 1e-6
        def _P(Vi): return R*T/(Vi - bm) - am/((Vi + d1*bm)*(Vi + d2*bm))
        dP_dV_T = (_P(V + h) - _P(V - h)) / (2 * h)
        return float(Cv_res - T * dP_dT_V**2 / dP_dV_T - R)

    def H_residual(self, T: float, P_bar: float, x: np.ndarray, phase: str) -> float:
        x = np.asarray(x, float)
        Z, B, am, bm = self._phase_Z_base(T, P_bar, x, phase)
        return self._H_from(T, Z, B, am, bm, self._dam_dT_num(T, x))

    def S_residual(self, T: float, P_bar: float, x: np.ndarray, phase: str) -> float:
        x = np.asarray(x, float)
        Z, B, am, bm = self._phase_Z_base(T, P_bar, x, phase)
        return self._S_from(Z, B, bm, self._dam_dT_num(T, x))

    def Cp_residual(self, T: float, P_bar: float, x: np.ndarray, phase: str) -> float:
        x = np.asarray(x, float)
        Z, B, am, bm = self._phase_Z_base(T, P_bar, x, phase)
        dam = self._dam_dT_num(T, x)
        d2am = self._d2am_dT2_num(T, x)
        return self._Cp_from(T, P_bar, Z, B, am, bm, dam, d2am)

    def residual_properties(self, T: float, P_bar: float, x: np.ndarray, phase: str) -> dict:
        """H, S, Cp and G residuals with one cubic solve, one da/dT and one d2a/dT2."""
        x = np.asarray(x, float)
        Z, B, am, bm = self._phase_Z_base(T, P_bar, x, phase)
        dam = self._dam_dT_num(T, x)
        d2am = self._d2am_dT2_num(T, x)
        H = self._H_from(T, Z, B, am, bm, dam)
        S = self._S_from(Z, B, bm, dam)
        Cp = self._Cp_from(T, P_bar, Z, B, am, bm, dam, d2am)
        return {'H_res': H, 'S_res': S, 'Cp_res': Cp, 'G_res': H - T * S}


def mixing_aij(sqrt_ai: np.ndarray, kij: np.ndarray) -> np.ndarray:
    """Mixing rule a_ij = (1 - kij) s_i s_j, with s_i = sqrt(a_i) *keeping its sign*.

    Single definition, used by CubicEOS (residual properties) and by the equilibrium
    kernel. With s_i > 0 it is the classic rule sqrt(a_i a_j) (1 - kij).
    """
    return np.outer(sqrt_ai, sqrt_ai) * (1.0 - kij)


class CubicEOS(ResidualPropertiesMixin):
    """Cubic EOS: a form (PR / SRK / RK) plus one alpha model per component."""
    R: float = R_GAS  # J/(mol K), default; an instance may override it (see __init__)

    def __init__(self, form: EOSForm | str, df: pd.DataFrame, alphas: Sequence[str],
                 alpha_params: Mapping[str, np.ndarray] | None = None,
                 kij: np.ndarray | None = None, R: float | None = None):
        if R is not None:
            self.R = float(R)                  # before bi, which depends on R
        self.form = FORMS[form] if isinstance(form, str) else form
        f = self.form
        self.delta1, self.delta2 = f.delta1, f.delta2
        self.Omega_a, self.Omega_b = f.Omega_a, f.Omega_b

        self.n = len(df)
        self.names = [str(c).lower() for c in df['Component']] if 'Component' in df else None
        self.Tc = df['Tc'].to_numpy(float)
        self.Pc = df['Pc'].to_numpy(float)
        self.omega = df['omega'].to_numpy(float)
        self.zi = df['frac_mol'].to_numpy(float)
        self.kij = np.zeros((self.n, self.n)) if kij is None else np.asarray(kij, float)
        self.bi = self.Omega_b * self.R * self.Tc / (self.Pc * 1e5)
        self.kappa = f.kappa(self.omega)

        alphas = [a.lower() for a in alphas]
        if len(alphas) != self.n:
            raise ValueError("alphas must have one name per component")
        self.alpha_names = alphas
        self._groups = []
        for nm in dict.fromkeys(alphas):
            model = get_alpha(nm)
            if f.name not in model.allowed_forms:
                raise ValueError(f"alpha '{nm}' is not compatible with form '{f.name}' "
                                 f"(allowed: {sorted(model.allowed_forms)})")
            idx = np.array([i for i, a in enumerate(alphas) if a == nm])
            if model.only_components and self.names is not None:
                bad = [self.names[i] for i in idx if self.names[i] not in model.only_components]
                if bad:
                    raise ValueError(f"alpha '{nm}' only applies to {sorted(model.only_components)}; "
                                     f"it was assigned to {bad}")
            p = {}
            for k in model.required + model.optional:
                if alpha_params and k in alpha_params:
                    p[k] = np.asarray(alpha_params[k], float)[idx]
            for k in model.required:
                if k not in p or not np.all(np.isfinite(p[k])):
                    comp = [self.names[i] if self.names else i for i in idx]
                    raise ValueError(f"alpha '{nm}' requires '{k}' for {comp}")
            self._groups.append((model, idx, p))

    def _alpha(self, T: float) -> np.ndarray:
        Tr = np.maximum(T / self.Tc, 0.001)
        out = np.empty(self.n)
        for model, idx, p in self._groups:
            out[idx] = model.alpha(Tr[idx], self.omega[idx], self.kappa[idx], p)
        return out

    def _sqrt_alpha(self, T: float) -> np.ndarray:
        Tr = np.maximum(T / self.Tc, 0.001)
        out = np.empty(self.n)
        for model, idx, p in self._groups:
            out[idx] = model.sqrt_alpha(Tr[idx], self.omega[idx], self.kappa[idx], p)
        return out

    def ai(self, T: float) -> np.ndarray:
        return self.Omega_a * self._alpha(T) * (self.R * self.Tc) ** 2 / (self.Pc * 1e5)

    def sqrt_ai(self, T: float) -> np.ndarray:
        """Signed sqrt(a_i); sqrt_ai(T)**2 == ai(T). See AlphaModel.sqrt_alpha."""
        return np.sqrt(self.Omega_a * (self.R * self.Tc) ** 2 / (self.Pc * 1e5)) * self._sqrt_alpha(T)

    def aij(self, T: float) -> np.ndarray:
        return mixing_aij(self.sqrt_ai(T), self.kij)

    def mixture_params(self, T: float, x):
        x = np.asarray(x, float)
        am = float(x @ self.aij(T) @ x)
        bm = float(np.dot(x, self.bi))
        return am, bm

    def _valid_roots(self, A, B):
        return physical_roots(A, B, self.delta1, self.delta2)

    def Z_roots(self, A, B):
        """Return (Zmax, Zmin)."""
        r = self._valid_roots(A, B)
        return r[-1], r[0]

    def liquid_Z_root(self, A, B):
        return self._valid_roots(A, B)[0]


def _eos_label(eos) -> str:
    """Readable EOS name, used in diagnostics."""
    return getattr(eos, 'label', None) or type(eos).__name__