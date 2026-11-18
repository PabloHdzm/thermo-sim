"""Empirical description of a DWSIM convention that differs from the standard thermodynamic expression.

STATUS: EMPIRICAL HYPOTHESIS. The rule below was obtained by fitting DWSIM 10.2.12.0 outputs; it has NOT been
checked against the DWSIM source code. It reproduces the DWSIM entropy of mixtures in the cases listed in
docs/validation_dwsim_entropy_mixing.md. It is used only to compare thermo-sim with DWSIM in the tests: it is
never applied inside thermo_sim, whose mixing entropy -R sum(x ln x) is the standard one.

The rule
--------
Ideal mixing entropy, standard form (thermo-sim):

    S_mix = -R * sum(x_i * ln x_i)                                  [kJ/(kmol K)]

Form that reproduces DWSIM (mass basis, with mole fractions x_i in the logarithm weight):

    s_mix = -R * sum((x_i / M_i) * ln x_i)                          [kJ/(kg K)]
    S_mix = s_mix * MM_mix                                          [kJ/(kmol K)]

so the difference thermo-sim minus DWSIM of the total molar entropy of a stream is

    offset(x) = S_mix_standard(x) - S_mix_dwsim_rule(x)

It is zero for pure components and changes sign with composition.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

R_DWSIM = 8.314        # kJ/(kmol K), value stored with the DWSIM reference data


def _arrays(x: Mapping[str, float], molar_mass: Mapping[str, float]):
    names = list(x)
    xi = np.array([x[n] for n in names], dtype=float)
    M = np.array([molar_mass[n] for n in names], dtype=float)
    return xi, M


def mixing_entropy_standard(x: Mapping[str, float]) -> float:
    """-R sum(x ln x) [kJ/(kmol K)]."""
    xi = np.array(list(x.values()), dtype=float)
    xi = xi[xi > 0]
    return float(-R_DWSIM * np.sum(xi * np.log(xi)))


def mixing_entropy_dwsim_empirical(x: Mapping[str, float], molar_mass: Mapping[str, float]) -> float:
    """-R * MM * sum((x_i / M_i) ln x_i) [kJ/(kmol K)]. Empirical rule, see the module docstring."""
    xi, M = _arrays(x, molar_mass)
    m = xi > 0
    return float(-R_DWSIM * (xi @ M) * np.sum(xi[m] / M[m] * np.log(xi[m])))


def entropy_offset_empirical(x: Mapping[str, float], molar_mass: Mapping[str, float]) -> float:
    """Expected S(thermo-sim) - S(DWSIM) of a stream of composition x [kJ/(kmol K)], empirical."""
    return mixing_entropy_standard(x) - mixing_entropy_dwsim_empirical(x, molar_mass)


def split_entropy_offset_empirical(z: Mapping[str, float], y: Mapping[str, float], x: Mapping[str, float],
                                   V: float, molar_mass: Mapping[str, float]) -> float:
    """Expected [S_out - S_in](thermo-sim) - [S_out - S_in](DWSIM) when a stream of composition z becomes a
    two-phase mixture of vapor y (molar fraction V) and liquid x. The composition-linear terms cancel."""
    return (V * entropy_offset_empirical(y, molar_mass)
            + (1.0 - V) * entropy_offset_empirical(x, molar_mass)
            - entropy_offset_empirical(z, molar_mass))
