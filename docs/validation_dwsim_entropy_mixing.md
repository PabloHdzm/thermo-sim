# DWSIM entropy of mixtures: empirical mixing rule

Status: **empirical hypothesis**, not a verified fact about DWSIM. Last updated 2026-11-18.

## What is observed

Compared at the same state, the molar entropy of a mixture from thermo-sim and from DWSIM 10.2.12.0
(Peng-Robinson) differs by an amount that depends on composition and vanishes for pure components. Enthalpy
of mixtures and entropy of pure components agree. The difference is not a reference-state offset.

## The rule that reproduces it

thermo-sim uses the standard ideal mixing entropy:

    S_mix = -R * sum(x_i * ln x_i)                                   [kJ/(kmol K)]

The DWSIM output is reproduced by a mass-basis form in which the mole fraction x_i is used in the logarithm and
weighted by x_i / M_i:

    s_mix = -R * sum((x_i / M_i) * ln x_i)                           [kJ/(kg K)]
    S_mix = s_mix * MM_mix                                           [kJ/(kmol K)]

Expected difference of the total molar entropy of a stream, thermo-sim minus DWSIM:

    offset(x) = -R * sum(x_i ln x_i) + R * MM * sum((x_i / M_i) ln x_i)

with R = 8.314 kJ/(kmol K). Implemented in `tests/dwsim_conventions.py`, used only to compare against DWSIM.
It is never applied inside `thermo_sim`.

The rule was found by numerical fitting. It has not been checked against the DWSIM source code. It is a description
of DWSIM output, and says nothing about how DWSIM computes it. Another expression could coincide in the range tested.

## Conditions

All comparisons at 298.15 K and 1.01325 bar (Tref, Pref), Peng-Robinson, DWSIM critical constants and kij from
`reference_data/dwsim/natgas_phase_envelope.json`, with a database without formation terms (`hf_298_k = 0`,
`gf_298_k = 0`, `n_gas_el = 1`) so that S and H are relative to the ideal gas at the reference state. The natural
gas chiller case is at 70 bar and uses entropy differences, which cancel the reference terms.

## Evidence

Stored data: `reference_data/dwsim/mixing_entropy_298K.json`. Tests: `tests/test_dwsim_mixing_entropy.py` and
`tests/test_natgas_dew_point_control.py`. Offsets and residuals in kJ/(kmol K).

| Case | Role | DWSIM value | Residual (thermo-sim minus rule, minus DWSIM) |
|---|---|---|---|
| C1/C2 sweep, x(C1) = 0.1 to 0.9 (9 points; includes the 50/50 and 90/10 cases) | **fit**: used to choose the form | raw | max 0.00031 |
| 5 components C1-C5 (80/10/5/3/2) | validation | raw | +0.00034 |
| C1/C2 99/1 | validation | raw | +0.00002 |
| Natural gas chiller 320 K to 240 K, 70 bar: vapor, liquid and feed streams | validation | raw | 0.068 in dS (note 1) |

Enthalpy of the four mixtures (5 components, C1/C2 50/50, 90/10 and 99/1) agrees with DWSIM within 0.0021 kJ/kmol
(thermo-sim minus DWSIM between -0.0021 and -0.0010), with no mixing correction.

Note 1. The natural gas case is a change across a phase split, so the composition-linear terms cancel and only the
mixing term remains. The rule alone predicts -0.743 for dS(feed to chilled); thermo-sim minus DWSIM is -24.176 minus
(-23.502) = -0.675. The remaining 0.068 matches the ideal-gas Cp difference: the chiller duty is 0.29 % lower than
DWSIM, and 0.0029 x 24.2 = 0.07. The test passes with a tolerance of 0.15 and checks that the comparison fails
without the rule (difference above 0.45).

Reproduction of the 9-point sweep from the stored values, with thermo-sim minus DWSIM entropy ("measured") and the
rule offset, in kJ/(kmol K):

| x(C1) | S DWSIM | measured | rule | residual |
|---|---|---|---|---|
| 0.1 | 4.05476 | -1.46950 | -1.46964 | +0.00015 |
| 0.2 | 5.78628 | -1.73321 | -1.73343 | +0.00022 |
| 0.3 | 6.52831 | -1.54715 | -1.54742 | +0.00027 |
| 0.4 | 6.62988 | -1.12280 | -1.12310 | +0.00030 |
| 0.5 | 6.27053 | -0.58729 | -0.58760 | +0.00031 |
| 0.6 | 5.56212 | -0.03801 | -0.03832 | +0.00031 |
| 0.7 | 4.57885 | +0.43638 | +0.43609 | +0.00029 |
| 0.8 | 3.36478 | +0.73940 | +0.73917 | +0.00024 |
| 0.9 | 1.91850 | +0.73493 | +0.73478 | +0.00015 |

The residual is smooth in composition (largest near 0.5, always positive). That shape is worth a look: it is small
(0.0003, of the order of the rounding of the printed values and of the Cp polynomial), but it is not random.

Pure components at 298.15 K and 1.01325 bar, DWSIM against thermo-sim (H in kJ/kmol, S in kJ/(kmol K)):

| Component | H DWSIM | H thermo-sim | S DWSIM | S thermo-sim |
|---|---|---|---|---|
| Methane | -18.3868 | -18.38781 | -0.0429696 | -0.04297 |
| Ethane | -59.1906 | -59.19390 | -0.128215 | -0.12822 |
| Propane | -113.455 | -113.46156 | -0.24167 | -0.24168 |
| n-Butane | -186.725 | -186.73502 | -0.394702 | -0.39472 |
| n-Pentane | -26576.8 | -26578.28766 | -85.6508 | -85.65555 |

n-Pentane differs most (0.006 % in H, 0.006 % in S). It is liquid at this state, where the result depends on the
liquid Cp and the enthalpy of vaporization, so it is not evidence against the rule.

Cross-check: the `ps_offset_kJ_kgK = 0.0821` stored in `natgas_phase_envelope.json` equals the offset of the
5-component feed in mass basis, 1.7423 / 21.2325 = 0.08206. It was a fit to that mixture, and the rule gives it
without a free parameter.

## How this evidence is kept

- Each new DWSIM comparison with a mixture is checked with the rule before looking at the result, with the tolerance
  fixed in advance (kJ/(kmol K): 1e-3 for mixtures at 1 atm, 0.15 for a stream change with a phase split, from the
  Cp difference seen so far).
- Cases that fail are recorded in this table with their residual. The rule is not adjusted to make a case pass.
- A case counts as validation only if its raw DWSIM output is stored in `reference_data/dwsim/` and it was not used
  to choose the form of the rule.

## Cases that would test the rule hardest

The rule weights by 1 / M_i, and all cases so far are hydrocarbons with molar masses between 16 and 72.

1. A mixture with a large molar mass contrast (H2 with n-pentane, N2 with H2).
2. A ternary mixture that is not made of hydrocarbons (CO2, N2, CH4, or CO, H2, CO2).
3. A case with a liquid phase at several compositions, since the natural gas chiller is the only one so far.

## Wording for reports

Acceptable: "thermo-sim reproduces the DWSIM entropy of mixtures once a mass-basis ideal mixing term, found by
fitting, is used for DWSIM; N cases within the stated tolerance."

Not acceptable: "DWSIM computes the mixing entropy as ...", until the DWSIM source code confirms it.