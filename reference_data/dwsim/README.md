# DWSIM reference data

Reference values extracted from the Master Property Table of DWSIM
10.2.12.0 using the Peng-Robinson property package. These values were
obtained from simulation runs performed by the author. No DWSIM code
is used or distributed in this repository. DWSIM is an independent
open-source project, and this repository is not affiliated with it.

## How the case was produced

- Components: hydrogen, nitrogen, ammonia (ordered by critical
  temperature).
- Feed: 0.726 / 0.242 / 0.032 mole fractions, 150 bar, vapor fraction
  0.99 (T solved).
- Equation of state: Peng-Robinson, using the critical constants,
  acentric factors and binary interaction parameters displayed by
  DWSIM for these components (see the JSON).
- DWSIM uses R = 8.314 J/(mol K); thermo-sim runs use
  `METHODS['PR'].but(R=8.314, ...)`.
- Liquid density options recorded: plain EOS, Rackett, COSTALD and
  EOS + Peneloux.

## Conventions

- DWSIM displays fugacity coefficients divided by 1e5.
- DWSIM's "Molar Enthalpy" excludes the enthalpy of formation.
  Therefore, the tests compare residual (departure) enthalpy, which is
  independent of the ideal-gas reference.
- The ideal-gas Cp values used by DWSIM for these components are
  recorded in the JSON, allowing the ideal contribution to be
  reconstructed independently of the thermo-sim database.

## Files

- `ammonia_loop_feed_150bar.json`: inputs, results and the list of
  quantities intentionally excluded from comparison.
- `tests/test_dwsim_ammonia_feed.py`: compares thermo-sim against this
  reference file.

# ammonia_feed_grid.csv

DWSIM 10.2.12.0, Peng-Robinson with DWSIM's own constants, kij and
R = 8.314. Feed H2/N2/NH3 = 0.726/0.242/0.032 (same as
`ammonia_loop_feed_150bar.json`).

The file contains 175 temperature-pressure points exported from
stream sensitivity analyses, varying one independent variable at a
time:

- `sensitivity_complete_v3`: T = 200-1000 K x P = 1-200 bar (143
  points, full precision).
- `sensitivity_refined_230-320K_v2`: T = 230-320 K step 10 at
  50/100/150/200 bar (32 additional points, exported with 3
  significant digits in the fugacity coefficients: `phi_digits = 3`).

Columns: fugacity coefficients are stored as true values (DWSIM
displays them divided by 1e5).

`n_phases = 2` only when the liquid fraction is above 1e-4.
Single-phase rows contain no physical liquid-phase values. In these
cases, DWSIM also reports hypothetical liquid Z and phi values (NaN
for T >= 500 K, with one isolated numerical anomaly at 300 K / 200
bar). These hypothetical values are excluded from the reference data.

Not used: the bubble temperature column (constant 69.58 K in the
export).

Enthalpy and entropy are compared as differences relative to the
1 bar row at the same temperature. This removes the contributions of
Cp, formation enthalpy and reference states, so only temperatures
with a corresponding 1 bar row are included in this comparison.

Independent validation: a plain numpy PR implementation reproduces Z,
phi and density for single-phase points (120 rows), as well as V and
x/y-derived phi and Z for two-phase points, to approximately 5e-6,
consistent with the exported print precision.

**Two-phase entropy notes:** At constant pressure, the mixture
entropy values in the two-phase region do not satisfy dH = T dS to
the same degree as the single-phase results. The difference is
approximately 1.8 R per mole of liquid, based on the 10 K rows at
100-200 bar. Single-phase points satisfy this relation to 1e-4 R. For
this reason, two-phase entropy values are excluded from the reference
comparisons.

# ammonia_feed_grid_srk.csv

Same feed, DWSIM 10.2.12.0, **SRK** package (Soave 1972:
kappa = 0.48 + 1.574 w - 0.176 w^2, Omega_a = 0.42748,
Omega_b = 0.08664), same critical constants and R = 8.314, all
kij = 0 (no interaction parameters are reported by DWSIM for this
package), and liquid density calculated from the EOS (no Peneloux).

The file contains 163 points (T = 200-1000 K x P = 1-200 bar, plus
the 230-320 K block at 50-200 bar). Of these, 32 are two-phase
points. All fugacity coefficients are stored at full precision
(`phi_digits = 6`).

Columns and filtering rules are the same as for the PR grid.

An independent numpy SRK implementation using the Soave-1972 kappa
reproduces the reference values with the following maximum
differences:

- V: 5e-7
- Z: 4.5e-6
- Z liquid: 2.5e-6
- phi vapor: 1e-6
- phi liquid: 2.7e-6
- Two-phase mixture density from EOS volumes: 3e-6

The thermo-sim default `srk` form (Graboski-Daubert kappa) produces
differences of up to 0.3 % in Z and 1.8 % in phi relative to this
reference. The `srk72` form is provided to reproduce the Soave-1972
formulation used in DWSIM.

# ammonia_feed_grid_prsv2.csv

Same feed, DWSIM 10.2.12.0, **PRSV2** package, same critical
constants and R = 8.314, all kij = 0 (no interaction parameters are
reported), and liquid density calculated from the EOS.

The file contains 135 points (T = 200-1000 K, P = 1-200 bar, plus the
230-320 K block).

DWSIM displays the following PRSV2 parameters:

| | kappa1 | kappa2 | kappa3 |
|---|---:|---:|---:|
| H2 | 0 | 0 | 0 |
| N2 | 0.02 | 0.3162 | 0.5350 |
| NH3 | 0.001 | -0.1265 | 0.51 |

The following implementation details reproduce the DWSIM reference
values, as verified with an independent numpy implementation and
confirmed in the DWSIM 10.2.12.0 PRSV2 source:

- **PRSV2 alpha formulation:** kappa = kappa0 + [k1 + k2 (k3 - Tr)
  (1 - sqrt(Tr))](1 + sqrt(Tr))(0.7 - Tr), using the Stryjek-Vera
  kappa0 polynomial (cubic coefficient 0.0196544 in DWSIM). The
  correction is applied over the entire Tr range. In this grid, N2
  has Tr > 1 throughout; omitting the correction above Tr = 1
  produces differences of approximately 3 % in Z.
- **Selection of the alpha formulation:** DWSIM applies the PRSV2
  expression only when kappa1 * kappa2 * kappa3 is nonzero.
  Otherwise, it uses the standard Peng-Robinson alpha,
  kappa = 0.37464 + 1.5422 w - 0.26992 w^2 (w <= 0.491). Since H2 has
  all three parameters set to zero, it uses the PR alpha
  (kappa = 0.02893) rather than the PRSV kappa0 value (0.04893).
  thermo-sim reproduces this behavior through
  `alpha_overrides={'hydrogen': 'std'}`.
- **Nitrogen kappa1 value:** The value used for N2 is 0.01996 (the
  Stryjek-Vera value), while DWSIM displays it rounded to 0.02. Pure
  N2 runs cannot distinguish these values at the displayed precision,
  but when the correction is omitted altogether
  (kappa1 = kappa2 = kappa3 = 0), the resulting Z values differ by
  1e-5 to 4e-4.
- **Mixing rule:** a_ij = sqrt(a_i a_j) (1 - x_i kij - x_j kij2)
  (Margules form). With kij = kij2 = 0, this reduces to the standard
  quadratic mixing rule.

## Pure-component validation

`ammonia_pure_prsv2.csv` contains additional pure-component
comparisons:

- H2 at 100, 200, 300 K and 50, 150 bar (PR alpha with 1.5422) agrees
  with DWSIM to 5e-6 in Z and 1e-5 in phi.
- N2 vapor at 90, 100, 110 K, evaluated at the DWSIM saturation
  pressures using PRSV2, agrees to 2e-6.
- The N2 liquid root at the same states agrees to 3e-6 in Z, 2e-6 in
  phi and 2e-7 in density.

At these points, DWSIM's own phi_l/phi_v values are 1.00085, 1.00265
and 1.00382, respectively, and thermo-sim produces the same values.
The corresponding pressures are 0.1 to 0.6 % below the EOS saturation
pressure. The fugacity coefficients nevertheless agree because both
phases are evaluated at the same T and P.

Using the coefficient 1.54226 (with more digits) produces larger
differences. Using kappa0 alone, without the alpha override, results
in differences of 0.1 to 0.8 % for H2.

## PRSV2 grid classification

Rows are classified in the `status` column. Only rows marked
`reproduced` (95, all at T <= 400 K) are included in the tests:

| status | rows | meaning |
|---|---:|---|
| reproduced | 95 | Independent numpy PRSV2 flash agrees within 2e-4 (most within 1e-5) |
| unstable_single_phase | 12 | DWSIM reports one vapor phase, while a dense N2 phase has a negative tangent-plane distance (350 K at 50 and 75 bar; 500 to 800 K) |
| dense_root_mismatch | 14 | T >= 500 K, Z of 0.02 to 0.1: agreement within 2e-4 to 6e-4, with sensitivity to the 4-digit kappa values |
| dwsim_liquid_only | 7 | DWSIM reports a single liquid phase at conditions where a vapor root also exists |
| two_phase_not_reproduced | 5 | DWSIM two-phase solutions not found by the independent flash calculation |
| nonfinite | 2 | Fugacity coefficients of 1e288 or infinity |

## PRSV2 energy calculation notes

**Energies are not compared for PRSV2.**

DWSIM's enthalpy and entropy values agree, to 1e-6 in dH/(R T), with
a calculation in which the temperature derivative of alpha is
evaluated with kappa frozen (kappa treated as constant). This
behavior is confirmed in the DWSIM source, where the calculation of
d(a)/dT omits dkappa/dT.

A full derivative that includes dkappa/dT produces differences of 0.3
to 0.4 R T in enthalpy and 0.3 to 0.4 R in entropy at 75 to 200 bar.

These results reflect two different derivative conventions for the
PRSV2 alpha function. thermo-sim differentiates the complete alpha
expression, including the temperature dependence of kappa, so its
energy calculations follow the full derivative rather than the
frozen-kappa convention. Consequently, the two sets of energy results
are not expected to coincide.

`tests/test_isobaric_consistency.py` verifies that thermo-sim's PRSV2
energy calculations are consistent with its own equation of state.

## PRSV2 behavior outside the correlation range

The behavior observed above 500 K is closely related to extrapolating
the PRSV2 correlation beyond its intended Tr < 1 range.

For N2, with kappa2 > 0 and kappa3 below Tr, the correction term
grows approximately as Tr^2. At 1000 K, kappa reaches approximately
-120, leading to extremely large attractive parameter values a(T).
This produces results such as Z = 0.0002 and fugacity coefficients of
1e288.

At 350 and 400 K, the model also predicts an N2-rich liquid phase
(vapor fraction 0.758 = 1 - y(N2)). The independent flash reproduces
these points (`reproduced`, 400 K at 75 to 200 bar), indicating that
they are valid solutions of the specified equation of state.

At 500 K and above, the feed is unstable as a single vapor phase
according to the tangent-plane distance criterion. The dense N2 phase
has tangent-plane distances ranging from -100 at 500 K to -457 at
600 K, in units of RT. The DWSIM results for these conditions contain
a single vapor phase, while a stability analysis such as the one
implemented in thermo-sim identifies a two-phase solution.

These results illustrate the sensitivity of PRSV2 extrapolation
outside its intended correlation range and the importance of
phase-stability analysis when evaluating these conditions.

# Ammonia synthesis loop (unit operations)

`ammonia_loop_streams.csv`, `ammonia_loop_units.json` and
`ammonia_loop_cp.json` contain the results of the DWSIM 10.2.12.0
"Ammonia Synthesis" flowsheet used to validate the unit operations.

Flowsheet:

Feed (300 K, 30 bar, 4 mol/s, H2 0.75 / N2 0.25) -> compressor
(200 bar, adiabatic efficiency 0.75) -> heater (700 K) -> isothermal
equilibrium reactor (reaction set ln K = 11000/T - 25, N2 conversion 
48.8758 %) -> cooler (250 K) -> cooler (250 K) -> flash drum.

Property package: Peng-Robinson with the grid kij and constants,
R = 8.314, inside-out flash, liquid volume from the EOS (no
Peneloux).

Tests: `tests/test_dwsim_loop.py` (helper: `tests/_dwsim_loop.py`).

Script containing all comparison differences:
`python scripts/compare_dwsim_loop.py`.

## What is compared and what is not

- Stream T, P, flows, phase compositions, Z and density, stream by
  stream.
- Duties and compressor power as relative differences. The reactor is
  (`EquilibriumReactor`) with P0 = 1.01325 bar and ln K evaluated at 
  T + 1 K (DWSIM conventions, see the equilibrium reactor section); 
  a StoichReactor at DWSIM's converged conversion is also checked. 
  The reference is the converged DWSIM export (R-1 tolerances 1e-9).
- Entropy is excluded from the comparison because the two-phase
  entropy results differ from the enthalpy-based thermodynamic
  consistency relation.

## Enthalpy reference

DWSIM's displayed "Molar Enthalpy" does **not** include the formation
enthalpy, whereas its "Energy Flow" (used in the unit balances) does.

thermo-sim's `H_total_mix` includes the formation enthalpy, so the
appropriate comparison is against Energy Flow divided by molar flow
(`energy_flow_per_kmol_kJ`).

## Ideal-gas Cp

The loop tests replace the ideal-gas Cp of H2, N2 and NH3 in the
database with the equation used by DWSIM:

`Cp = A + exp(B/T + C + D*T + E*T^2)`

Cp is expressed in J/(kmol K), with coefficients stored in
`ammonia_loop_cp.json`. These coefficients were fitted with a quartic
over 240-720 K, with the higher-order columns set to zero.

With this Cp formulation, the differences relative to DWSIM are:

| Quantity | Difference |
|---|---|
| Compressor T_out | 0.019 K |
| Compressor power, reactor, cooler duty | below 1e-4 relative |
| Heater duty | 2.4e-4 relative |
| Z, density, composition | 5e-6 (flash phases 2e-5) |

The packaged database fits ideal-gas Cp to CoolProp, whose values
differ from the DWSIM equation by 0.3 to 0.5 %. Using the packaged
database, duties differ by 0.15 to 0.9 % and the compressor outlet
temperature differs by +0.56 K.

The equilibrium results (Z, composition and vapor fraction) remain
the same because they do not depend on ideal-gas Cp.

`tests/test_dwsim_loop.py` records this as a known difference between
the two Cp sources, with acceptance tolerances of 1.5 % for duties
and 1 K for T_out. This preserves regression coverage for the energy
balances while accounting for the different ideal-gas Cp
correlations.

## Equilibrium reactor

`thermo_sim.unitops.EquilibriumReactor` solves one reaction in a single vapor phase:

    prod_i (phi_i y_i P / P0)^nu_i = K(T),    P0 = 1 bar

- `K(T)` comes from the formation data of the database (`hf_298_k`, `gf_298_k`, tabulated for an
  ideal gas at 1 bar) and the ideal-gas Cp polynomials: `dH0(T) = dH0(298.15) + int dCp dT`,
  `dS0(T) = (dH0 - dG0)(298.15) / 298.15 + int dCp / T dT`, `dG0 = dH0 - T dS0`, `ln K = -dG0 / (R T)`.
  It uses the same Cp (and the same linear extrapolation outside the fitted range) as `Stream`, so
  `K(T)` is consistent with the stream enthalpy and entropy. `reaction_thermo()` returns these values.
- `phi_i` are the EOS fugacity coefficients of the outlet composition (successive substitution).
- Options: `lnK=` (a function of T replaces the computed `K`), `P_ref=` (standard-state pressure
  in bar), `fugacity=False` (partial pressures, `phi = 1`), `T_out=`, `dP=`.
- Limits: one reaction, one vapor phase (an error is raised otherwise).

### Tests

`tests/test_equilibrium_reactor.py`:

- `K(T)`: `dG0(298.15)` equals the tabulated value and `d ln K / dT = dH0 / (R T^2)` from 400 to 900 K.
- The solution satisfies the equilibrium condition to 1e-8; the reaction runs backwards from the
  products or from pure NH3 to the same state; the conversion rises with P and falls with T.
- DWSIM 10.2.12.0 equilibrium reactor, ammonia synthesis, 3 H2 : 1 N2 feed, 500 K / 1 bar and
  600, 700, 800 K / 200 bar, in two DWSIM modes (Gibbs energy of formation and the equation
  `ln K = 11000 / T - 25`). Reference values: `equilibrium_reactor.csv`.

### Agreement with DWSIM

DWSIM must be run with its internal and external tolerances at 1e-9. With the defaults
(1e-3 and 1e-2) it stops 0.1 to 0.2 conversion points early and the result depends on the starting
point.

Two conventions of the DWSIM equilibrium reactor explain the difference with thermo-sim, found
with the ammonia synthesis and confirmed with the water-gas shift:

- **Standard-state pressure P0 = 1 atm.** With a constant `ln K`, `ln Q - ln K` (Q on a 1 bar basis)
  is -0.02632, -0.02630 and -0.02634 for `ln K` = -9.2857, -5 and -12, and `2 ln(1.01325)` is
  0.02633. With P0 = 1.01325 bar the residual is 3e-5 or less, so the reaction quotient, the mass
  balance, the EOS and the solver of DWSIM agree with thermo-sim.
- **The expression `ln K(T)` is evaluated at T + 1 K.** The temperature T' for which
  `11000 / T' - 25` reproduces the `ln K` that DWSIM used is 501.0000, 600.9993, 701.0006 and
  801.0004 K at 500, 600, 700 and 800 K. On the water-gas shift at 600 K (`4577.8 / T - 4.33`, sum of
  nu = 0, 1 atm) DWSIM gives 83.7935 %, against 83.887 % with K at T and 83.801 % with K at T + 1.
  The expression `ln K` without T (a constant) is not affected. This looks like a DWSIM
  implementation detail, not a thermodynamic effect: with these two conventions the ammonia
  conversions agree to 0.01 conversion points or better (equation mode, 500 to 800 K).

In Gibbs mode (K from the formation data) DWSIM does not show the T + 1 shift. With P0 = 1 atm,
thermo-sim is within +0.12, +0.06, +0.005 and -0.045 conversion points at 500, 600, 700 and
800 K; the remaining difference is not identified.

Standard state of the formation data: the formation Gibbs energy of NH3 in the database
(-16.37 kJ/mol, from NIST) is for 1 bar; DWSIM's (-16.40 kJ/mol, DIPPR) is for 1 atm. They are the same
data (the formation entropy differs by R ln 1.01325). `EquilibriumReactor` uses P0 = 1 bar by
default, consistent with the database; compare with DWSIM using `P_ref=1.01325`.

Carbon monoxide is not in the packaged database, so the water-gas shift is not part of the tests.

## Propane refrigeration cycle (valve, condenser, evaporator)

`propane_cycle_streams.csv` and `propane_cycle_units.json` hold the results of the DWSIM 10.2.12.0
propane refrigeration flowsheet used to validate `Valve` and the compressor/cooler pair on a pure
component that crosses the saturation dome: 1_evap_out (saturated vapor, 2.45 bar, 1 mol/s) ->
compressor C-1 (12.5 bar, adiabatic efficiency 0.8) -> condenser CD-1 (305 K) -> isenthalpic valve
V-1 (2.45 bar) -> evaporator EV-1 (253 K) -> 5_evap_out. Property package: Peng-Robinson with the
DWSIM propane constants (Tc 369.83 K, Pc 42.48 bar, omega 0.152, Vc 0.2 m3/kmol, ZRA 0.2763) and
R = 8.314. DWSIM results: W = 4.20275 kW, Q_cond = 15.9372 kW, Q_evap = 11.7348 kW, COP = 2.79.

Tests: `tests/test_dwsim_cycle.py` (helper: `tests/_dwsim_cycle.py`). Script with all the
differences: `python scripts/compare_dwsim_cycle.py`. The streams file is regenerated from a DWSIM
export with `python scripts/import_dwsim_cycle_streams.py <export.csv>`.

- The ideal-gas Cp is DWSIM's own equation, A + exp(B/T + C + D T + E T^2), fitted with a quartic
  over 230 to 340 K (other Cp columns zeroed). With the packaged CoolProp-based Cp the duties
  differ by the Cp difference only.
- Absolute enthalpies are compared as differences with respect to stream 1: thermo-sim and DWSIM use
  slightly different formation enthalpies of propane.
- `Valve`: for a pure component the PH flash cannot solve T inside the dome (T is fixed at the
  saturation temperature and H(T) jumps), so the valve solves the vapor fraction with the VF-P flash.
  For mixtures it uses the PH flash directly.

### Known difference: where DWSIM puts the saturation point of a pure component

With DWSIM's own constants, Peng-Robinson gives Z(liquid, 305 K, 12.5 bar) = 0.0436526 and
Z(vapor, 252.995 K, 2.45 bar) = 0.934560, identical to DWSIM (thermo-sim agrees to 1e-6 on stream 3).
The saturation temperature at 2.45 bar from the EOS equal-fugacity condition is 253.2214 K (thermo-sim
253.221 K, and an independent solution of the cubic). DWSIM reports 252.995 K, because for a pure
compound it places the saturation points with the vapor-pressure correlation of the compound,
`exp(A + B/T + C ln T + D T^E)` (propane: A 55.2725, B -3396.95, C -5.42339, D 8.95373e-6, E 2, Pa):

| T [K] | correlation [bar] | DWSIM | EOS [bar] |
|---|---|---|---|
| 252.995 | 2.45004 | 2.45 | 2.4302 |
| 305 | 11.37562 | 11.3756 | 11.3023 |
| 325.138 | 17.96016 | 17.9601 | 17.9583 |

and its Tsat(12.5 bar) = 308.9471 K (correlation) against 309.1735 K (EOS). The same reason explains
why liquid N2 forced with a vapor fraction did not match. Consequences:

- Streams on the dome (1, 2, 4, 5) inherit the 0.226 K shift; those cases are marked xfail.
- The evaporator is taken to saturated vapor (`Cooler(V_frac=1.0)`); DWSIM heats to 253 K, 0.005 K
  above its own dew point, which is liquid in the EOS.
- The inlet offset cancels in the compressor temperature rise (72.141 K against 72.143 K), and compressor
  power, condenser and evaporator duty agree to 0.1 to 0.15 %, which is Cp x 0.226 K.

## P-T phase envelope: natural-gas-like mixture (DWSIM 10.2.12.0)

Mixture: 0.80 CH4, 0.10 C2H6, 0.05 C3H8, 0.03 n-C4H10, 0.02 n-C5H12. Peng-Robinson with the DWSIM
constants, k_ij and R = 8.314 (see `natgas_phase_envelope.json`). Operating point 300 K, 1 atm (vapor).

DWSIM options: phase identification curve and operating point; stability curve and quality lines off.
Bubble-point initialization, initial flash PVF, T_max 1000 K, dP 50000 Pa, dT 2.5, 300 points.
Two exports are stored (`natgas_phase_envelope.csv`): default initialization (the bubble branch
starts at 85.75 K) and custom initialization at P = 1.01325 bar (the same starting point as
`PhaseEnvelope.trace`, which needs no initialization options).

Method (`thermo_sim.phase_envelope.PhaseEnvelope`): Michelsen continuation in (ln K, ln T, ln P) from
the bubble and dew points at 1 atm up to the critical point, which is interpolated across the sign change of ln K (a smooth
polynomial fit of ln T and ln P on the last points of both branches). The cricondentherm and cricondenbar are the extrema of T and P along the curve.
The phase identification curve is the equal-Gibbs-energy line of the mixture as a pseudo component
below the pseudo-critical point and Pi = 1 (Venkatarathnam and Oellrich) above it.

| Quantity | thermo-sim | DWSIM |
|---|---|---|
| Critical point | 246.738 K, 98.0392 bar | 246.738 K, 98.0392 bar |
| Cricondentherm | 300.903 K, 60.33 bar | 300.903 K, 60.05 to 60.55 bar |
| Cricondenbar | 108.528 bar at 270.35 K | 108.528 bar at 270.238 K (1 K grid) |
| Bubble and dew points | max distance 0.0024 and 0.0072 (K, 100 ln P) | |

Known differences, not model differences: DWSIM reports P = 1E-5 bar on the phase identification
curve below 178.42 K; its Pi = 1 points carry finite-difference noise of about 5e-4 in P.