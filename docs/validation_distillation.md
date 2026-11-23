# Validation of distillation columns

This document records how the shortcut column (`DSTWU`, `ShortcutColumn`) and the rigorous
column (`RigorousColumn`, Naphtali-Sandholm) of `thermo-sim` were compared with DWSIM 10.2.12.0
on the benzene/toluene system, what agrees, what does not and why. Every number is checked by
a test (section 8).

> **Scope.** One binary (benzene/toluene) with Peng-Robinson and DWSIM's constants, at 1 atm and
> at 2 bar. The shortcut column is compared in nine configurations (feed state, composition,
> reflux, pressure, condenser type). The rigorous column is compared on the DWSIM tutorial
> (30 stages, total condenser) and, for robustness only, on variants with no DWSIM data.
> Height and diameter of the rigorous column are not computed yet; they will come with column
> internals. Agreement with DWSIM validates this implementation, not accuracy against
> experiment.

## 1. Shortcut column (DSTWU)

DWSIM's Shortcut Column was run behind a heater that sets the feed condition: 100 mol/s of
benzene/toluene at 300 K, heated to 370 K, total condenser, reflux ratio 3, 5 % of the light
key in the bottoms and 5 % of the heavy key in the distillate, 1.01325 bar.

The algorithm was recovered from DWSIM's numbers and then reproduced:

* **One relative volatility, taken at the feed state.** DWSIM evaluates `alpha = K_LK / K_HK`
  once, at 370 K and 1.01325 bar, and uses it in both Fenske (minimum stages 6.76391) and
  Underwood (theta 1.61215, minimum reflux 1.84130). The mean of the values at the top and
  bottom would give 6.7007 and 1.8225, which the tests show are not DWSIM's numbers.
* **For a single-phase feed**, the K values use the stability trial phase: `K = phi_L(x*) /
  phi_V(z)` for a vapor feed, with `x*` the incipient liquid (the fugacity coefficients are in
  DWSIM's stream table), and `phi_L(z) / phi_V(y*)` for a liquid feed.
* **Gilliland in the Eduljee form**, `Y = 0.75 (1 - X^0.5668)`, gives 11.49027 stages (DWSIM
  11.4903). The Molokanov form would give 11.699.
* **Feed stage** is `N n_S / (n_R + n_S)`, with `n_R` and `n_S` the Fenske minimum stages of the
  rectifying and stripping sections using the same alpha. Kirkbride is 8 % off for the 70/30
  feed and 15 % off for the 90/10 feed.

| Case | Compared | Tolerance |
|---|---|---|
| 50/50, two-phase feed (tutorial case) | minimum reflux, minimum stages | 1e-4 absolute |
| | stages, feed stage | 1e-3 |
| | internal flows (rectifying) | 1e-9 mol/s |
| | internal flows (stripping) | 5e-3 mol/s |
| | condenser duty, reboiler duty | 1e-4, 3e-4 relative |
| | product temperatures | 0.01 K |
| 70/30 and 90/10 as 100 % vapor at 370 K | all of the above | 5e-4 relative; duties 2e-4 and 3e-4 |
| 90/10 as subcooled liquid at 300 K (no heater) | minimum reflux (negative: q = 1) | 2e-4 |
| | stages, flows, duties | as above, reboiler 1e-3 |
| Reflux ratio 2 and 10 | stages and feed stage | 5e-5 relative |
| Column pressure 2 bar, and 1.01325 bar top with 1.2 bar bottom | all of the above | as the base case |
| Partial condenser (vapor distillate) | stages, flows, duties, distillate temperature | as the base case |

Further behavior, checked against DWSIM or against the definitions:

* A reflux ratio below the minimum is refused (DWSIM raises an error at R = 1.5 for a minimum of
  1.8413), and so is a condenser pressure above the reboiler pressure.
* The Underwood equations and the Fenske equation hold for the solution; the mass and energy
  balances close; the flows follow from `q`; the recovery specification is equivalent to the
  composition one; giving the number of stages returns the reflux ratio, and the reflux can be
  given as a multiple of the minimum.
* **Height.** DWSIM's single reported estimate is reproduced by `0.5 (N + 2)` m for a 0.5 m
  tray spacing (the tutorial case and the reflux 2 and 10 cases, to 1e-4). It is a hypothesis
  fitted to those points. The estimated diameter is not compared.
* Options that DWSIM does not have (Winn alpha, averaged alpha, Molokanov, number of stages as
  input, several components) are tested for internal consistency, not for agreement with a
  simulator.

## 2. Rigorous column: model

`RigorousColumn` solves the full MESH equations with the Naphtali-Sandholm method: component
balances, phase equilibrium, summation of fractions and energy balances on every stage.

* **Unknowns** per stage: the logarithms of the component flows of the liquid and of the vapor,
  and the temperature. The condenser uses the logarithm of the total condensate, the fractions
  of the incipient vapor and the temperature.
* **Equations**: component balances; equilibrium in logarithmic form, `ln(y/x) = ln K` with
  `K = phi_L(x) / phi_V(y)` from the equation of state; energy balances. The reboiler replaces
  its energy balance with the bottoms specification; the duties of condenser and reboiler are
  outputs.
* **Newton method** with damping (logarithmic steps at most 2, temperature steps at most 15 K)
  and a backtracking line search on the 2-norm of the residual. The Jacobian is built by finite
  differences with stage coloring: stage `j` couples only to `j-1`, `j` and `j+1`, so the cost
  of a Jacobian does not grow with the number of stages. Energy and flow scales are chosen so
  that all residuals are of order one.
* **Initial estimate**: a sharp split by volatility at the feed, linear interpolation of
  `ln x` between top and bottom compositions, bubble-point temperature per stage, constant
  molar overflow.
* **`StageThermo`** evaluates K values and phase enthalpies for a stage directly from the
  equation of state, without building a `Stream`. It is identical to `Stream` to 1e-12 in
  enthalpy and about 1 ms per evaluation against tens of ms for a `Stream`.
* **Supported now**: reflux ratio and bottoms flow as specifications, total condenser, uniform
  pressure, efficiency 1. Anything else raises `NotImplementedError` (see section 7).

Because the converged MESH equations have the same solution whichever algorithm solves them
(DWSIM offers Wang-Henke, Naphtali-Sandholm and Inside-Out, with a tolerance of 1e-5), the
validation compares profiles and not iteration paths. The DWSIM "Iterations Taken: 0" entry is
meaningless for this case and is not compared.

## 3. Rigorous column: DWSIM conventions found

* Stage 1 is the condenser and stage N the reboiler, and N = 30 counts both.
* The feed enters stage 15 as a whole, liquid and vapor together (the stage balance of 351.732
  mol/s closes on both sides).
* The reported liquid flow `L` is the liquid leaving the stage (for stage 1, the reflux of 150
  mol/s; the distillate leaves separately) and `V` the vapor leaving it (zero for a total
  condenser).
* DWSIM prints the reboiler duty as a negative number (heat added); thermo-sim reports both
  duties as positive magnitudes.
* The tutorial feed at 370 K is two-phase (vapor fraction 0.721625) although the tutorial text
  calls it saturated liquid, and the comparison uses the real state.

## 4. Rigorous column: results

DWSIM tutorial: benzene/toluene, 100 mol/s at 370 K, 30 stages, feed at stage 15, total
condenser, reflux ratio 3, bottoms 50 mol/s, 1.01325 bar, Peng-Robinson with DWSIM's
constants.

| Quantity | Difference | Test tolerance |
|---|---:|---:|
| Stage temperatures | 0.0012 K | 0.004 K |
| Liquid and vapor flows | 7e-5 relative | 3e-4 |
| Compositions x and y | 4.5e-5 absolute | 2e-4 |
| Composition of the minority component | 4e-4 relative | 2e-3 |
| Condenser duty | +0.0055 % | 2e-4 relative |
| Reboiler duty | -0.0017 % | 2e-4 relative |
| Product temperatures | below 0.004 K | 0.004 K |
| Product densities | | 2e-4 relative |
| Preheated feed, vapor fraction | | 1e-4 |
| Preheated feed, y/x | | 1e-5 |

The solver needs 6 Newton iterations. Independently of DWSIM, the solution closes its own
equations: component and stage balances, equilibrium, sum of fractions and the overall energy
balance (below 1e-5 kW); the condenser liquid is at its bubble point; temperature decreases
monotonically from reboiler to condenser; and a higher reflux gives a purer distillate with
higher duties.

### What the remaining difference is

The residual difference to DWSIM is the ideal-gas heat capacity. DWSIM uses
`A + exp(B/T + C + D T + E T^2)`, the library a polynomial fitted to CoolProp. Applying DWSIM's
polynomial through a diagnostic hook (`h_ig_correction`, to the enthalpy of every stage and of
the feed, which must be corrected consistently) improves every figure by a factor of 2 to 10:

| Quantity | Plain | With DWSIM Cp |
|---|---:|---:|
| Temperatures | 0.0012 K | 0.0005 K |
| Flows | 7e-5 | 9e-6 |
| Compositions | 4.5e-5 | 4e-6 |

That is down to the digits DWSIM prints. What remains in the minority component
(3e-4 relative) is DWSIM's own tolerance and the sharp split of the initial estimate. The hook
is a diagnostic, not a model option: the effect of the Cp polynomial on a column is about
0.007 %, so no override is needed. (An earlier estimate of 0.1 to 0.3 % was conservative.)

### Robustness

Without DWSIM data, six variants converge and keep every balance: other feed states (liquid and
vapor), feed stages, reflux ratios and column sizes. In a wider run, 36 random variants of these
converged.

## 5. Reading the results

* **The equations agree.** The temperature profile to 1 mK, flows and compositions to 1e-4 and
  the duties to a few parts in 1e4, with 30 stages of two components, validates the K values,
  the enthalpies, the stage assembly and the Naphtali-Sandholm solver together.
* **The shortcut column reproduces DWSIM's algorithm, not only its results.** The choices
  (alpha at the feed state, trial phase for single-phase feeds, Eduljee form, feed stage rule)
  were recovered from the numbers and each one is tested against the alternative that does
  not reproduce them.
* **Differences are the ideal-gas Cp** in both columns (section 4), and the DWSIM printing
  digits. None of the differences is unexplained.

## 6. Known differences and open points

* DWSIM's feed-stage rule and height estimate for the shortcut column are recovered
  hypotheses, fitted to several points.
* The rigorous column has a total condenser only. The partial condenser of the shortcut column
  is validated; the rigorous one is not implemented yet.

## 7. Limitations and next steps

Phase 1 of the rigorous column supports only reflux ratio and bottoms flow. Next, with DWSIM
runs for each:

1. Specification types: distillate flow, reboiler duty, component fraction or recovery,
   boil-up ratio, temperature.
2. Pressure profile, subcooling, partial condenser.
3. Murphree efficiency.
4. Several components, several feeds and side draws.
5. An Inside-Out solver, then column internals (height and diameter).

Rate-based models, secondary specifications, absorber modes and dynamics are out of scope.

## 8. Reproducing

```bash
pip install -e ".[dev]"
pytest tests/test_dstwu.py tests/test_column.py -v
```

Reference data and provenance:
[`reference_data/dwsim/dstwu_benzene_toluene.json`](../reference_data/dwsim/dstwu_benzene_toluene.json)
and
[`reference_data/dwsim/rigorous_column_benzene_toluene.json`](../reference_data/dwsim/rigorous_column_benzene_toluene.json).
DWSIM is an independent open-source project; its results are used here only as reference
numbers, and no DWSIM code is used or distributed.
