# DWSIM reference data

Numbers read from the Master Property Table of DWSIM 10.2.12.0 (Peng-Robinson property
package). They are results of a simulation run by the author; no DWSIM code is used or
distributed here. DWSIM is an independent open-source project and this repository is not
affiliated with it.

## How the case was produced

- Components: hydrogen, nitrogen, ammonia (ordered by critical temperature).
- Feed: 0.726 / 0.242 / 0.032 mole fractions, 150 bar, vapor fraction 0.99 (T solved).
- Equation of state: Peng-Robinson with the critical constants, acentric factors and
  binary interaction parameters that DWSIM shows for these components (see the JSON).
- DWSIM uses R = 8.314 J/(mol K); thermo-sim runs use `METHODS['PR'].but(R=8.314, ...)`.
- Liquid density options recorded: plain EOS, Rackett, COSTALD and EOS + Peneloux.

## Conventions

- DWSIM prints fugacity coefficients divided by 1e5.
- DWSIM "Molar Enthalpy" excludes the enthalpy of formation, so the tests compare the
  residual (departure) enthalpy, which does not depend on the ideal-gas reference.
- The ideal-gas Cp used by DWSIM for these components is recorded in the JSON so that the
  ideal part can be rebuilt independently of the thermo-sim database.


## Files

- `ammonia_loop_feed_150bar.json`: inputs, results and the list of quantities that are
  deliberately not compared.
- `tests/test_dwsim_ammonia_feed.py` compares thermo-sim against this file.

# ammonia_feed_grid.csv

DWSIM 10.2.12.0, Peng-Robinson with DWSIM's own constants, kij and R = 8.314. Feed H2/N2/NH3 =
0.726/0.242/0.032 (same as `ammonia_loop_feed_150bar.json`). 175 temperature-pressure points
exported from stream sensitivity analyses (one independent variable at a time):

* `sensitivity_complete_v3`: T = 200-1000 K x P = 1-200 bar (143 points, full precision).
* `sensitivity_refined_230-320K_v2`: T = 230-320 K step 10 at 50/100/150/200 bar (32 new points,
  exported with 3 significant digits in the fugacity coefficients: `phi_digits = 3`).

Columns: fugacity coefficients are the true values (DWSIM prints them divided by 1e5).
`n_phases = 2` only when the liquid fraction is above 1e-4. Rows with a single phase carry no
liquid values: DWSIM then reports hypothetical liquid Z and phi (NaN for T >= 500 K, and one
visible glitch at 300 K / 200 bar), which are not physical and are removed.

Not used: the bubble temperature column (constant 69.58 K in the export). Enthalpy and entropy
are compared as differences against the 1 bar row at the same T (cancels Cp, formation enthalpy
and reference states), so only temperatures that have a 1 bar row enter that comparison.
Independent check: a plain numpy PR implementation reproduces Z, phi, density (single phase,
120 rows) and V, x/y-derived phi, Z (two-phase rows) to ~5e-6, the print precision.

Known DWSIM artifact: in two-phase points the mixture entropy does not satisfy dH = T dS at
constant P (the excess is ~1.8 R per mole of liquid, checked on the 10 K rows at 100-200 bar),
while single-phase points do (to 1e-4 R). Two-phase entropy is therefore not used as a
reference.

# ammonia_feed_grid_srk.csv

Same feed, DWSIM 10.2.12.0, **SRK** package (Soave 1972: kappa = 0.48 + 1.574 w - 0.176 w^2,
Omega_a = 0.42748, Omega_b = 0.08664), same critical constants and R = 8.314, all kij = 0
(DWSIM reports no interaction parameters for this package), liquid density from the EOS (no
Peneloux). 163 points (T = 200-1000 K x P = 1-200 bar plus the 230-320 K block at 50-200 bar),
32 of them two-phase, all with full-precision fugacity coefficients (`phi_digits = 6`).
Columns and filtering rules are the same as for the PR grid. An independent numpy SRK with the
Soave-1972 kappa reproduces the file to 5e-7 (V), 4.5e-6 (Z), 2.5e-6 (Z liquid), 1e-6 (phi vapor),
2.7e-6 (phi liquid) and 3e-6 (two-phase mixture density from the EOS volumes). The thermo-sim
default `srk` form (Graboski-Daubert kappa) differs from DWSIM by up to 0.3 % in Z and 1.8 % in
phi, which is why the `srk72` form exists.