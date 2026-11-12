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

# ammonia_feed_grid_prsv2.csv

Same feed, DWSIM 10.2.12.0, **PRSV2** package, same critical constants and R = 8.314, all kij = 0
(DWSIM reports no interaction parameters), liquid density from the EOS. 135 points (T = 200-1000 K,
P = 1-200 bar plus the 230-320 K block). DWSIM shows these PRSV2 parameters:

| | kappa1 | kappa2 | kappa3 |
|---|---:|---:|---:|
| H2 | 0 | 0 | 0 |
| N2 | 0.02 | 0.3162 | 0.5350 |
| NH3 | 0.001 | -0.1265 | 0.51 |

What reproduces the DWSIM numbers (checked with an independent numpy implementation and
confirmed in the DWSIM 10.2.12.0 PRSV2 source):

* PRSV2 alpha, kappa = kappa0 + [k1 + k2 (k3 - Tr)(1 - sqrt(Tr))](1 + sqrt(Tr))(0.7 - Tr), with
  the Stryjek-Vera kappa0 polynomial (cubic coefficient 0.0196544 in DWSIM), **applied at every
  Tr** (N2 has Tr > 1 over the whole grid; dropping the correction above Tr = 1 gives 3 % errors
  in Z).
* **DWSIM uses this form only if kappa1 * kappa2 * kappa3 is nonzero.** Otherwise it falls back
  to the plain Peng-Robinson alpha, kappa = 0.37464 + 1.5422 w - 0.26992 w^2 (w <= 0.491). H2 has
  all three parameters at zero, so it follows the PR alpha (kappa = 0.02893), not kappa0 of PRSV
  (0.04893). thermo-sim reproduces this with `alpha_overrides={'hydrogen': 'std'}`.
* kappa1 of N2 is 0.01996 (the Stryjek-Vera value); DWSIM displays it rounded to 0.02. The pure
  N2 runs cannot tell the two apart (both agree to the printed digits), but without the
  correction (kappa1 = kappa2 = kappa3 = 0) they differ by 1e-5 to 4e-4 in Z.
* Mixing rule: a_ij = sqrt(a_i a_j) (1 - x_i kij - x_j kij2) (Margules form), which with
  kij = kij2 = 0 is the standard quadratic rule.

Pure components (`ammonia_pure_prsv2.csv`): H2 at 100, 200, 300 K and 50, 150 bar (PR alpha with
1.5422) agrees with DWSIM to 5e-6 in Z and 1e-5 in phi; N2 vapor at 90, 100, 110 K on the DWSIM
saturation pressures (PRSV2) agrees to 2e-6; the N2 liquid root at the same states agrees to 3e-6 in Z, 2e-6 in phi and 2e-7 in density. DWSIM's own phi_l/phi_v at those points is 1.00085, 1.00265 and 1.00382 (thermo-sim: the same), so its pressures are not the EOS saturation pressure (0.1 to 0.6 % below it); the numbers match anyway, since both phases are evaluated at the same T and P. The coefficient 1.54226 (more digits) is worse, and
kappa0 alone (no override) is off by 0.1 to 0.8 % for H2.

Rows are classified in the `status` column and only `reproduced` ones (95, all at T <= 400 K) are
used by the tests:

| status | rows | meaning |
|---|---:|---|
| reproduced | 95 | independent numpy PRSV2 flash agrees within 2e-4 (most within 1e-5) |
| unstable_single_phase | 12 | DWSIM reports one vapor phase, but a dense N2 phase has a negative tangent-plane distance (350 K at 50 and 75 bar; 500 to 800 K) |
| dense_root_mismatch | 14 | T >= 500 K, Z of 0.02 to 0.1: agrees to 2e-4 to 6e-4, too sensitive to the 4-digit kappa values |
| dwsim_liquid_only | 7 | DWSIM reports a single liquid phase where a vapor root exists |
| two_phase_not_reproduced | 5 | DWSIM two-phase solutions that the independent flash did not find |
| nonfinite | 2 | fugacity coefficients of 1e288 or infinity |

**Energies are not compared for PRSV2.** DWSIM's enthalpy and entropy match, to 1e-6 in dH/(R T),
a calculation in which the temperature derivative of alpha is taken with kappa frozen (kappa
treated as constant; confirmed in the DWSIM source, where d(a)/dT omits dkappa/dT). The exact derivative, which includes dkappa/dT, differs by 0.3 to 0.4 R T
(and 0.3 to 0.4 R in entropy) at 75 to 200 bar. DWSIM's PRSV2 energies are therefore not derived consistently from
its own equation of state, and thermo-sim (which differentiates the full alpha) is not expected to
reproduce them. `tests/test_isobaric_consistency.py` checks that thermo-sim's PRSV2 energies are
consistent with its own equation instead.

**The instability above 500 K is the equation, not only the flash.** With kappa2 > 0 and kappa3
below Tr, the N2 correction term grows like Tr^2 and kappa reaches about -120 at 1000 K, so the
attractive parameter a(T) of N2 becomes absurdly large (Z of 0.0002, fugacity coefficients of
1e288). PRSV2 is a correlation for Tr < 1 and DWSIM extrapolates it. At 350 and 400 K it even
produces an N2-rich "liquid" (vapor fraction 0.758 = 1 - y(N2)); the independent flash reproduces
those points (`reproduced`, 400 K at 75 to 200 bar), so they are valid solutions of the equation.
At 500 K and above the feed is unstable as a single vapor (tangent-plane distance of dense N2 from
-100 at 500 K to -457 at 600 K, in units of RT), yet DWSIM reports one vapor phase: those rows are
`unstable_single_phase`, and a stability test (as in thermo-sim) correctly finds two phases there.