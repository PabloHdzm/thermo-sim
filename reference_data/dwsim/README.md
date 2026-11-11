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
