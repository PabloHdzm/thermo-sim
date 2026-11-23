# Changelog

All notable changes to this project are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Planned
- Column specifications beyond reflux ratio and bottoms flow (distillate flow, reboiler duty,
  component fraction or recovery, boil-up ratio, temperature), pressure profile, subcooling,
  partial condenser, Murphree efficiency, several feeds and side draws.
- Inside-Out solver and column internals (height and diameter).
- DWSIM cascade of compressor and cooler blocks to validate the multistage compressor train.

## [0.1.0] - 2026-11-23

First public release.

### Added
- Cubic equations of state (Peng-Robinson, SRK, SRK with Soave's 1972 kappa, RK) with
  interchangeable alpha functions, including PRSV2 as implemented by DWSIM.
- Phase equilibrium: Michelsen stability test, TP, PH, PS and vapor-fraction flashes, bubble and
  dew points, P-T phase envelopes by continuation, gamma-phi liquid model.
- `Stream` with enthalpy, entropy, heat capacity, fugacity coefficients, density, viscosity and
  thermal conductivity, and stream algebra.
- Unit operations: compressor (isentropic and polytropic), compressor train, cooler and heater,
  valve, flash drum, heat exchanger, stoichiometric, equilibrium and Gibbs reactors, recycle
  solver, shortcut column (DSTWU) and rigorous column (MESH, Naphtali-Sandholm).
- `StageThermo`: fast per-stage K values and enthalpies for column solvers.
- Validation against DWSIM 10.2.12.0 and CoolProp, with reference data and provenance in
  `reference_data/dwsim/` and notes in `docs/`.
- Public-hygiene test and continuous integration.
