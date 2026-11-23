# thermo-sim

[![CI](https://github.com/PabloHdzm/thermo-sim/actions/workflows/ci.yml/badge.svg)](https://github.com/PabloHdzm/thermo-sim/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Thermodynamics and unit-operation library for process simulation, written in Python
(NumPy / SciPy / pandas). It implements cubic equations of state, phase equilibrium, the
full set of stream properties used for process design, and a set of unit operations
(compressors, coolers, flash drums, reactors, recycle loops, shortcut and rigorous
distillation columns, phase envelopes), all compared with DWSIM 10.2.12.0 and CoolProp.

```python
from thermo_sim import METHODS, Stream, load_database
from thermo_sim.unitops.column import RigorousColumn

feed = Stream(description='feed', fm=None, comp_mass=None, fn=0.1,          # 100 mol/s
              comp_mol={'benzene': 0.5, 'toluene': 0.5}, V_frac=None,
              T=370.0, P=1.01325, Tref=298.15, Pref=1.01325, gamma=None,
              component_data=load_database(), kij=None, method=METHODS['PR'],
              emit_warnings=False)

col = RigorousColumn(n_stages=30, feed_stage=15, P_top=1.01325,
                     reflux_ratio=3.0, bottoms_flow=50.0)       # bottoms in mol/s
res = col.solve(feed)                                           # Naphtali-Sandholm, full MESH

print(res.converged, res.iterations)
print(f"Qc = {res.Qc_kW:.1f} kW   Qr = {res.Qr_kW:.1f} kW")
print(res.profile()[['T_K', 'L_mol_s', 'V_mol_s', 'x_benzene']].iloc[[0, 14, 29]].round(4))
```

```
True 6
Qc = 6074.4 kW   Qr = 3827.4 kW
         T_K   L_mol_s   V_mol_s  x_benzene
1   352.9334  150.0000    0.0000     0.9993
15  370.4149  165.5937  187.2353     0.3367
30  383.8151   50.0000  114.7077     0.0007
```

This run uses the constants of the package database. With DWSIM's own constants the same
column reproduces the DWSIM tutorial to 0.0012 K in temperature and 0.006 % in duty
(see [Validation at a glance](#validation-at-a-glance)).

## Validation at a glance

Every row is reproduced by a test in [`tests/`](tests). Details, conventions and known
differences are in [`docs/validation_unit_operations.md`](docs/validation_unit_operations.md) and
[`docs/validation_distillation.md`](docs/validation_distillation.md). The reference numbers and their
provenance are in [`reference_data/dwsim/`](reference_data/dwsim). Agreement with DWSIM
validates this implementation (same equation, constants and parameters), not accuracy
against experiment. Known differences are listed with their cause, not hidden.

| System | Reference | Largest difference | Test |
|---|---|---|---|
| H2/N2/NH3 TP grids, Peng-Robinson (175 points) and SRK (163 points), 200 to 1000 K, 1 to 200 bar | DWSIM | vapor fraction 1e-6, Z 5e-6, fugacity coefficients 5e-6, enthalpy dH/(R T) 1.3e-5 | `test_dwsim_grid.py` |
| H2/N2/NH3 flash at 150 bar | DWSIM | temperature 4e-4 K, phase compositions 6e-5, fugacity coefficients 2e-5 | `test_dwsim_ammonia_feed.py` |
| Benzene/toluene flash: bubble and dew points, PT, PH and PS | DWSIM | bubble and dew temperature 0.0005 K, fugacity coefficients 2e-5 | `test_flash_dwsim_benzene_toluene.py` |
| Pure benzene and toluene, 300 to 400 K | DWSIM | enthalpy and entropy after removing formation terms and the Cp polynomial difference (liquid Cp 0.4 to 0.8 % below DWSIM) | `test_pure_components_dwsim.py` |
| Ammonia synthesis loop (compressor, heater, equilibrium reactor, coolers, flash drum) | DWSIM | with DWSIM's ideal-gas Cp: compressor outlet 0.019 K, power and duties below 1e-4 (heater 2.4e-4) | `test_dwsim_loop.py` |
| Propane refrigeration cycle (compressor, condenser, valve, evaporator) | DWSIM | work, condenser and evaporator duty within 0.15 %, COP 2.79 | `test_dwsim_cycle.py` |
| Equilibrium reactor, ammonia synthesis | DWSIM | conversion within 0.01 points (equation mode, 500 to 800 K) | `test_equilibrium_reactor.py` |
| Water-gas shift: conversion, equilibrium and Gibbs reactors | DWSIM | composition 5e-4 (equilibrium) and 1e-3 (Gibbs), conversion within 0.3 points | `test_wgs_reactors.py` |
| Heat exchanger (UA, LMTD and effectiveness-NTU modes, vapor-fraction specifications), DWSIM Heat Exchanger Design tutorial | DWSIM | outlet temperatures within 0.35 K, duty within 0.2 to 1 %, LMTD within 0.1 K | `test_heat_exchanger.py` |
| Recycle loop, DWSIM Recycle Loops tutorial (water, gamma-phi) | DWSIM | flows 1e-4, heater duty 1.2 % (liquid Cp 0.6 % above IAPWS-IF97) | `test_recycle_loop.py` |
| Natural gas dew-point control: chiller and separator at 70 bar | DWSIM | chiller duty 0.6 %, vapor fraction 0.002, compositions 5e-4 (1.6e-5 observed) | `test_natgas_dew_point_control.py` |
| Methanol synthesis flowsheet | DWSIM | cooler and separator on the DWSIM reactor outlet: vapor fraction 2e-4, compositions 2e-5 | `test_methanol_synthesis.py` |
| P-T, P-H, P-S, T-H, T-S, V-P and V-T phase envelopes, natural gas and N2/CO2 mixtures | DWSIM | critical points identical, curves within 0.01 in (K, 100 ln P) | `test_dwsim_phase_envelope.py` |
| Shortcut column (DSTWU), four feed states | DWSIM | minimum reflux and stages 1e-4, duties 1e-4 to 3e-4 | `test_dstwu.py` |
| Rigorous column (MESH), 30 stages, benzene/toluene tutorial | DWSIM | temperature 0.0012 K, flows 7e-5, compositions 4.5e-5, duties 0.006 % | `test_column.py` |
| Pure N2 and H2 with PRSV2 | DWSIM | Z 5e-6 and fugacity coefficients 1e-5 (H2), N2 within 3e-6 | `test_dwsim_pure.py` |
| Entropy of mixing | DWSIM | empirical rule, presented as a hypothesis with its evidence | `test_dwsim_mixing_entropy.py`, [`docs/validation_dwsim_entropy_mixing.md`](docs/validation_dwsim_entropy_mixing.md) |

Thermodynamic consistency does not depend on any simulator:
[`tests/test_isobaric_consistency.py`](tests/test_isobaric_consistency.py) integrates dH / T
along isobars with thermo-sim's own flashes and finds the entropy consistent to 7e-8 R in
the single-phase and two-phase regions.

### Rigorous column

`RigorousColumn` solves the full MESH equations (component balances, equilibrium, summation
and energy balances) with the Naphtali-Sandholm method: unknowns are the logarithms of the
component flows of both phases and the temperature of each stage, Newton steps are damped
and line-searched, and the Jacobian is built by finite differences with stage coloring
(a stage couples only to its neighbours, so the cost of a Jacobian does not grow with the
number of stages).

Compared with the DWSIM benzene/toluene tutorial (30 stages including condenser and reboiler,
feed at stage 15, total condenser, reflux ratio 3, bottoms 50 mol/s, 1 atm):

| Quantity | Difference |
|---|---:|
| Stage temperatures | 0.0012 K |
| Liquid and vapor flows | 7e-5 relative |
| Compositions x and y | 4.5e-5 absolute (4e-4 relative for the minority component) |
| Condenser duty | +0.0055 % |
| Reboiler duty | -0.0017 % |

The residual difference is traced to the ideal-gas heat capacity: applying DWSIM's polynomial
through a diagnostic hook improves temperatures to 0.0005 K and flows to 9e-6, down to the digits
DWSIM prints. The converged MESH solution does not depend on the algorithm, so profiles
are compared and not iteration paths. The column also closes its own equations (component and
energy balances, equilibrium, sums of fractions) to solver precision independently of DWSIM.
Phase 1 supports reflux ratio and bottoms flow as specifications, a total condenser, uniform
pressure and a stage efficiency of 1; other options raise `NotImplementedError` and are on the
roadmap.

### Temperature-pressure grids (H2/N2/NH3)

All comparisons use a H2/N2/NH3 feed (0.726 / 0.242 / 0.032). The critical constants,
acentric factors, binary interaction parameters and R = 8.314 are DWSIM's, so what is
compared is the implementation. Stream sensitivity analyses exported from DWSIM are compared
point by point with a TP flash
([`tests/test_dwsim_grid.py`](tests/test_dwsim_grid.py); run
`python scripts/compare_dwsim_grid.py` for the full table of differences).

| | Peng-Robinson | SRK (`srk72`) |
|---|---|---|
| Points | 175 (32 two-phase) | 163 (32 two-phase) |
| Range | 200 to 1000 K, 1 to 200 bar | 200 to 1000 K, 1 to 200 bar |
| kij | DWSIM's (H2-N2 0.0711, N2-NH3 0.2193) | all zero, as DWSIM |
| Liquid volume | `eos` | `eos` |

| Largest difference over the grid | PR | SRK |
|---|---:|---:|
| Vapor fraction (absolute) | 1e-6 | 1e-6 |
| Z, vapor (absolute) | 5e-6 | 5e-6 |
| Z, liquid (relative) | 3e-6 | 3e-6 |
| Fugacity coefficients, vapor (relative) | 5e-6 | 5e-6 |
| Fugacity coefficients, liquid (relative) | 4e-6 | 4e-6 |
| Mixture density (relative) | 7e-6 (all rows) | 6e-6 (all rows) |
| Isothermal enthalpy difference, dH / (R T) | 1.3e-5 | 9e-6 |
| Isothermal entropy difference, dS / R, single phase | 6.4e-6 | 8.5e-6 |

* **Energy** is compared as X(T, P) - X(T, 1 bar). This removes the ideal-gas Cp, the formation
  enthalpy and the different reference states of the two programs.
* The `srk` form (Graboski-Daubert kappa) is *not* the one DWSIM uses; it differs from DWSIM by
  up to 0.3 % in Z. Use `METHODS['SRK72']` to reproduce DWSIM.
* **DWSIM artifact, two-phase entropy.** In the two-phase region DWSIM's mixture entropy does
  not satisfy dH = T dS at constant pressure (the excess is about 1.8 R per mole of liquid),
  while thermo-sim's does. Those points are marked as expected failures, and the consistency
  test above shows which program is consistent.
* PRSV2 is compared for the 95 rows that an independent implementation also reproduces. Energies
  are not compared: DWSIM freezes kappa in the temperature derivative of alpha, thermo-sim
  differentiates the complete expression ([details](reference_data/README.md)).

### Known differences with DWSIM

These are not defects of either side; each has a documented cause.

* **Ideal-gas heat capacity.** DWSIM uses `A + exp(B/T + C + D T + E T^2)`; thermo-sim uses a
  polynomial fitted to CoolProp. With the packaged data, enthalpy differs by a few J/mol and
  a heated or compressed stream by 0.25 to 0.6 K; with DWSIM's equation substituted the
  differences in power and duties fall to 1e-4 or 2e-4. For columns the effect is about 0.007 %.
* **Saturation point of a pure component.** DWSIM places it with the compound's vapor-pressure
  correlation, not with the equation of state (propane at 2.45 bar: 252.995 K against 253.221 K
  from the EOS). Streams on the dome inherit the shift and are marked as expected failures.
* **Equilibrium reactor conventions.** DWSIM uses a standard-state pressure of 1 atm and
  evaluates `ln K(T)` at T + 1 K when it is given as an expression.
* **Mixing entropy.** DWSIM's two-phase and mixture entropy follows an empirical rule that
  thermo-sim reproduces as a hypothesis ([evidence](docs/validation_dwsim_entropy_mixing.md)).
* **Formation terms.** DWSIM's displayed molar enthalpy excludes the enthalpy of formation, its
  energy flow includes it; tests compare like with like.

## Features

* **Equations of state:** Peng-Robinson, SRK (`srk`, Graboski-Daubert kappa, and `srk72`,
  Soave's original 1972 kappa as used by DWSIM) and RK, with interchangeable alpha functions
  (Soave/standard, Boston-Mathias, Mathias-Copeman, PRSV/iPRSV, Graboski-Daubert for H2)
  assignable per component, and ready-made presets (`METHODS`).
* **Phase equilibrium:** Michelsen stability test, TP flash, vapor-fraction/pressure
  flash, PH and PS flashes, bubble and dew temperatures, pure-component saturation from the
  EOS spinodals, P-T phase envelopes by continuation. The liquid phase can use the EOS
  (phi-phi) or a gamma-phi model with a pluggable activity model.
* **Stream properties:** enthalpy, entropy and heat capacity split into ideal-gas,
  departure and total parts, for the mixture and for each pure component, per phase;
  fugacity coefficients, compressibility, density (four liquid-volume models),
  viscosity and thermal conductivity.
* **Stream algebra:** adiabatic mixing (`a + b`), subtraction, scaling, splitting.
* **Unit operations:** isentropic and polytropic compressors, coolers and heaters, heat exchanger
  (UA, LMTD, effectiveness-NTU), valve, flash drum, equilibrium, stoichiometric and Gibbs reactors (isothermal and adiabatic), recycle
  loops, shortcut column (DSTWU) and rigorous column (MESH, Naphtali-Sandholm).
* **Property database** built from CoolProp correlations (`scripts/build_db.py`) with a
  fit-quality report.

## Installation

```bash
git clone https://github.com/PabloHdzm/thermo-sim.git
cd thermo-sim
pip install -e ".[dev]"
pytest
```

Python 3.10 or newer. Rebuilding the database additionally needs `pip install -e ".[db]"`.

## Quick start

```bash
python examples/dwsim_ammonia_feed.py
```

```
T = 248.4686 K   phase = two_phase   vapor fraction = 0.9900
vapor  y = [0.733274 0.244432 0.022293]
liquid x = [0.005841 0.001185 0.992974]
H  total  (mix / vapor / liquid) = -3416.82 / -2741.69 / -70255.45 kJ/kmol
S  total  (mix / vapor / liquid) = -46.3730 / -44.8007 / -202.0322 kJ/(kmol K)
Cp total  (mix / vapor / liquid) = 31.585 / 31.149 / 74.704 kJ/(kmol K)
density   (mix / vapor / liquid) = 62.22 / 61.13 / 659.78 kg/m3
Z (vapor / liquid) = 1.0340 / 0.2041
```

### Conventions

* Gas constant: `R` is a parameter of the thermodynamic method and defaults to the CODATA
  value (`R_GAS = 8.314462618` J/(mol K) in `constants.py`). DWSIM uses 8.314; use
  `METHODS['PR'].but(R=8.314)` to reproduce it. Phase equilibrium does not depend on `R`;
  molar volumes, densities, entropies and the Rackett/Peneloux volumes do.
* Units: T [K], P [bar], molar flow [kmol/s], mass flow [kg/s], H [kJ/kmol],
  S and Cp [kJ/(kmol K)], density [kg/m3]. Lower-case `h`, `s`, `cp` are per kg. Column
  profiles and the column bottoms specification are in mol/s, as in the DWSIM tables.
* Energy properties are named `<X>_<kind>_<scope>[_<phase>]`, with `X` in `H, S, Cp`,
  `kind` in `ig` (ideal gas), `dep` (departure), `total`; `scope` in `mix`, `indiv`
  (pure components); phase `gas` or `liq`. Without a phase suffix you get the active
  phase, or the molar-weighted value for a two-phase stream.
* **A phase that is absent from the stream is `NaN`**, never a made-up value.
* Components are ordered by critical temperature; pass `kij` as a DataFrame indexed by
  component name, or as an array matching the active components.

### Choosing a thermodynamic method

```python
from thermo_sim import METHODS
METHODS['PR']                                   # Peng-Robinson, database constants
METHODS['PR'].but(liquid_volume='poly')         # change one layer
METHODS['PR'].but(constants_user={'hydrogen': {'Tc': 33.19, 'Pc': 13.13}}, R=8.314)
```

A method is a combination of EOS form, alpha function (global and per component),
critical constants, liquid-volume model (`eos`, `eos_peneloux`, `rackett`, `poly`),
liquid phase model and gas constant. Critical constants can be overridden per component
(`constants_user`) or from a table (`OVERRIDES_CSV_PATH` in `methods.py`).

## Performance

Median time per call on a Windows machine with Python 3.12 (single core; times depend on the
machine). Each `Stream` construction runs the flash and assembles the property set.

| Operation (H2/N2/NH3 mixture unless noted) | Time |
|---|---:|
| TP flash, single phase (500 K, 100 bar) | 96 ms |
| TP flash, two phase (254 K, 150 bar) | 108 ms |
| Vapor-fraction/pressure flash (V_frac 0.99, 150 bar) | 221 ms |
| Isentropic compressor stage (pure H2) | 103 ms |
| Polytropic compressor stage (pure H2, 20 integration steps) | 1.35 s |
| 100 x 100 temperature-pressure grid (10 000 TP flashes) | 622 s (62 ms per flash) |

The rigorous column avoids building a `Stream` per stage: `StageThermo` evaluates K values and
phase enthalpies directly from the equation of state (identical to `Stream` to 1e-12, about
1 ms per stage evaluation against tens of ms for a `Stream`), and the 30-stage tutorial converges in 6 Newton iterations. This is
otherwise an unoptimized pure-Python implementation; profiling and vectorization (for example
with Numba) are on the roadmap.

## Project layout

```
src/thermo_sim/
├── constants.py           physical constants
├── alpha.py               alpha functions
├── eos.py                 cubic EOS, closed-form roots, residual properties
├── volume.py              liquid molar-volume models
├── methods.py             ThermoMethod and presets (METHODS)
├── equilibrium.py         stability, flash, bubble/dew, saturation
├── phase_envelope.py      P-T envelope by Michelsen continuation
├── activity.py            activity-coefficient models, gamma-phi liquid
├── properties.py          StreamProperties (result container)
├── stream_equilibrium.py  phase assignment and fugacities for a stream
├── stream.py              Stream: the stream object
├── stage_thermo.py        fast per-stage K values and enthalpies for column solvers
├── database.py            load_database()
├── unitops/
│   ├── compressor.py      single-stage compressor (isentropic, polytropic)
│   ├── cooler.py          isobaric cooler/heater
│   ├── valve.py           isenthalpic valve
│   ├── heat_exchanger.py  heat exchanger (UA, LMTD, effectiveness-NTU)
│   ├── train.py           multistage compressor train with intercoolers
│   ├── flash_drum.py      flash drum with saturated vapor and liquid outlets
│   ├── reactor.py         stoichiometric and equilibrium reactors
│   ├── gibbs_reactor.py   Gibbs-energy minimization reactor
│   ├── recycle.py         recycle loop convergence
│   ├── dstwu.py           shortcut column (Fenske, Underwood, Gilliland)
│   └── column.py          rigorous column (MESH, Naphtali-Sandholm)
└── data/                  property database, critical-constant overrides
scripts/                   database builder and DWSIM comparison scripts
tests/                     pytest suite (EOS, equilibrium, stream, unit operations,
                           DWSIM comparisons, thermodynamic consistency)
reference_data/dwsim/      DWSIM reference results with provenance
examples/                  runnable examples
docs/                      validation notes and fit quality of the property database
```

## Known limitations

* Validation covers the systems in the table above. Viscosity and thermal conductivity use
  simple mass-weighted mixing and are not yet validated.
* The multistage compressor train (`CompressorTrain`) is tested against closed-form ideal-gas
  results and internal balances only: DWSIM has no multistage compressor block. A cascade of validated compressor
  and cooler blocks in DWSIM is on the roadmap.
* The rigorous column supports reflux ratio and bottoms flow as specifications, a total
  condenser, uniform pressure and efficiency 1. Column height and diameter are not computed yet.
* Ideal-gas Cp is a polynomial fitted to CoolProp over 200 to 1000 K (maximum fit error
  0.11 %); outside that range it is extrapolated linearly and a warning is raised.
* No Henry's law; the gamma-phi model takes user-supplied activity coefficients only.
* Bubble and dew temperatures of a liquid close to pure NH3 above its critical pressure
  return NaN (and some H2-rich mixtures too); this is covered by expected-failure tests.
* The volume shift of `eos_peneloux` is applied to the liquid only.

## Roadmap

* Column specifications (distillate flow, reboiler duty, component recovery or fraction, boil-up
  ratio, temperature), pressure profile, subcooling, partial condenser, Murphree efficiency,
  multiple feeds and side draws, then an Inside-Out solver and column internals.
* DWSIM cascade of compressor and cooler blocks to validate the multistage train.
* Performance: profiling and vectorization of the flash kernels.

## Data sources and license

Pure-component correlations are fitted to [CoolProp](http://www.coolprop.org) (MIT
license). Formation enthalpy and Gibbs energy come from NIST (NIST-JANAF tables and the
NIST Chemistry WebBook); the `source` column of `data/raw/formation_data.csv` gives the
origin of each value. Released under the MIT license (see `LICENSE`).

[DWSIM](https://dwsim.org) is an independent open-source project. Its results are used here
only as reference numbers; no DWSIM code is used or distributed, and this project is not
affiliated with it.