# thermo-sim

Thermodynamic property calculator for process streams, written in Python
(NumPy / SciPy / pandas). It implements cubic equations of state, phase equilibrium and
the full set of stream properties used for process design.

> **Status.** Compared with DWSIM 10.2.12.0 on **one mixture (H2/N2/NH3)**: Peng-Robinson at
> 175 temperature-pressure points and SRK at 163 points (200 to 1000 K, 1 to 200 bar, 64 of
> them two-phase), plus a detailed single-point comparison at 150 bar, and with CoolProp for the
> pure-component data it is built from. Agreement with DWSIM validates this implementation
> (same equation, constants and interaction parameters), not accuracy against experiment.
> Other systems, PRSV and equipment are not covered yet; see
> [Known limitations](#known-limitations).

```python
import numpy as np
from thermo_sim import METHODS, Stream, load_database

kij = np.array([[0.0,    0.0711, 0.0],        # H2, N2, NH3 (ordered by Tc)
                [0.0711, 0.0,    0.2193],
                [0.0,    0.2193, 0.0]])

feed = Stream(description='feed reactor', fm=None, comp_mass=None, fn=100.0,
              comp_mol={'hydrogen': 0.726, 'nitrogen': 0.242, 'ammonia': 0.032},
              V_frac=0.99,      # specification: 99 % of the moles in the vapor
              T=300.0,          # with V_frac given, T is only the initial guess of the solver
              P=150.0, Tref=298.15, Pref=1.01325, gamma=None,
              component_data=load_database(), kij=kij,
              method=METHODS['PR'].but(liquid_volume='poly'), emit_warnings=False)

feed.T                      # solved from the vapor-fraction specification
feed.props.H_total_mix      # kJ/kmol
feed.props.phi_v            # fugacity coefficients in the vapor phase
```

## Features

* **Equations of state:** Peng-Robinson, SRK (`srk`, Graboski-Daubert kappa, and `srk72`,
  Soave's original 1972 kappa as used by DWSIM) and RK, with interchangeable alpha functions
  (Soave/standard, Boston-Mathias, Mathias-Copeman, PRSV/iPRSV, Graboski-Daubert for H2)
  assignable per component, and ready-made presets (`METHODS`).
* **Phase equilibrium:** Michelsen stability test, TP flash, vapor-fraction/pressure
  flash, bubble and dew temperatures, pure-component saturation from the EOS spinodals.
  The liquid phase can use the EOS (phi-phi) or a gamma-phi model with a pluggable activity model.
* **Stream properties:** enthalpy, entropy and heat capacity split into ideal-gas,
  departure and total parts, for the mixture and for each pure component, per phase;
  fugacity coefficients, compressibility, density (four liquid-volume models),
  viscosity and thermal conductivity.
* **Stream algebra:** adiabatic mixing (`a + b`), subtraction, scaling, splitting, and
  PH flash (specify enthalpy instead of temperature).
* **Property database** built from CoolProp correlations (`scripts/build_db.py`) with a
  fit-quality report.
* **Unit operations:** isentropic and polytropic compressors, coolers, multistage trains
  with intercooling and a flash drum, built on `Stream` and tested against closed-form
  ideal-gas results.

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
  value (`R_GAS = 8.314462618` J/(mol K) in `constants.py`). Use `METHODS['PR'].but(R=8.314)`
  to reproduce a program that uses another value. Phase equilibrium does not depend on `R`;
  molar volumes, densities, entropies and the Rackett/Peneloux volumes do.
* Units: T [K], P [bar], molar flow [kmol/s], mass flow [kg/s], H [kJ/kmol],
  S and Cp [kJ/(kmol K)], density [kg/m3]. Lower-case `h`, `s`, `cp` are per kg.
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
The Rackett parameter Zra is taken from the user, then the database (`Z_RA`, fitted to
CoolProp saturated-liquid densities), then Zc.

## Validation

All comparisons are against DWSIM 10.2.12.0 with a H2/N2/NH3 feed (0.726 / 0.242 / 0.032). The
critical constants, acentric factors, binary interaction parameters and R = 8.314 are DWSIM's,
so what is compared is the implementation. The reference numbers and their provenance are in
[`reference_data/dwsim/`](reference_data/dwsim).

### Temperature-pressure grids

Stream sensitivity analyses exported from DWSIM, compared point by point with a TP flash in
thermo-sim ([`tests/test_dwsim_grid.py`](tests/test_dwsim_grid.py); run
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

Notes on what is compared:

* Fugacity coefficients are compared to 5e-6 on the rows exported with six digits. The 32 PR rows
  between 230 and 320 K come from a second export with three digits; they are compared at the
  resolution of the file (6e-3 relative).
* **Energy** is compared as X(T, P) - X(T, 1 bar). This removes the ideal-gas Cp, the formation
  enthalpy and the different reference states of the two programs.
* **Liquid density**: in both grids DWSIM used the EOS molar volume (no Peneloux), so the
  two-phase mixture density is compared as well, with thermo-sim's `eos` liquid-volume model.
* The `srk` form of thermo-sim (Graboski-Daubert kappa) is *not* the one DWSIM uses; it differs
  from DWSIM by up to 0.3 % in Z and 1.8 % in the vapor fugacity coefficients on this grid. Use
  `ThermoMethod('srk72', 'std')` (preset `METHODS['SRK72']`) to reproduce DWSIM.
* **DWSIM artifact, two-phase entropy.** In the two-phase region DWSIM's mixture entropy does
  not satisfy dH = T dS at constant pressure (the excess is about 1.8 R per mole of liquid; it
  appears in the 10 K rows at 100 to 200 bar), while its single-phase entropy does, to 1e-4 R.
  thermo-sim gives a difference of about 0.05 R on those 23 points per case, which are marked
  as expected failures. The check that shows which program is consistent does not depend on
  DWSIM: [`tests/test_isobaric_consistency.py`](tests/test_isobaric_consistency.py) integrates
  dH / T along isobars with thermo-sim's own flashes and finds the entropy consistent to
  7e-8 R, in both the two-phase and the single-phase region.

### Single point: flash at 150 bar

Peng-Robinson, vapor fraction 0.99 at 150 bar
([`tests/test_dwsim_ammonia_feed.py`](tests/test_dwsim_ammonia_feed.py)).

| Quantity | Difference |
|---|---:|
| Flash temperature (248.469 K) | 4e-4 K |
| Phase mole fractions x, y | below 6e-5 relative |
| Fugacity coefficients, vapor and liquid | below 2e-5 relative |
| Z, vapor and liquid | 5e-6 and 2e-7 absolute |
| Dew temperature of the feed (257.724 K) | 2e-4 K |
| Vapor density | 5e-5 relative |
| Liquid density, EOS root | within 5e-4 relative (test tolerance) |
| Residual enthalpy, feed / vapor / liquid | 0.06 / 0.03 / 1.4 kJ/kmol |

* The residual enthalpy is compared instead of the total because the ideal-gas part comes from
  different Cp data (DWSIM's correlation against a polynomial fitted to CoolProp); the test
  removes DWSIM's own ideal part using its published coefficients.
* Liquid density: DWSIM's COSTALD gives 662.4 kg/m3; the `poly` model is 0.4 % lower and
  `rackett` 0.3 % higher. These are different correlations of the same quantity.
* Not compared: Peneloux densities (DWSIM applies the shift to both phases, thermo-sim to the
  liquid only) and absolute entropy (different reference states).

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

The benchmark script is not part of the repository. This is an unoptimized pure-Python implementation. Profiling and vectorization (for example
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
├── activity.py            activity-coefficient models, gamma-phi liquid
├── properties.py          StreamProperties (result container)
├── stream_equilibrium.py  phase assignment and fugacities for a stream
├── stream.py              Stream: the stream object
├── database.py            load_database()
├── unitops/
│   ├── compressor.py      single-stage compressor (isentropic, polytropic)
│   ├── cooler.py          isobaric cooler/heater
│   ├── train.py           multistage compressor train with intercoolers
│   └── flash_drum.py      flash drum with saturated vapor and liquid outlets
└── data/                  property database, critical-constant overrides
scripts/
├── build_db.py            builds the database from CoolProp
└── compare_dwsim_grid.py  differences with the DWSIM grids (max / median per quantity)
tests/                     pytest suite (EOS, equilibrium, stream, unit operations,
                           DWSIM single point and grids, thermodynamic consistency)
reference_data/dwsim/      DWSIM reference results with provenance
examples/                  runnable examples
docs/                      fit quality of the property database
```

## Known limitations

* Validation covers one mixture (H2/N2/NH3) with PR and SRK. PRSV, RK and the unit operations
  (compressors, coolers, flash drum) have no external reference yet.
* Ideal-gas Cp is a polynomial fitted to CoolProp over 200 to 1000 K (maximum fit error
  0.11 %); outside that range it is extrapolated linearly and a warning is raised.
* Viscosity and thermal conductivity use simple mass-weighted mixing and are not yet
  validated.
* No Henry's law; the gamma-phi model takes user-supplied activity coefficients only.
* Bubble and dew temperatures of a liquid close to pure NH3 above its critical pressure
  return NaN (and some H2-rich mixtures too); this is covered by expected-failure tests.
* The volume shift of `eos_peneloux` is applied to the liquid only.

## Data sources and license

Pure-component correlations are fitted to [CoolProp](http://www.coolprop.org) (MIT
license). Formation enthalpy and Gibbs energy come from NIST (NIST-JANAF tables and the
NIST Chemistry WebBook); the `source` column of `data/raw/formation_data.csv` gives the
origin of each value. Released under the MIT license (see `LICENSE`).

[DWSIM](https://dwsim.org) is an independent open-source project. Its results are used here
only as reference numbers; no DWSIM code is used or distributed, and this project is not
affiliated with it.