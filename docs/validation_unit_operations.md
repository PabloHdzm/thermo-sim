# Validation of unit operations

This document records how the unit operations of `thermo-sim` (`thermo_sim.unitops`) were
compared with DWSIM 10.2.12.0, what agrees, what does not and why. Every number below is
checked by a test; the files are listed in section 11.

> **Scope.** Compressor, cooler, flash drum, valve, equilibrium, stoichiometric and Gibbs
> reactors, heat exchanger and recycle solver are compared with DWSIM flowsheets (ammonia
> synthesis loop, propane refrigeration cycle, water-gas shift, methanol synthesis, natural
> gas dew-point control, and the DWSIM heat-exchanger and recycle tutorials). The
> multistage compressor train has no external reference because DWSIM has no equivalent
> block; it is tested against closed-form ideal-gas results. Distillation columns are in
> [`validation_distillation.md`](validation_distillation.md).
> Agreement with DWSIM validates this implementation (same equation of state, constants and
> conventions), not accuracy against experiment.

## 1. How the comparisons are made

* **Same thermodynamics on both sides.** Peng-Robinson with the critical constants, acentric
  factors and binary parameters that DWSIM displays for each case, and `R = 8.314` J/(mol K)
  (DWSIM's value; the default of the library is CODATA, see the README).
* **Tolerances** are two to five times the observed difference, limited by the digits that
  DWSIM prints. Where a difference has a known cause, the test records it instead of widening
  the tolerance silently.
* **Enthalpy.** DWSIM's displayed molar enthalpy excludes the enthalpy of formation, its
  *energy flow* includes it. `H_total_mix` in thermo-sim includes it, so duties and energy
  flows are compared directly and displayed enthalpies as differences.
* **Ideal-gas heat capacity.** DWSIM uses `Cp = A + exp(B/T + C + D T + E T^2)`; the library
  database holds a polynomial fitted to CoolProp. They differ by 0.3 to 0.5 % for the gases of
  these flowsheets. Where energy matters, the tests run twice: with DWSIM's equation fitted
  to a quartic over the working range (the comparison that isolates the unit operation) and
  with the packaged database (the difference a user would see). Phase equilibrium does not
  depend on the ideal-gas Cp, so compositions and phase splits agree in both runs.
* **Entropy** is not compared in two-phase streams unless stated: DWSIM's two-phase entropy
  does not satisfy dH = T dS at constant pressure (see the README), and its mixing term
  follows an empirical rule ([`validation_dwsim_entropy_mixing.md`](validation_dwsim_entropy_mixing.md)).

## 2. Models

* **Compressor.** Isentropic mode finds the ideal outlet by a PS flash and applies
  `dH = dH_is / eta`. Polytropic mode integrates `dH = V dP / eta_p` along the path in pressure
  steps (Heun predictor-corrector in ln P). Power is the indicated (fluid enthalpy) power.
* **Cooler / heater.** Isobaric to a given outlet temperature or vapor fraction, optional
  pressure drop; duty is `fn (H_out - H_in)`, positive when heat is added.
* **Valve.** Isenthalpic. For a pure component inside the dome the PH flash cannot fix the
  temperature, so the valve solves the vapor fraction at the saturation temperature.
* **Flash drum.** Flashes at the drum conditions and builds saturated vapor and liquid
  outlets. Specifications: temperature, duty or vapor fraction.
* **Reactors.** Stoichiometric (fixed conversion), equilibrium (one reaction, `K(T)` from
  formation data or a user expression, fugacity coefficients of the outlet, one vapor phase)
  and Gibbs (element balance, no reaction needed). Isothermal or adiabatic.
* **Heat exchanger.** Specifications by UA, duty, effectiveness, either outlet temperature or
  either outlet vapor fraction, counter-current or parallel flow, with an LMTD correction
  factor; effectiveness-NTU and LMTD relations; the temperature profile is
  integrated to report the pinch and a profile UA.
* **Recycle.** Successive substitution with optional acceleration and smoothing, convergence
  on temperature, pressure and mass flow as in DWSIM.

## 3. Ammonia synthesis loop

DWSIM flowsheet: feed (300 K, 30 bar, 4 mol/s, H2 0.75 / N2 0.25) -> compressor (200 bar,
adiabatic efficiency 0.75) -> heater (700 K) -> isothermal equilibrium reactor (reaction
`ln K = 11000/T - 25`, N2 conversion 48.8758 %) -> cooler (250 K) -> cooler (250 K) -> flash
drum. Inside-Out flash, liquid volume from the EOS. DWSIM was run with reactor tolerances of
1e-9 (with its defaults it stops 0.1 to 0.2 conversion points early).

With DWSIM's ideal-gas Cp:

| Quantity | Difference |
|---|---|
| Compressor outlet temperature | 0.019 K |
| Compressor power, reactor duty, cooler duty | below 1e-4 relative |
| Heater duty | 2.4e-4 relative |
| Z, density, composition, every stream | 5e-6 (flash phases 2e-5) |

With the packaged database Cp the duties differ by 0.15 to 0.9 % and the compressor outlet by
+0.56 K, while the cooler vapor fraction is unchanged: the whole difference is the ideal-gas
Cp. A fixed-conversion reactor at DWSIM's converged conversion reproduces its outlet and duty,
which checks the heat of reaction separately from the equilibrium solver.

## 4. Propane refrigeration cycle

DWSIM flowsheet: saturated vapor (2.45 bar, 1 mol/s) -> compressor (12.5 bar, adiabatic
efficiency 0.8) -> condenser (305 K) -> isenthalpic valve (2.45 bar) -> evaporator (253 K).
DWSIM results: work 4.20275 kW, condenser 15.9372 kW, evaporator 11.7348 kW, COP 2.79.

* Compressor power, condenser and evaporator duty agree with DWSIM to 0.1 to 0.15 %, and the
  compressor temperature rise to 72.141 K against 72.143 K.
* The liquid root at the condenser outlet and the vapor root at the evaporator inlet agree to
  1e-6 in Z, so the equation of state is the same.
* The valve conserves enthalpy to solver precision and the energy balance of the cycle closes.

**Known difference: where DWSIM puts the saturation point of a pure component.** For a pure
compound DWSIM places the saturation points with the vapor-pressure correlation of the
compound, not with the equation of state:

| T [K] | correlation [bar] | DWSIM | EOS [bar] |
|---|---|---|---|
| 252.995 | 2.45004 | 2.45 | 2.4302 |
| 305 | 11.37562 | 11.3756 | 11.3023 |
| 325.138 | 17.96016 | 17.9601 | 17.9583 |

Its saturation temperature at 2.45 bar is therefore 252.995 K, against 253.221 K from the
equal-fugacity condition of the EOS (confirmed by an independent solution of the cubic).
Streams that start or end on the dome inherit the 0.226 K shift; those comparisons are
`xfail`. The evaporator is taken to saturated vapor (DWSIM heats to 253 K, 0.005 K above its
own dew point). The shift cancels in the temperature rise of the compressor.

## 5. Reactors

### Equilibrium reactor (ammonia and water-gas shift)

Two conventions of the DWSIM equilibrium reactor explain the difference with the library, found
with the ammonia synthesis and confirmed with the water-gas shift:

* **Standard-state pressure P0 = 1 atm.** With a constant `ln K`, `ln Q - ln K` (Q on a 1 bar
  basis) is -0.02632, -0.02630 and -0.02634 for `ln K` = -9.2857, -5 and -12, and
  `2 ln(1.01325)` is 0.02633. With P0 = 1.01325 bar the residual is 3e-5 or less.
* **The expression `ln K(T)` is evaluated at T + 1 K.** The temperature for which
  `11000 / T' - 25` reproduces the `ln K` DWSIM used is 501.0000, 600.9993, 701.0006 and
  801.0004 K at 500, 600, 700 and 800 K. On the water-gas shift at 600 K DWSIM gives
  83.7935 % conversion, against 83.887 % with K at T and 83.801 % with K at T + 1. A constant
  `ln K` is not affected. This looks like an implementation detail of DWSIM, not a
  thermodynamic effect.

With both conventions the ammonia conversions agree to 0.01 conversion points or better
(equation mode, 500 to 800 K). In Gibbs mode (K from formation data) DWSIM shows no T + 1 K
shift; with P0 = 1 atm the library is within +0.12, +0.06, +0.005 and -0.045 conversion points
at 500, 600, 700 and 800 K, and the remaining difference is not identified. The formation Gibbs
energy of NH3 differs between the database (-16.37 kJ/mol, NIST, 1 bar) and DWSIM (-16.40
kJ/mol, DIPPR, 1 atm) by the standard-state term.

### Water-gas shift, three reactors

DWSIM "Reaction Systems" tutorial: 10 mol/s at 600 K and 10.1325 bar, CO/H2O/CO2/H2 =
40/40/10/10 mol %, three isothermal reactors.

| Reactor | Compared | Tolerance |
|---|---|---|
| Conversion (80 % of CO) | flows, duty, outlet energy flow | flows 1e-9, duty 2e-3 relative, energy 0.3 kW |
| Equilibrium (`ln K = 4577.8/T - 4.33`) | composition, extent, conversion, duty | 5e-4, 0.01 mol/s, 0.2 points, 2e-3 |
| Gibbs (element balance only) | composition, conversions, duty | 1e-3, 0.3 points, 3e-3 |

Adiabatic versions of the three reactors reproduce DWSIM's outlet temperature within 0.1 K
(conversion), 0.2 K (equilibrium) and 0.5 K (Gibbs), and the energy balance closes to 1e-6 kW.
The Gibbs reactor and the equilibrium reactor with the same thermochemistry give the same
state to 1e-8. DWSIM's own equilibrium product sits 0.0069 in `ln K` away from its expression
(it stops slightly early), which sets the achievable agreement.

### Methanol synthesis (single pass)

DWSIM "Methanol Synthesis" tutorial: syngas (3 mol/s, H2:CO = 2:1, 300 K, 30 bar) -> compressor
(adiabatic, 75 %, 80 bar) -> heater (525 K) -> equilibrium reactor (isothermal 525 K,
`ln K = 11833/T - 29.07`) -> cooler (308 K) -> separator.

| Block | Compared | Tolerance |
|---|---|---|
| Compressor | outlet temperature, power | 0.2 K, 1e-3 relative |
| Compressor, polytropic mode | DWSIM's polytropic efficiency reproduced | 0.1 K, efficiency 1e-3 |
| Heater | duty | 0.03 kW |
| Reactor, whole chain | extent, duty, CO conversion | 2 %, 2 %, 1.5 points |
| Cooler and separator on the DWSIM reactor outlet | vapor fraction | 2e-4 |
| | outlet compositions | 2e-5 |
| | molar flow, mass flow | 1e-4, 2e-4 |
| | energy flows | 0.1 kW |

The whole chain accumulates small differences: the equilibrium extent is 1.1 % above DWSIM's.
Running the cooler and the separator on the composition DWSIM reports at the reactor outlet
separates the thermodynamics of the cooled stream from that accumulated difference. DWSIM's
separator report has its two mass-flow labels swapped; the test documents it.

## 6. Heat exchanger

DWSIM "Heat Exchanger Design" tutorial, water on both sides (IAPWS-IF97 in DWSIM, gamma-phi route
in thermo-sim). Closed-form checks come first and need no database: with two incompressible
liquids of constant Cp the effectiveness-NTU relations are exact, and the UA, effectiveness,
duty and outlet-temperature specifications must round-trip through UA to 1e-9.

Against DWSIM:

| Case | Quantity | Tolerance |
|---|---|---|
| UA specified | outlet temperatures | 0.35 K |
| | heat load | 2e-3 relative |
| | LMTD | 0.1 K |
| | thermal efficiency | 0.25 points |
| Maximum duty (efficiency 1) | duty, hot outlet, cold outlet | 1e-2 relative, 1e-6 K, 0.6 K |
| Outlet vapor fraction of the hot or the cold side (condenser, evaporator) | saturation temperature, other outlet | 0.3 K, 1 K |
| | heat load, LMTD, UA | 1 %, 1 %, 1.5 % |

The database liquid Cp of water is about 0.6 % above IAPWS-IF97 over 300 to 400 K (0.63 %
observed on the maximum duty), and the gamma-phi enthalpy of vaporization is about 0.3 % above
IF97 at 1 atm; these set the 1 % tolerances. Outlet pressures are exact. Phase change inside
the exchanger is flagged in `warnings`, and the solver needs fewer than 100 stream evaluations
for a UA specification.

## 7. Recycle solver

DWSIM "Recycle Loops" tutorial (water): fresh feed + recycle -> heater (350 K) -> 70/30 split
-> recycle. The flowsheet is built only with stream algebra (`+`, `split`) and a `Cooler`.

* The solver has a known fixed point: the mixed stream is the fresh flow divided by 0.7, the
  product equals the fresh flow, and the recycle is 0.3/0.7 of it. Mass flows match the closed
  form to 1e-6.
* Against DWSIM: mass flows 1e-4 kg/s, molar flows 1e-4 relative, mixed temperature 0.1 K,
  heater duty 1.2 % (the liquid Cp, as in section 6), constant pressure, all streams liquid.
* It converges with DWSIM's tolerances in fewer than 15 iterations, and every acceleration
  method gives the same loop to 1e-6.
* Separate tests on loops with a known solution cover the linear and the coupled nonlinear
  case, the damping of slow loops, the report of non-convergence and invalid settings.

## 8. Natural gas dew-point control

DWSIM tutorial: wellhead gas (C1 to C5, 100 mol/s, 70 bar) -> chiller (320 K to 240 K) ->
separator. DWSIM constants and binary parameters of the same mixture as the phase envelope.

| Quantity | Tolerance (observed) |
|---|---|
| Chiller duty | 0.6 % |
| Vapor fraction | 0.002 |
| Outlet molar flows (sales gas, NGL) | 0.1 mol/s |
| NGL composition | 5e-4 (1.6e-5 observed) |
| Sales gas composition | 2e-4 |
| C3+ recovery in the NGL | 2e-3 |
| Feed dew temperature | 0.1 K |
| Feed bubble temperature | 1 K |

The adiabatic separator closes to 1e-6 kW and the entropy change across the chiller equals the
integral of dH/T to 2e-3. The DWSIM entropy change is reproduced with the empirical mixing rule
(0.15 tolerance, treated as a hypothesis).

## 9. Closed-form and internal checks (no external reference)

* **Compressor.** Isentropic and polytropic outlets match the closed forms of an ideal gas
  (`T2 = T1 (P2/P1)^((gamma-1)/gamma)` and the polytropic equivalent); the polytropic
  integration converges with the number of steps; liquid and two-phase feeds, expansions and
  invalid arguments are rejected.
* **Multistage train.** Equal pressure ratio per stage, coolers after any chosen stage,
  aftercooler and cooler pressure drop, all against closed-form ideal-gas results.
* **Flash drum.** Control flow with a fake stream (constant-K Rachford-Rice, constant Cp, no
  database), and consistency of the three specification modes on a reactor-effluent feed.

These checks validate the logic and the assembly, not agreement with a simulator.

## 10. Reading the results

* **The equation of state and the flashes agree at the level of the printed digits** in every
  flowsheet: Z and density 5e-6, compositions 2e-5 to 5e-4, vapor fractions 2e-5 to 2e-3.
* **Energy differences are the ideal-gas Cp.** With DWSIM's own Cp equation, duties and powers
  agree to 1e-4 to 2.4e-4; with the packaged CoolProp-based Cp they differ by 0.15 to 0.9 %
  and a heated or compressed gas by 0.2 to 0.6 K. The packaged-database tests record this as
  a known, understood difference and keep the energy balances under regression.
* **Differences caused by DWSIM conventions** are documented where they appear: the saturation
  point of a pure component (section 4), the standard-state pressure and the T + 1 K
  evaluation of `ln K` in the equilibrium reactor (section 5), and the entropy rules of
  section 1.
* **Water** goes through the gamma-phi route, whose liquid Cp is 0.6 % above IAPWS-IF97; every
  water tolerance (sections 6 and 7) is set by that number.

## 11. Limitations and next steps

* The multistage compressor train has no external reference. A cascade of DWSIM compressor and
  cooler blocks with the intermediate pressures computed by thermo-sim would validate it, and
  is on the roadmap.
* Compressors take vapor feeds only; efficiencies are constant (no performance curves) and
  there is no surge or choke check. Only isentropic and polytropic efficiency definitions are
  implemented.
* Equilibrium reactors handle one reaction in a single vapor phase; the Gibbs reactor has the
  same limit. Kinetic and multiphase reactors are not implemented.
* The heat exchanger is validated with water only; phase change is reported but not resolved
  along the exchanger.
* The flash drum is validated through the flowsheets above; the liquid outlet of an almost
  pure NH3 stream above its critical pressure has no bubble or dew temperature (expected
  failure).
* Viscosity, thermal conductivity and pressure drop correlations are not validated.

## 12. Reproducing

```bash
pip install -e ".[dev]"
pytest tests/test_dwsim_loop.py tests/test_dwsim_cycle.py tests/test_equilibrium_reactor.py \
       tests/test_wgs_reactors.py tests/test_methanol_synthesis.py tests/test_heat_exchanger.py \
       tests/test_recycle_loop.py tests/test_natgas_dew_point_control.py -v
pytest tests/test_compressor.py tests/test_train.py tests/test_flash_drum.py -v   # closed forms
```

The reference data and their provenance are in
[`reference_data/dwsim/`](../reference_data/dwsim). DWSIM is an independent open-source
project; its results are used here only as reference numbers, and no DWSIM code is used or
distributed.
