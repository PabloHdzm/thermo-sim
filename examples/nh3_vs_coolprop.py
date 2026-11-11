"""Pure NH3: thermo-sim (PR) against CoolProp (reference EOS) along the saturation curve.

Saturation pressure: for each temperature CoolProp gives Psat. thermo-sim solves the
saturation temperature at that pressure (bubble point of the pure fluid). If thermo-sim
needs a higher temperature than CoolProp to reach the same pressure, its Psat at the
given T is lower, so  Psat error = -dT_sat * dlnPsat/dT  (dlnPsat/dT from CoolProp).

Densities: compared at the SAME (T, P) in CoolProp and thermo-sim, with the vapor 3 %
below and the liquid 3 % above CoolProp's Psat so that both programs are single-phase.

Requires CoolProp (pip install -e ".[db]").
Usage:  python examples/nh3_vs_coolprop.py
"""
import math

import CoolProp.CoolProp as CP

from thermo_sim import METHODS, Stream, load_database

FLUID = 'Ammonia'
DB = load_database()
METHODS_TESTED = {
    'PR, eos liquid volume': METHODS['PR'],
    'PR, poly liquid volume': METHODS['PR'].but(liquid_volume='poly'),
}
MARGIN = 0.03


def stream(P, T, V_frac, method):
    return Stream(description='NH3', fm=None, comp_mass=None, fn=1.0,
                  comp_mol={'ammonia': 1.0}, V_frac=V_frac, T=T, P=P, Tref=298.15,
                  Pref=1.01325, gamma=None, component_data=DB, kij=None, method=method,
                  emit_warnings=False)


for label, method in METHODS_TESTED.items():
    print(f"\n{label}")
    print(f"{'T [K]':>7} {'Psat CP [bar]':>14} {'dT_sat [K]':>11} {'Psat err':>9} "
          f"{'rho_V err':>10} {'rho_L err':>10}")
    for T in (230, 250, 270, 290, 310, 330, 350, 370, 390, 400):
        P = CP.PropsSI('P', 'T', T, 'Q', 0, FLUID) / 1e5
        dlnP_dT = (math.log(CP.PropsSI('P', 'T', T + 0.1, 'Q', 0, FLUID))
                   - math.log(CP.PropsSI('P', 'T', T - 0.1, 'Q', 0, FLUID))) / 0.2
        try:
            dT = stream(P, T, 0.0, method).T - T
            Pv, Pl = P * (1 - MARGIN), P * (1 + MARGIN)
            ev = stream(Pv, T, None, method).props.dens_mix / CP.PropsSI(
                'D', 'T', T, 'P', Pv * 1e5, FLUID) - 1
            el = stream(Pl, T, None, method).props.dens_mix / CP.PropsSI(
                'D', 'T', T, 'P', Pl * 1e5, FLUID) - 1
            print(f"{T:7.1f} {P:14.4f} {dT:11.3f} {-dT * dlnP_dT:9.2%} {ev:10.2%} {el:10.2%}")
        except Exception as e:  # report and continue
            print(f"{T:7.1f} {P:14.4f}   failed: {type(e).__name__}: {e}")