"""H2/N2/NH3 feed at 150 bar and vapor fraction 0.99, with the constants and kij that
DWSIM shows for Peng-Robinson. Compare with reference_data/dwsim/ and
tests/test_dwsim_ammonia_feed.py."""
import numpy as np
from thermo_sim import METHODS, Stream, load_database

# Binary interaction parameters, components ordered by Tc: H2, N2, NH3
KIJ_PR = np.array([[0.0,    0.0711, 0.0],
                   [0.0711, 0.0,    0.2193],
                   [0.0,    0.2193, 0.0]])

# thermo-sim units: Tc [K], Pc [bar], omega [-], Vc [cm3/mol], Zra [-]
# DWSIM reports Zra = 0 (not set) for ammonia, so it is left out: the database value is used.
CONSTANTS = {
    'hydrogen': dict(Tc=33.19, Pc=13.13, omega=-0.215993, Vc=64.147, Zra=0.3139),
    'nitrogen': dict(Tc=126.2, Pc=33.98, omega=0.037, Vc=90.1, Zra=0.2906),
    'ammonia':  dict(Tc=405.4, Pc=113.53, omega=0.256, Vc=72.47),
}

method = METHODS['PR'].but(liquid_volume='poly', constants_user=CONSTANTS, R=8.314)

feed = Stream(
    description='feed reactor', fm=None, comp_mass=None,
    fn=100,                                           # kmol/s
    comp_mol={'hydrogen': 0.726, 'nitrogen': 0.242, 'ammonia': 0.032},
    V_frac=0.99,                                      # VF-P flash: T is solved
    T=300.0,                                          # initial guess
    P=150.0,                                          # bar
    Tref=298.15, Pref=1.01325, psat_factor=None, gamma=None,
    component_data=load_database(), kij=KIJ_PR, method=method, k_mode='cubic',
    emit_warnings=False)

p = feed.props
print(f"T = {feed.T:.4f} K   phase = {p.phase}   vapor fraction = {p.V_frac:.4f}")
print(f"vapor  y = {np.round(p.yi, 6)}")
print(f"liquid x = {np.round(p.xi, 6)}")
print(f"H  total  (mix / vapor / liquid) = {p.H_total_mix:.2f} / {p.H_total_mix_gas:.2f} / {p.H_total_mix_liq:.2f} kJ/kmol")
print(f"S  total  (mix / vapor / liquid) = {p.S_total_mix:.4f} / {p.S_total_mix_gas:.4f} / {p.S_total_mix_liq:.4f} kJ/(kmol K)")
print(f"Cp total  (mix / vapor / liquid) = {p.Cp_total_mix:.3f} / {p.Cp_total_mix_gas:.3f} / {p.Cp_total_mix_liq:.3f} kJ/(kmol K)")
print(f"density   (mix / vapor / liquid) = {p.dens_mix:.2f} / {p.dens_mix_gas:.2f} / {p.dens_mix_liq:.2f} kg/m3")
print(f"Z (vapor / liquid) = {p.Z_gas:.4f} / {p.Z_liq_eos:.4f}")
print(f"phi vapor  = {p.phi_v}")
print(f"phi liquid = {p.phi_l}")
