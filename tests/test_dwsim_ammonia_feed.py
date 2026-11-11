"""thermo-sim against DWSIM 10.2.12.0 (Peng-Robinson): H2/N2/NH3 feed at 150 bar, VF = 0.99.

Reference numbers: reference_data/dwsim/ammonia_loop_feed_150bar.json (see its README).
The equation of state, constants, kij and R are DWSIM's, so what is compared is the
implementation: flash temperature, phase compositions, Z, fugacity coefficients, densities
and the residual enthalpy. Observed differences are written next to each tolerance (about
2x margin). Liquid density is compared with DWSIM's COSTALD only to within the difference
between two correlations of the same quantity.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from thermo_sim import METHODS, Stream, load_database
from thermo_sim.unitops import FlashDrum

REF_FILE = (Path(__file__).resolve().parents[1] / 'reference_data' / 'dwsim'
            / 'ammonia_loop_feed_150bar.json')
REF = json.loads(REF_FILE.read_text(encoding='utf-8'))
NAMES = REF['components']
INP, RES = REF['inputs'], REF['results']

# DWSIM reports Zra = 0 for ammonia; thermo-sim warns and falls back to the next source.
pytestmark = pytest.mark.filterwarnings(r'ignore:Zra \(user\)')


def _method(liquid_volume='poly'):
    return METHODS['PR'].but(constants_user=INP['constants'], R=REF['meta']['R_J_per_mol_K'],
                             liquid_volume=liquid_volume)


def _feed(liquid_volume='poly'):
    return Stream(
        description='feed', fm=None, comp_mass=None, fn=100.0,
        comp_mol=dict(zip(NAMES, INP['feed_mole_fractions'])), V_frac=INP['V_frac'], T=300.0,
        P=INP['P_bar'], Tref=INP['Tref_K'], Pref=1.01325, psat_factor=None, gamma=None,
        component_data=load_database(), kij=np.array(INP['kij']), k_mode='cubic',
        method=_method(liquid_volume), emit_warnings=False)


@pytest.fixture(scope='module')
def feed():
    return _feed('poly')


@pytest.fixture(scope='module')
def drum(feed):
    return FlashDrum(V_frac=INP['V_frac'], P=feed.P, T_tol=1e-3).run(feed)


# ---------------------------------------------------------------- equilibrium

def test_flash_temperature(feed):
    assert feed.T == pytest.approx(RES['T_K'], abs=2e-3)               # observed 4e-4 K


def test_phase_compositions(feed):
    p = feed.props
    assert p.xi == pytest.approx(RES['phase_mole_fractions']['liquid_x'], rel=2e-4)  # obs 6e-5
    assert p.yi == pytest.approx(RES['phase_mole_fractions']['vapor_y'], rel=2e-4)


def test_fugacity_coefficients(feed):
    p, s = feed.props, RES['phi_print_scale']
    assert p.phi_v == pytest.approx(np.array(RES['phi_vapor_as_printed']) * s, rel=1e-4)  # obs 1e-5
    assert p.phi_l == pytest.approx(np.array(RES['phi_liquid_as_printed']) * s, rel=1e-4)


def test_compressibility_factors(feed):
    assert feed.props.Z_gas == pytest.approx(RES['Z_vapor'], abs=2e-5)         # observed 5e-6
    assert feed.props.Z_liq_eos == pytest.approx(RES['Z_liquid'], abs=2e-6)    # observed 2e-7


def test_dew_temperatures(feed, drum):
    assert feed.props.TDew == pytest.approx(RES['dew_T_feed_K'], abs=5e-3)     # observed 2e-4 K
    assert drum.vapor.props.TDew == pytest.approx(RES['dew_T_vapor_outlet_K'], abs=2e-3)


@pytest.mark.xfail(strict=False, reason=(
    "Known limitation: the liquid outlet is nearly pure NH3 above its critical pressure, so "
    "its phase envelope is very thin and TBubble comes back as NaN."))
def test_bubble_temperature_of_the_liquid_outlet(drum):
    assert drum.liquid.props.TBubble == pytest.approx(RES['bubble_T_liquid_outlet_K'], abs=0.05)


# ---------------------------------------------------------------- densities

@pytest.mark.parametrize('model', ['eos', 'eos_peneloux', 'rackett', 'poly'])
def test_vapor_density_does_not_depend_on_the_liquid_model(model):
    rho = _feed(model).props.dens_mix_gas
    assert rho == pytest.approx(RES['density_kg_per_m3']['eos']['VAPOR'], rel=1e-4)  # obs 5e-5


def test_eos_densities_match_dwsim():
    p = _feed('eos').props
    ref = RES['density_kg_per_m3']['eos']
    assert p.dens_mix == pytest.approx(ref['FEED'], rel=2e-4)
    assert p.dens_mix_liq == pytest.approx(ref['LIQUID'], rel=5e-4)


@pytest.mark.parametrize('model', ['poly', 'rackett'])
def test_liquid_density_close_to_dwsim_costald(model):
    """Different correlations of the same quantity: observed poly -0.39 %, rackett +0.33 %."""
    p = _feed(model).props
    ref = RES['density_kg_per_m3']['costald']
    assert p.dens_mix_liq == pytest.approx(ref['LIQUID'], rel=6e-3)
    assert p.dens_mix == pytest.approx(ref['FEED'], rel=2e-4)


# ---------------------------------------------------------------- enthalpy

def _dwsim_ideal_enthalpy(z, T, Tref=298.15, n=64):
    """sum z_i * integral_Tref^T Cp_i dT with DWSIM's ideal-gas Cp [kJ/(kmol K)]."""
    c = INP['ideal_gas_cp']
    A, B, C, D, E = (np.array(c[k]) for k in 'ABCDE')
    x, w = np.polynomial.legendre.leggauss(n)
    t = 0.5 * (T - Tref) * x + 0.5 * (T + Tref)
    cp = (A[:, None] + np.exp(B[:, None] / t + C[:, None] + D[:, None] * t
                              + E[:, None] * t * t)) / 1000.0
    return 0.5 * (T - Tref) * float(np.asarray(z, float) @ (cp @ w))


def test_dwsim_ideal_enthalpy_reproduces_the_recorded_integral():
    """Guards the transcription of the Cp coefficients: pure H2 at 248.469 K."""
    h = _dwsim_ideal_enthalpy([1.0, 0.0, 0.0], RES['T_K'])
    assert -1500.0 < h < -1300.0                      # about -1.46e3 kJ/kmol for 50 K of cooling


@pytest.mark.parametrize('stream,tol', [('FEED', 0.15), ('VAPOR', 0.15), ('LIQUID', 2.5)])
def test_residual_enthalpy_matches_dwsim(feed, drum, stream, tol):
    """Residual = H(formation-free) - ideal. Removing DWSIM's own ideal part isolates the EOS.

    Observed differences: feed 0.06, vapor 0.03, liquid 1.4 kJ/kmol (of 5e2 and 2e4).
    """
    s = {'FEED': feed, 'VAPOR': drum.vapor, 'LIQUID': drum.liquid}[stream]
    h_dwsim = RES['molar_enthalpy_formation_free_kJ_per_kmol'][stream]
    dep_dwsim = h_dwsim - _dwsim_ideal_enthalpy(s.z, RES['T_K'], INP['Tref_K'])
    assert s.props.H_dep_mix == pytest.approx(dep_dwsim, abs=tol)
