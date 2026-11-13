"""EquilibriumReactor: thermodynamic consistency of K(T) and of the solution, plus the ammonia
synthesis reactor of the DWSIM loop (equilibrium reactor, 700 K, 200 bar, 48.8008 % conversion of N2).

The consistency tests need no DWSIM data: K(T) must reproduce the tabulated Gibbs energy of
formation at 298.15 K and obey the van't Hoff equation, and the solution must satisfy
prod (phi_i y_i P / P0)^nu_i = K with the fugacity coefficients of the outlet stream.
"""
import numpy as np
import pytest

from thermo_sim import load_database
from thermo_sim.unitops import EquilibriumReactor, reaction_thermo

from _dwsim_loop import SYNTHESIS, UNITS, loop

pytestmark = pytest.mark.filterwarnings(r"ignore:Zra \(user\):UserWarning")

R_GAS = 8.314


def _lnK_residual(res):
    """sum nu ln(phi_i y_i P/P0) - ln K at the solution."""
    out = res.outlet
    names = out.component_data['Component'].astype(str).str.lower().tolist()
    phi = dict(zip(names, np.asarray(out.props.phi_v, float)))
    tot = sum(res.n_out.values())
    r = -res.ln_K
    for c, nu in SYNTHESIS.items():
        r += nu * np.log(phi[c] * res.n_out[c] / tot * res.P_out)
    return r


def test_gibbs_energy_at_reference_temperature():
    db = load_database()
    th = reaction_thermo(db, SYNTHESIS, 298.15, R_GAS)
    gf = db.assign(c=db['Component'].astype(str).str.lower()).set_index('c')['gf_298_k']
    expected = sum(nu * gf[c] for c, nu in SYNTHESIS.items()) * 1000.0
    assert th['dG0'] == pytest.approx(expected, rel=1e-12)
    assert th['ln_K'] == pytest.approx(-expected / (R_GAS * 298.15), rel=1e-12)


@pytest.mark.parametrize('T', [400.0, 500.0, 700.0, 900.0])
def test_vant_hoff_consistency(T):
    """d ln K / dT = dH0 / (R T^2) with dH0 from the same Cp polynomial."""
    db, h = load_database(), 0.01
    lnK = lambda t: reaction_thermo(db, SYNTHESIS, t, R_GAS)['ln_K']
    dlnK = (lnK(T + h) - lnK(T - h)) / (2 * h)
    dH = reaction_thermo(db, SYNTHESIS, T, R_GAS)['dH0']
    assert dlnK == pytest.approx(dH / (R_GAS * T**2), rel=1e-6)


def test_solution_satisfies_equilibrium_condition():
    hot = loop()['streams']['hot']
    res = EquilibriumReactor(SYNTHESIS).run(hot)
    assert abs(_lnK_residual(res)) < 1e-8
    assert abs(res.ln_K_phi + res.ln_K_y - res.ln_K) < 1e-8
    assert 0.0 < res.conversion < 1.0
    assert res.duty_kW < 0.0                                  # exothermic


def test_reaction_runs_backwards_from_the_products():
    """Feeding the equilibrium composition (or pure NH3) returns the same equilibrium state."""
    hot = loop()['streams']['hot']
    fwd = EquilibriumReactor(SYNTHESIS).run(hot)
    back = EquilibriumReactor(SYNTHESIS).run(fwd.outlet)
    assert back.extent == pytest.approx(0.0, abs=1e-9 * hot.fn)
    for c in fwd.n_out:
        assert back.n_out[c] == pytest.approx(fwd.n_out[c], rel=1e-8, abs=1e-14)
    # the same atoms as pure ammonia (the feed is 3 H2 : 1 N2) give the same equilibrium
    n_nh3 = 2.0 * hot.fn * hot.comp_mol['nitrogen']
    pure = EquilibriumReactor(SYNTHESIS).run(hot.at(comp_mol={'ammonia': 1.0}, fn=n_nh3))
    assert pure.extent < 0.0
    for c in fwd.n_out:
        assert pure.n_out[c] == pytest.approx(fwd.n_out[c], rel=1e-7, abs=1e-14)


def test_conversion_increases_with_pressure_and_decreases_with_temperature():
    hot = loop()['streams']['hot']
    base = EquilibriumReactor(SYNTHESIS).run(hot).conversion
    assert EquilibriumReactor(SYNTHESIS).run(hot.at(P=250.0)).conversion > base
    assert EquilibriumReactor(SYNTHESIS, T_out=650.0).run(hot).conversion > base
    assert EquilibriumReactor(SYNTHESIS, T_out=750.0).run(hot).conversion < base


def test_no_room_for_reaction_raises():
    hot = loop()['streams']['hot']
    only_h2 = hot.at(comp_mol={'hydrogen': 1.0})
    with pytest.raises(ValueError):
        EquilibriumReactor(SYNTHESIS).run(only_h2)


def test_user_lnK_and_reference_pressure_are_equivalent():
    """prod (f/P0)^nu = K  is  prod f^nu = K P0^sum(nu): P_ref = a equals shifting ln K by sum(nu) ln a."""
    hot = loop()['streams']['hot']
    f = lambda T: 11000.0 / T - 25.0
    a = 1.01325
    r1 = EquilibriumReactor(SYNTHESIS, lnK=f, P_ref=a).run(hot)
    r2 = EquilibriumReactor(SYNTHESIS, lnK=lambda T: f(T) - 2.0 * np.log(a)).run(hot)
    assert r1.conversion == pytest.approx(r2.conversion, rel=1e-9)
    assert r1.ln_K == pytest.approx(f(700.0))


def test_partial_pressure_basis_has_no_fugacity_coefficients():
    """With phi = 1 the solution is the ideal-gas equilibrium: no dependence on the EOS."""
    hot = loop()['streams']['hot']
    res = EquilibriumReactor(SYNTHESIS, lnK=lambda T: 11000.0 / T - 25.0, fugacity=False).run(hot)
    assert res.ln_K_phi == 0.0
    assert res.conversion == pytest.approx(0.465629, abs=1e-5)


# DWSIM 10.2.12.0 equilibrium reactor, isothermal, 3 H2 : 1 N2 feed, converged (internal and external
# tolerances 1e-9; with the default 1e-3 / 1e-2 DWSIM stops 0.1 to 0.2 points early and the result
# depends on the starting point). Conversion of N2 in two modes:
#   gibbs:    K(T) from the Gibbs energy of formation (DWSIM compound data, 1 atm standard state)
#   equation: ln K = 11000 / T - 25 (reaction set of the tutorial)
# DWSIM uses the standard-state pressure P0 = 1 atm (101325 Pa) in the reaction quotient, so the
# comparison uses P_ref = 1.01325 bar. In equation mode DWSIM also evaluates the expression
# ln K(T) at T + 1 K (found on the ammonia reaction: the temperature that reproduces its conversions
# is T + 1.000 K at 500, 600, 700 and 800 K); the test evaluates the expression at T + 1 to match.
# The water-gas shift at 600 K confirms it (83.7935 % with tolerance 1e-9; 83.801 % predicted with
# K at T + 1, 83.887 % with K at T), but CO is not in the database, so it is not a test.
# Gibbs mode (K from the formation data) is compared as is; the remaining 0.04 to 0.14 points are
# not identified.
DWSIM_EQ = [
    # T [K], P [bar], mode, DWSIM conversion of the base component, base component, feed
    (500.0, 1.0, 'gibbs', 0.154957, 'nitrogen', '3:1'),
    (500.0, 1.0, 'equation', 0.116174, 'nitrogen', '3:1'),
    (600.0, 200.0, 'gibbs', 0.748017, 'nitrogen', '3:1'),
    (600.0, 200.0, 'equation', 0.726903, 'nitrogen', '3:1'),
    (700.0, 200.0, 'gibbs', 0.48463, 'nitrogen', '3:1'),
    (700.0, 200.0, 'equation', 0.488759, 'nitrogen', '3:1'),
    (800.0, 200.0, 'gibbs', 0.260928, 'nitrogen', '3:1'),
    (800.0, 200.0, 'equation', 0.292661, 'nitrogen', '3:1'),
]
TOL_CONV_DWSIM = {'equation': 2e-5, 'gibbs': 2e-3}      # observed: equation <= 1e-6 (DWSIM prints 6 digits), Gibbs <= 1.4e-3


def _dwsim_case(T, P, mode, base, feed):
    hot = loop()['streams']['hot']
    inlet = hot.at(T=T, P=P) if feed == '3:1' else hot.at(
        T=T, P=P, comp_mol={'hydrogen': 1 / 3, 'nitrogen': 1 / 3, 'ammonia': 1 / 3})
    kw = {} if mode == 'gibbs' else dict(lnK=lambda t: 11000.0 / (t + 1.0) - 25.0)   # DWSIM: T + 1 K
    res = EquilibriumReactor(SYNTHESIS, T_out=T, P_ref=1.01325, **kw).run(inlet)
    return 1.0 - res.n_out[base] / res.n_in[base]


@pytest.mark.parametrize('T,P,mode,ref,base,feed', DWSIM_EQ,
                         ids=[f"{c[0]:.0f}K_{c[1]:g}bar_{c[2]}" for c in DWSIM_EQ])
def test_equilibrium_conversion_vs_dwsim(T, P, mode, ref, base, feed):
    got = _dwsim_case(T, P, mode, base, feed)
    assert abs(got - ref) < TOL_CONV_DWSIM[mode]