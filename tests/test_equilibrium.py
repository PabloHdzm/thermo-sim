"""Phase-equilibrium engine: physical invariants and reference values."""
import numpy as np
import pytest

from thermo_sim import build_eos
from thermo_sim.equilibrium import EquilibriumEngine

P = 150.0


@pytest.fixture
def engine(h2_n2_nh3):
    return EquilibriumEngine(build_eos('PR', h2_n2_nh3))


@pytest.fixture
def engine_nh3(nh3):
    return EquilibriumEngine(build_eos('PR', nh3))


# ── pure component ──────────────────────────────────────────────────────────
def test_ammonia_psat_at_minus_20C_matches_reference(engine_nh3):
    """NIST: Psat(NH3, 253.15 K) = 1.902 bar."""
    assert engine_nh3.pure_psat(253.15) == pytest.approx(1.902, rel=0.01)


def test_tsat_inverts_psat(engine_nh3):
    ps = engine_nh3.pure_psat(253.15)
    assert engine_nh3.pure_Tsat(ps).T == pytest.approx(253.15, abs=1e-6)


def test_pure_ammonia_phase_at_253K(engine_nh3):
    assert engine_nh3.phase_pure(253.15, 1.0) == 'vapor'
    assert engine_nh3.phase_pure(253.15, 5.0) == 'liquid'


# ── flash TP ────────────────────────────────────────────────────────────────
@pytest.mark.parametrize('T', [200.0, 240.0, 253.75])
def test_two_phase_flash_balances_and_fugacities(engine, h2_n2_nh3, T):
    z = h2_n2_nh3['frac_mol'].to_numpy()
    f = engine.flash_TP(T, P)
    assert f.phase == 'two_phase' and f.converged
    # material balance and normalization
    assert f.beta * f.y + (1 - f.beta) * f.x == pytest.approx(z, abs=1e-9)
    assert f.x.sum() == pytest.approx(1.0) and f.y.sum() == pytest.approx(1.0)
    # isofugacity: y_i phi_i^V = x_i phi_i^L
    lnv = engine.k.phi(T, P, f.y, 'v')[0] + np.log(f.y)
    lnl = engine.k.phi(T, P, f.x, 'l')[0] + np.log(f.x)
    assert lnv == pytest.approx(lnl, abs=1e-7)
    # K-values consistent with the phase compositions
    assert f.K == pytest.approx(f.y / f.x, rel=1e-7)


def test_single_gas_phase_when_hot(engine, h2_n2_nh3):
    f = engine.flash_TP(300.0, P)
    assert f.phase == 'vapor'
    assert f.beta == pytest.approx(1.0)
    assert f.y == pytest.approx(h2_n2_nh3['frac_mol'].to_numpy())


def test_dew_point_separates_one_and_two_phase_regions(engine):
    dew = engine.dew_T(P)
    assert dew.status == 'ok' and np.isfinite(dew.T)
    assert engine.flash_TP(dew.T - 3.0, P).phase == 'two_phase'
    assert engine.flash_TP(dew.T + 3.0, P).phase == 'vapor'


def test_no_bubble_point_for_supercritical_light_gases(engine):
    """H2 and N2 are supercritical: there is no bubble temperature at 150 bar."""
    bub = engine.bubble_T(P)
    assert np.isnan(bub.T) and bub.status == 'no_crossing'


# ── flash VP ────────────────────────────────────────────────────────────────
def test_flash_vp_returns_requested_vapor_fraction(engine):
    T, f = engine.flash_VP(P, 0.99)
    assert f.beta == pytest.approx(0.99, abs=1e-9)
    back = engine.flash_TP(T, P)
    assert back.beta == pytest.approx(0.99, abs=1e-6)
    assert back.x == pytest.approx(f.x, abs=1e-6)


# ── robustness ──────────────────────────────────────────────────────────────
def test_zero_fraction_component_does_not_change_the_result(engine, feed_with_argon):
    eng4 = EquilibriumEngine(build_eos('PR', feed_with_argon))
    f3, f4 = engine.flash_TP(240.0, P), eng4.flash_TP(240.0, P)
    assert f4.beta == pytest.approx(f3.beta, abs=1e-9)
    assert f4.x[:3] == pytest.approx(f3.x, abs=1e-9)
    assert f4.y[:3] == pytest.approx(f3.y, abs=1e-9)
    assert f4.x[3] == 0.0 and f4.y[3] == 0.0


# ── robustness of the stability test ────────────────────────────────────────
def test_unconverged_stability_is_not_reported_as_stable(h2_n2_nh3):
    """With max_ss=1 the TPD iterations cannot converge: the outcome must be
    'inconclusive' (never 'stable') and the flash must still find the two phases."""
    from thermo_sim.equilibrium import SolverOptions
    eng = EquilibriumEngine(build_eos('PR', h2_n2_nh3), SolverOptions(max_ss=1))
    z = h2_n2_nh3['frac_mol'].to_numpy()
    assert eng._stability(240.0, P, z).status == 'inconclusive'
    f = eng.flash_TP(240.0, P)
    assert f.phase == 'two_phase' and 0.0 < f.beta < 1.0


def test_stability_status_for_clear_cases(engine, h2_n2_nh3):
    z = h2_n2_nh3['frac_mol'].to_numpy()
    assert engine._stability(240.0, P, z).status == 'unstable'
    assert engine._stability(400.0, P, z).status == 'stable'


def test_extra_starts_agree_with_default(h2_n2_nh3):
    from thermo_sim.equilibrium import SolverOptions
    a = EquilibriumEngine(build_eos('PR', h2_n2_nh3))
    b = EquilibriumEngine(build_eos('PR', h2_n2_nh3), SolverOptions(stability_extra_starts=True))
    for T in (200.0, 240.0, 253.75, 300.0):
        fa, fb = a.flash_TP(T, P), b.flash_TP(T, P)
        assert fa.phase == fb.phase
        assert fa.beta == pytest.approx(fb.beta, abs=1e-8)


def test_no_artificial_eos_root_in_normal_use(engine):
    """The root fallback warns; a normal T-P sweep must never trigger it."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        for T in np.linspace(150.0, 600.0, 31):
            for Pb in (1.0, 50.0, 150.0, 300.0):
                engine.flash_TP(float(T), Pb)


# ── single source of truth ──────────────────────────────────────────────────
def test_kernel_and_eos_share_the_same_roots(h2_n2_nh3):
    eos = build_eos('PR', h2_n2_nh3)
    k = EquilibriumEngine(eos).k
    for A, B in [(0.5, 0.05), (1.2, 0.08), (3.0, 0.2), (0.01, 0.001)]:
        assert k.roots(A, B) == eos._valid_roots(A, B)


def test_pure_saturation_fugacities_are_equal(engine_nh3):
    """phi_L = phi_V at Psat, evaluated with the kernel (same formula used inside pure_psat)."""
    T = 253.15
    ps = engine_nh3.pure_psat(T)
    k = engine_nh3.k
    one = np.array([1.0])
    lnl = k.phi_eos(T, ps, one, 'l')[0]
    lnv = k.phi_eos(T, ps, one, 'v')[0]
    assert lnl == pytest.approx(lnv, abs=1e-8)


def test_two_phase_labelling_does_not_depend_on_the_starting_k(engine, h2_n2_nh3):
    """An inverted starting K must not return the mirrored split (liquid = denser phase)."""
    z = h2_n2_nh3['frac_mol'].to_numpy()
    T = 240.0
    ref = engine.flash_TP(T, P)
    for K0 in (ref.K, 1.0 / ref.K, engine._wilson(T, P), 1.0 / engine._wilson(T, P)):
        beta, x, y, K, Zl, Zv, conv, _ = engine._two_phase(T, P, z, np.clip(K0, 1e-12, 1e12))
        assert beta == pytest.approx(ref.beta, abs=1e-8)
        assert x == pytest.approx(ref.x, abs=1e-8) and y == pytest.approx(ref.y, abs=1e-8)
        assert Zl < Zv


def test_trial_phase_is_oriented_by_density_not_by_start_label(h2_n2_nh3):
    """With pure-component starts, the NH3-rich stationary point can be reached by a 'v' or an 'l'
    trial (equal S). Its orientation must be liquid-like either way, so K0 is never inverted."""
    from thermo_sim.equilibrium import SolverOptions
    z = h2_n2_nh3['frac_mol'].to_numpy()
    for extra in (False, True):
        eng = EquilibriumEngine(build_eos('PR', h2_n2_nh3), SolverOptions(stability_extra_starts=extra))
        for T in (200.0, 240.0, 253.75):
            st = eng._stability(T, P, z)
            assert st.status == 'unstable' and st.trial == 'l'
            K0 = z / st.w
            assert K0[2] < 1.0          # NH3 goes to the liquid


# ── phase labels: only vapor, liquid, two_phase ─────────────────────────────
def test_only_three_phase_labels_over_a_wide_grid(engine):
    labels = {engine.flash_TP(float(T), Pb).phase
              for T in np.linspace(150.0, 700.0, 23) for Pb in (1.0, 50.0, 150.0, 400.0)}
    assert labels <= {'vapor', 'liquid', 'two_phase'}


def test_pure_component_has_no_supercritical_label(engine_nh3):
    # NH3: Tc = 405.56 K, Pc = 113.63 bar
    assert engine_nh3.phase_pure(450.0, 200.0) == 'vapor'
    assert engine_nh3.phase_pure(450.0, 50.0) == 'vapor'
    assert engine_nh3.phase_pure(300.0, 200.0) == 'liquid'


def test_above_component_criticals_flag(engine):
    # feed: Tc_max = 405.56 K (NH3), Pc_max = 113.63 bar
    assert engine.above_component_criticals(500.0, 150.0)
    assert not engine.above_component_criticals(300.0, 150.0)
    assert not engine.above_component_criticals(500.0, 50.0)
    hot = engine.flash_TP(500.0, 150.0)
    assert hot.phase == 'vapor' and hot.above_criticals
    assert not engine.flash_TP(240.0, 150.0).above_criticals