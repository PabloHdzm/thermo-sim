"""Stream: phase structure, balances and algebra.

Uses the packaged property database (src/thermo_sim/data/properties_db.csv).
"""
import numpy as np
import pytest

from thermo_sim import Stream, load_database

KIJ_PR = np.array([[0.0, 0.1030, 0.0],
                   [0.1030, 0.0, 0.2193],
                   [0.0, 0.2193, 0.0]])          # H2, N2, NH3 (ordered by Tc)
FN = 1340e3 / (17.031 * 3600)
FEED = {'hydrogen': 0.726, 'nitrogen': 0.242, 'ammonia': 0.032}


@pytest.fixture(scope='module')
def db():
    return load_database()


@pytest.fixture(scope='module')
def make(db):
    def _make(**over):
        kw = dict(description='feed', fm=None, comp_mass=None, fn=FN, comp_mol=FEED,
                  V_frac=0.99, T=300.0, P=150.0, Tref=298.15, Pref=1.01325,
                  psat_factor=None, gamma=None, component_data=db, kij=KIJ_PR,
                  method='PR', k_mode='cubic', emit_warnings=False)
        kw.update(over)
        return Stream(**kw)
    return _make


@pytest.fixture(scope='module')
def two_phase(make):
    return make()                                  # VF = 0.99, T solved by the flash


@pytest.fixture(scope='module')
def gas(make):
    return make(V_frac=None, T=300.0)


@pytest.fixture(scope='module')
def liquid(make):
    return make(V_frac=None, T=230.0, comp_mol={'hydrogen': 0.001, 'nitrogen': 0.001, 'ammonia': 0.998})


@pytest.mark.parametrize('X', ['H', 'S', 'Cp'])
@pytest.mark.parametrize('kind', ['ig', 'dep', 'total'])
class TestPhaseStructure:
    def test_gas_only(self, gas, X, kind):
        p = gas.props
        m, g, l = (getattr(p, f'{X}_{kind}_mix{s}') for s in ('', '_gas', '_liq'))
        assert m == pytest.approx(g) and np.isnan(l)

    def test_liquid_only(self, liquid, X, kind):
        p = liquid.props
        m, g, l = (getattr(p, f'{X}_{kind}_mix{s}') for s in ('', '_gas', '_liq'))
        assert m == pytest.approx(l) and np.isnan(g)

    def test_two_phase_is_weighted_and_has_no_nan(self, two_phase, X, kind):
        p = two_phase.props
        m, g, l = (getattr(p, f'{X}_{kind}_mix{s}') for s in ('', '_gas', '_liq'))
        assert not np.isnan([m, g, l]).any()
        assert m == pytest.approx(p.V_frac * g + p.L_frac * l)
        ind = getattr(p, f'{X}_{kind}_indiv')
        assert not np.isnan(ind).any()


@pytest.mark.parametrize('X', ['H', 'S', 'Cp'])
def test_total_is_ideal_plus_departure(two_phase, X):
    p = two_phase.props
    assert getattr(p, f'{X}_total_mix') == pytest.approx(getattr(p, f'{X}_ig_mix') + getattr(p, f'{X}_dep_mix'))


def test_component_material_balance_across_phases(two_phase):
    p = two_phase.props
    assert p.fn_gas + p.fn_liq == pytest.approx(p.molar_flows, rel=1e-12)
    assert p.V + p.L == pytest.approx(two_phase.fn)


def test_mass_and_molar_flows_are_consistent(two_phase):
    assert two_phase.props.mass_flows.sum() == pytest.approx(two_phase.fm)
    assert two_phase.fm == pytest.approx((two_phase.props.molar_flows * two_phase.MM).sum())


def test_vapor_fraction_specification_is_met(two_phase):
    assert two_phase.props.V_frac == pytest.approx(0.99, abs=1e-9)


def test_ph_flash_recovers_temperature(make):
    a = make(V_frac=None, T=253.7)
    b = make(V_frac=None, T=None, H_flow=a.fn * a.props.H_total_mix, T_guess=240.0)
    assert b.T == pytest.approx(253.7, abs=1e-6)


def test_adiabatic_mixing_conserves_enthalpy_and_flow(make):
    a, b = make(V_frac=None, T=260.0), make(V_frac=None, T=320.0, fn=FN * 0.5)
    c = a + b
    assert c.fn == pytest.approx(a.fn + b.fn)
    assert c.fn * c.props.H_total_mix == pytest.approx(a.fn * a.props.H_total_mix + b.fn * b.props.H_total_mix, rel=1e-6)
    assert min(a.T, b.T) <= c.T <= max(a.T, b.T)


def test_subtracting_a_stream_removes_its_flow(make):
    a, b = make(V_frac=None, T=300.0), make(V_frac=None, T=300.0, fn=FN * 0.25)
    assert (a - b).fn == pytest.approx(0.75 * FN)


def test_scaling_changes_flows_not_intensive_properties(two_phase):
    h = two_phase * 0.5
    assert h.fn == pytest.approx(0.5 * two_phase.fn)
    assert h.T == two_phase.T
    assert h.props.H_total_mix == pytest.approx(two_phase.props.H_total_mix)
    assert h.props.V == pytest.approx(0.5 * two_phase.props.V)


def test_split_preserves_total_flow(two_phase):
    parts = two_phase.split([0.3, 0.2])
    assert len(parts) == 3
    assert sum(p.fn for p in parts) == pytest.approx(two_phase.fn)


def test_invalid_specifications_are_rejected(make):
    with pytest.raises(ValueError):
        make(T=None, H_flow=None, T_guess=None)
    with pytest.raises(ValueError):
        make(T=300.0, H_flow=1.0)
    with pytest.raises(ValueError):
        make(fn=None, comp_mol=FEED)


@pytest.mark.parametrize('T', [150.0, 1100.0])
def test_warns_when_temperature_is_outside_cp_fit_range(make, T):
    with pytest.warns(RuntimeWarning, match='ideal-gas Cp'):
        make(V_frac=None, T=T, comp_mol={'hydrogen': 1.0}, kij=None, emit_warnings=True)

def test_no_range_warning_inside_fit_range(make, recwarn):
    make(V_frac=None, T=300.0, emit_warnings=True)
    assert not [w for w in recwarn if 'ideal-gas Cp' in str(w.message)]


# Density: an absent phase is NaN, like the energy properties
GAS_FIELDS = ('MM_g', 'Vm_gas', 'Vm_gas_eos', 'Z_gas_eos', 'Z_gas', 'dens_mix_gas')
LIQ_FIELDS = ('MM_l', 'Vm_liq', 'Vm_liq_eos', 'Z_liq_eos', 'Z_liq_rho', 'Z_liq', 'dens_mix_liq')


def test_gas_only_density_fields(gas):
    p = gas.props
    assert all(np.isnan(getattr(p, f)) for f in LIQ_FIELDS)
    assert np.isnan(p.dens_pure_liq).all() and p.liquid_volume_source == ''
    assert all(np.isfinite(getattr(p, f)) for f in GAS_FIELDS)
    assert np.isfinite(p.dens_pure_gas).all()
    assert p.dens_mix == pytest.approx(p.dens_mix_gas)
    assert p.dens_pure == pytest.approx(p.dens_pure_gas)


def test_liquid_only_density_fields(liquid):
    p = liquid.props
    assert all(np.isnan(getattr(p, f)) for f in GAS_FIELDS)
    assert np.isnan(p.dens_pure_gas).all()
    assert all(np.isfinite(getattr(p, f)) for f in LIQ_FIELDS)
    assert np.isfinite(p.dens_pure_liq).all()
    assert p.dens_mix == pytest.approx(p.dens_mix_liq)
    assert p.dens_pure == pytest.approx(p.dens_pure_liq)


def test_two_phase_density_fields_are_all_defined(two_phase):
    p = two_phase.props
    assert all(np.isfinite(getattr(p, f)) for f in GAS_FIELDS + LIQ_FIELDS)
    assert np.isfinite(p.dens_pure_gas).all() and np.isfinite(p.dens_pure_liq).all()


def test_two_phase_density_is_mass_weighted_volume_average(two_phase):
    p = two_phase.props
    expected = 1.0 / (p.V_frac_m_total / p.dens_mix_gas + p.L_frac_m_total / p.dens_mix_liq)
    assert p.dens_mix == pytest.approx(expected)
    assert p.dens_mix_liq > 5 * p.dens_mix_gas             # the liquid is far denser than the gas


def test_volumetric_flow_follows_density(two_phase, gas, liquid):
    for s in (two_phase, gas, liquid):
        assert s.props.fv == pytest.approx(s.fm / s.props.dens_mix)


# Reference: CoolProp (reference EOS), NH3 at 230 K and 150 bar = 700.27 kg/m3.
# The raw PR volume root under-predicts the liquid density; the other models are the
# usual corrections. The tolerances document the accuracy of each option.
@pytest.mark.parametrize('model,max_rel_err', [('eos', 0.13), ('eos_peneloux', 0.06),
                                               ('poly', 0.02), ('rackett', 0.02)])
def test_liquid_density_models_against_reference(make, model, max_rel_err):
    from thermo_sim import METHODS
    s = make(V_frac=None, T=230.0, comp_mol={'ammonia': 1.0}, kij=None,
             method=METHODS['PR'].but(liquid_volume=model))
    assert s.props.dens_mix_liq == pytest.approx(700.27, rel=max_rel_err)
