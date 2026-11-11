"""Rackett Z (Zra) resolution: priority user > database Z_RA > Zc, and zra_default='zc'.

Priority is implemented in methods._volume_data. With zra_default='zc' (Zc = Pc*Vc/(R*Tc))
the database Z_RA is ignored, so results must not move when the database carries that column.
"""

import numpy as np
import pandas as pd
import pytest

from thermo_sim import METHODS, Stream, ThermoMethod, build_eos, load_database

KIJ_PR = np.array([[0.0, 0.1030, 0.0],
                   [0.1030, 0.0, 0.2193],
                   [0.0, 0.2193, 0.0]])          # H2, N2, NH3 (ordered by Tc)
FEED = {'hydrogen': 0.726, 'nitrogen': 0.242, 'ammonia': 0.032}


def _nh3(z_ra=None, vc=73.014):
    """Single-component table with the columns build_eos needs; Z_RA only if given."""
    d = pd.DataFrame({'Component': ['ammonia'], 'Tc (K)': [405.56], 'Pc (bar)': [113.63],
                      'w': [0.2526], 'frac_mol': [1.0], 'Vc (cm3/mol)': [vc]})
    if z_ra is not None:
        d['Z_RA'] = z_ra
    return d


# ----------------------------------------------------------------------------------------
# Priority: user > database > Zc
# ----------------------------------------------------------------------------------------
def test_database_value_is_used_when_there_is_no_override():
    eos = build_eos('PR', _nh3(z_ra=0.2477))
    assert eos.Zra[0] == pytest.approx(0.2477)
    assert eos.zra_source[0] == 'db'


def test_zc_is_used_when_the_database_has_no_z_ra():
    eos = build_eos('PR', _nh3())
    assert eos.Zra[0] == pytest.approx(eos.Zc[0])
    assert eos.zra_source[0] == 'Zc'


def test_user_override_wins_over_database():
    m = METHODS['PR'].but(constants_user={'ammonia': {'Zra': 0.30}})
    eos = build_eos(m, _nh3(z_ra=0.2477))
    assert eos.Zra[0] == pytest.approx(0.30)
    assert eos.zra_source[0] == 'user'


def test_priority_is_resolved_per_component():
    d = pd.DataFrame({'Component': ['hydrogen', 'nitrogen', 'ammonia'],
                      'Tc (K)': [33.19, 126.192, 405.56], 'Pc (bar)': [13.13, 33.958, 113.63],
                      'w': [-0.216, 0.0372, 0.2526], 'frac_mol': [0.726, 0.242, 0.032],
                      'Vc (cm3/mol)': [64.5, 89.4, 73.0],
                      'Z_RA': [0.3187, np.nan, 0.2477]})       # nitrogen has no database value
    m = METHODS['PR'].but(constants_user={'hydrogen': {'Zra': 0.3139}})
    eos = build_eos(m, d)
    assert list(eos.zra_source) == ['user', 'Zc', 'db']
    assert eos.Zra[0] == pytest.approx(0.3139)
    assert eos.Zra[1] == pytest.approx(eos.Zc[1])
    assert eos.Zra[2] == pytest.approx(0.2477)


# ----------------------------------------------------------------------------------------
# Invalid values fall through to the next source (and warn)
# ----------------------------------------------------------------------------------------
def test_invalid_user_value_falls_back_to_database():
    """DWSIM reports Zra = 0 for ammonia (not set): it must not disable Rackett."""
    m = METHODS['PR'].but(constants_user={'ammonia': {'Zra': 0.0}})
    with pytest.warns(UserWarning, match=r'Zra \(user\)'):
        eos = build_eos(m, _nh3(z_ra=0.2477))
    assert eos.Zra[0] == pytest.approx(0.2477)
    assert eos.zra_source[0] == 'db'


def test_invalid_database_value_falls_back_to_zc():
    with pytest.warns(UserWarning, match=r'Zra \(db\)'):
        eos = build_eos('PR', _nh3(z_ra=1.5))
    assert eos.Zra[0] == pytest.approx(eos.Zc[0])
    assert eos.zra_source[0] == 'Zc'


def test_without_vc_and_without_database_value_zra_is_missing():
    d = _nh3().drop(columns='Vc (cm3/mol)')
    eos = build_eos('PR', d)
    assert np.isnan(eos.Zra[0])
    assert eos.zra_source[0] == 'no data'


# ----------------------------------------------------------------------------------------
# zra_default
# ----------------------------------------------------------------------------------------
def test_zra_default_zc_ignores_the_database():
    eos = build_eos(METHODS['PR'].but(zra_default='zc'), _nh3(z_ra=0.2477))
    assert eos.Zra[0] == pytest.approx(eos.Zc[0])
    assert eos.zra_source[0] == 'Zc'


def test_zra_default_zc_still_honours_the_user_override():
    m = METHODS['PR'].but(zra_default='zc', constants_user={'ammonia': {'Zra': 0.30}})
    assert build_eos(m, _nh3(z_ra=0.2477)).zra_source[0] == 'user'


def test_invalid_zra_default_is_rejected():
    with pytest.raises(ValueError):
        ThermoMethod('pr', 'std', zra_default='other')


def test_general_presets_use_the_database():
    assert METHODS['PR'].zra_default == 'db'


# ----------------------------------------------------------------------------------------
# Stream level: zra_default='zc' results do not move when the database has Z_RA
# ----------------------------------------------------------------------------------------
@pytest.fixture(scope='module')
def db():
    return load_database()


def _stream(db, method, comp_mol, **over):
    kw = dict(description='x', fm=None, comp_mass=None, fn=100.0, comp_mol=comp_mol,
              V_frac=None, T=230.0, P=150.0, Tref=298.15, Pref=1.01325, psat_factor=None,
              gamma=None, component_data=db, kij=None, k_mode='cubic', method=method,
              emit_warnings=False)
    kw.update(over)
    return Stream(**kw)


def test_database_has_a_valid_z_ra_for_every_component(db):
    assert 'Z_RA' in db.columns
    z = db['Z_RA'].to_numpy(float)
    assert np.all((z > 0.1) & (z < 1.0))


@pytest.mark.parametrize('model', ['eos_peneloux', 'rackett', 'poly'])
def test_zc_default_density_is_independent_of_the_database_z_ra(db, model):
    """zra_default='zc': removing the Z_RA column must not change the liquid density at all."""
    m = METHODS['PR'].but(zra_default='zc', liquid_volume=model)
    with_col = _stream(db, m, {'ammonia': 1.0})
    without = _stream(db.drop(columns=['Z_RA']), m, {'ammonia': 1.0})
    assert with_col.props.dens_mix_liq == pytest.approx(without.props.dens_mix_liq, rel=1e-12)


@pytest.mark.parametrize('model', ['eos_peneloux', 'rackett'])
def test_general_preset_density_uses_the_database_z_ra(db, model):
    """PR (zra_default='db'): the Z_RA column does change Rackett and Peneloux."""
    m = METHODS['PR'].but(liquid_volume=model)
    with_col = _stream(db, m, {'ammonia': 1.0})
    without = _stream(db.drop(columns=['Z_RA']), m, {'ammonia': 1.0})
    assert with_col.props.dens_mix_liq != pytest.approx(without.props.dens_mix_liq, rel=1e-6)


def test_eos_and_poly_do_not_depend_on_zra(db):
    for model in ('eos', 'poly'):
        m = METHODS['PR'].but(liquid_volume=model)
        a = _stream(db, m, {'ammonia': 1.0}).props.dens_mix_liq
        b = _stream(db.drop(columns=['Z_RA']), m, {'ammonia': 1.0}).props.dens_mix_liq
        assert a == pytest.approx(b, rel=1e-12)


def test_equilibrium_does_not_depend_on_the_liquid_volume_model(db):
    """The liquid volume model (and Zra) only enters the density: same flash temperature."""
    kw = dict(V_frac=0.99, T=300.0, P=150.0, kij=KIJ_PR)
    t = [_stream(db, METHODS['PR'].but(liquid_volume=lv), FEED, **kw).T
         for lv in ('eos', 'eos_peneloux', 'rackett', 'poly')]
    assert np.ptp(t) < 1e-8


def test_zra_source_is_reported_on_the_stream(db):
    s = _stream(db, METHODS['PR'].but(liquid_volume='rackett'), FEED, V_frac=0.99, T=300.0,
                P=150.0, kij=KIJ_PR)
    assert set(s.props.zra_source) == {'db'}
    s = _stream(db, METHODS['PR'].but(zra_default='zc', liquid_volume='rackett'), FEED, V_frac=0.99,
                T=300.0, P=150.0, kij=KIJ_PR)
    assert 'db' not in set(s.props.zra_source)
