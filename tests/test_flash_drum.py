"""Flash drum tests.

Part 1: control flow with a fake stream (constant-K Rachford-Rice, constant Cp), no database.
Part 2: internal consistency of the three specification modes on a reactor-effluent-like feed
(H2 0.6, N2 0.2, NH3 0.2, 280 K, 150 bar).
"""

import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from thermo_sim.unitops import FlashDrum


# ---------------------------------------------------------------- part 1: fake stream

class _FakeStream:
    Cp, MM, K, shift = 30.0, 20.0, np.array([4.0, 0.2]), 0.0

    def __init__(self, T, P, fn, comp_mol, force=None):
        self.T, self.P, self.fn, self.comp_mol = T, P, fn, dict(comp_mol)
        self.component_data = pd.DataFrame({'Component': list(comp_mol)})
        z = np.array(list(comp_mol.values()), float)
        self.fm = fn * self.MM
        V, y, x, phase = self._flash(z) if force is None else force(z)
        self.props = SimpleNamespace(
            phase=phase, V_frac=V, yi=y, xi=x, TDew=T, TBubble=T,
            H_total_mix=self.Cp * (T - 298.15), S_total_mix=0.0)

    def _flash(self, z):
        K = self.K
        f = lambda V: float(np.sum(z * (K - 1) / (1 + V * (K - 1))))
        if f(0.0) <= 0:
            return 0.0, z, z, 'liquid'
        if f(1.0) >= 0:
            return 1.0, z, z, 'vapor'
        lo, hi = 0.0, 1.0
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
        V = 0.5 * (lo + hi)
        x = z / (1 + V * (K - 1))
        return V, K * x / np.sum(K * x), x / np.sum(x), 'two_phase'

    def at(self, T=None, P=None, description=None, *, fn=None, comp_mol=None, V_frac=None,
           H_flow=None, T_guess=None):
        fn_new = self.fn if fn is None else fn
        comp = self.comp_mol if comp_mol is None else comp_mol
        P_new = self.P if P is None else P
        guess = self.T if T_guess is None else T_guess
        if H_flow is not None:
            return _FakeStream(298.15 + H_flow / fn_new / self.Cp, P_new, fn_new, comp)
        if V_frac is not None:
            ph = 'vapor' if V_frac == 1.0 else 'liquid' if V_frac == 0.0 else 'two_phase'
            return _FakeStream(guess + self.shift, P_new, fn_new, comp,
                               force=lambda z: (V_frac, z, z, ph))
        return _FakeStream(self.T if T is None else T, P_new, fn_new, comp)


@pytest.fixture
def feed():
    return _FakeStream(300.0, 50.0, 2.0, {'a': 0.5, 'b': 0.5})


def test_spec_validation():
    with pytest.raises(ValueError, match='exactly one'):
        FlashDrum()
    with pytest.raises(ValueError, match='exactly one'):
        FlashDrum(T=300.0, duty_kW=1.0)
    with pytest.raises(ValueError, match='not both'):
        FlashDrum(T=300.0, P=40.0, dP=1.0)
    with pytest.raises(ValueError, match='V_frac'):
        FlashDrum(V_frac=1.5)


def test_isothermal_split_closes_balances(feed):
    r = FlashDrum(T=300.0, dP=2.0).run(feed)
    assert r.P == 48.0 and 0.0 < r.V_frac < 1.0
    assert r.vapor.fn == pytest.approx(r.V_frac * feed.fn)
    assert r.liquid.fn == pytest.approx((1 - r.V_frac) * feed.fn)
    assert r.mass_error < 1e-12
    assert r.duty_kW == pytest.approx(0.0, abs=1e-9)
    assert r.energy_error_kW == pytest.approx(0.0, abs=1e-9)
    assert 'Flash drum' in r.report()


def test_duty_mode_solves_temperature(feed):
    r = FlashDrum(duty_kW=100.0).run(feed)
    assert r.T == pytest.approx(300.0 + 100.0 / (feed.fn * feed.Cp))
    assert r.duty_kW == pytest.approx(100.0)
    assert r.energy_error_kW == pytest.approx(0.0, abs=1e-9)
    assert FlashDrum(duty_kW=-100.0).run(feed).T < 300.0       # negative duty cools


def test_vfrac_mode(feed):
    r = FlashDrum(V_frac=0.4).run(feed)
    assert r.V_frac == pytest.approx(0.4)
    assert r.vapor.fn == pytest.approx(0.4 * feed.fn)


def test_single_phase_feeds_have_one_outlet():
    vap = FlashDrum(T=300.0).run(_FakeStream(300.0, 50.0, 1.0, {'a': 0.95, 'b': 0.05}))
    assert vap.liquid is None and vap.vapor is not None and vap.V_frac == 1.0
    liq = FlashDrum(T=300.0).run(_FakeStream(300.0, 50.0, 1.0, {'a': 0.05, 'b': 0.95}))
    assert liq.vapor is None and liq.liquid is not None and liq.V_frac == 0.0


def test_outlet_on_another_root_raises(feed, monkeypatch):
    monkeypatch.setattr(_FakeStream, 'shift', 0.01)
    with pytest.raises(RuntimeError, match='another root'):
        FlashDrum(T=300.0).run(feed)


# ---------------------------------------------------------------- part 2: effluent, real Stream

KIJ_PR = np.array([[0.0, 0.1030, 0.0], [0.1030, 0.0, 0.2193], [0.0, 0.2193, 0.0]])


@pytest.fixture(scope='module')
def eff():
    from thermo_sim import METHODS, Stream, load_database
    return Stream(
        description='effluent', fm=None, comp_mass=None, fn=1.0,
        comp_mol={'hydrogen': 0.6, 'nitrogen': 0.2, 'ammonia': 0.2}, V_frac=None,
        T=280.0, P=150.0, Tref=298.15, Pref=1.01325, gamma=None,
        component_data=load_database(), kij=KIJ_PR,
        method=METHODS['PR'].but(liquid_volume='poly'), emit_warnings=False)


def test_effluent_drum_closes_balances(eff):
    r = FlashDrum(T=280.0).run(eff)
    assert 0.0 < r.V_frac < 1.0
    assert r.mass_error < 1e-10
    assert r.duty_kW == pytest.approx(0.0, abs=1e-6)
    assert r.energy_error_kW == pytest.approx(0.0, abs=1e-2)


def test_effluent_modes_are_consistent(eff):
    warm = eff.at(T=300.0)
    base = FlashDrum(T=280.0).run(warm)                  # duty needed to go 300 K -> 280 K
    by_duty = FlashDrum(duty_kW=base.duty_kW).run(warm)
    assert by_duty.T == pytest.approx(280.0, abs=1e-4)
    assert by_duty.V_frac == pytest.approx(base.V_frac, abs=1e-6)
    by_vf = FlashDrum(V_frac=base.V_frac).run(warm)
    assert by_vf.T == pytest.approx(280.0, abs=1e-3)


def test_vapor_outlet_is_saturated_at_drum_temperature(eff):
    r = FlashDrum(T=280.0).run(eff)
    assert r.vapor.props.TDew == pytest.approx(280.0, abs=1e-3)


@pytest.mark.xfail(strict=False, reason=(
    "Known limitation: the liquid outlet is nearly pure NH3 at 150 bar, above the NH3 critical "
    "pressure, so its phase envelope is very thin. TBubble comes back as NaN here. To be "
    "revisited with the bubble/dew validation."))
def test_liquid_outlet_saturation_temperatures_are_defined(eff):
    p = FlashDrum(T=280.0).run(eff).liquid.props
    assert np.isfinite(p.TBubble) and np.isfinite(p.TDew)
    assert p.TBubble <= p.TDew