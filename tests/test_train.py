"""Compressor train and cooler tests against closed-form ideal-gas results (no database needed)."""

import math
from types import SimpleNamespace

import pytest

from thermo_sim.unitops import Cooler, CompressorTrain
from thermo_sim.constants import R_GAS



class _IdealStream:
    def __init__(self, T, P, fn=0.105, MM=28.0134, Cp=29.2, phase='vapor'):
        self.T, self.P, self.fn, self.MM, self.Cp, self.phase = T, P, fn, MM, Cp, phase

    @property
    def fm(self):
        return self.fn * self.MM

    @property
    def props(self):
        return SimpleNamespace(
            H_total_mix=self.Cp * (self.T - 298.15),
            S_total_mix=self.Cp * math.log(self.T / 298.15) - R_GAS * math.log(self.P / 1.01325),
            Cp_total_mix=self.Cp, dens_mix=self.P * 100.0 * self.MM / (R_GAS * self.T),
            phase=self.phase)

    def at(self, T=None, P=None, description=None):
        return _IdealStream(self.T if T is None else T, self.P if P is None else P,
                            self.fn, self.MM, self.Cp, self.phase)


@pytest.fixture
def feed():
    return _IdealStream(T=303.15, P=5.0)


def test_cooler_duty_and_pressure_drop(feed):
    hot = feed.at(T=400.0)
    c = Cooler(T_out=318.15, dP=0.3).run(hot)
    assert c.outlet.T == 318.15 and c.outlet.P == pytest.approx(4.7)
    assert c.duty_kW == pytest.approx(-hot.fn * hot.Cp * (400.0 - 318.15))
    assert Cooler(500.0).run(hot).duty_kW > 0         # heating gives positive duty


def test_cooler_rejects_condensed_outlet(feed):
    class _Liq(_IdealStream):
        def at(self, T=None, P=None, description=None):
            return _IdealStream(T, P, self.fn, self.MM, self.Cp, phase='liquid')
    with pytest.raises(ValueError, match='condensation'):
        Cooler(300.0).run(_Liq(400.0, 5.0))
    assert Cooler(300.0, allow_condensation=True).run(_Liq(400.0, 5.0)).outlet.T == 300.0


def test_train_matches_closed_form(feed):
    N, eta, Tc = 5, 0.72, 318.15
    r = (155.0 / 5.0) ** (1.0 / N)
    res = CompressorTrain(155.0, N, eta, T_cool=Tc).run(feed)
    assert res.ratio_per_stage == pytest.approx(r, rel=1e-12)
    T_in, W, Q = feed.T, 0.0, 0.0
    for k in range(N):
        T2 = T_in + (T_in * r ** (R_GAS / feed.Cp) - T_in) / eta
        W += feed.fn * feed.Cp * (T2 - T_in)
        if k < N - 1:
            Q += feed.fn * feed.Cp * (T2 - Tc)
        T_in = Tc
    assert res.W_ind_kW == pytest.approx(W, rel=1e-9)
    assert res.duty_cooling_kW == pytest.approx(-Q, rel=1e-9)
    assert res.stages[-1].P_out == 155.0
    assert len(res.coolers) == N - 1


def test_train_with_cooler_pressure_drop(feed):
    res = CompressorTrain(155.0, 5, 0.72, T_cool=318.15, dP_cooler=0.2).run(feed)
    assert res.stages[-1].P_out == pytest.approx(155.0)
    for s, c, nxt in zip(res.stages, res.coolers, res.stages[1:]):
        assert nxt.P_in == pytest.approx(s.P_out - 0.2)
        assert s.P_out / s.P_in == pytest.approx(res.ratio_per_stage, rel=1e-9)


def test_aftercooler_adds_a_cooler(feed):
    a = CompressorTrain(155.0, 5, 0.72, T_cool=318.15).run(feed)
    b = CompressorTrain(155.0, 5, 0.72, T_cool=318.15, aftercooler=True).run(feed)
    assert len(b.coolers) == 5 and b.W_ind_kW == pytest.approx(a.W_ind_kW)
    assert b.duty_cooling_kW < a.duty_cooling_kW
    assert b.outlet.T == 318.15


def test_train_argument_checks(feed):
    with pytest.raises(ValueError, match='T_cool'):
        CompressorTrain(155.0, 3, 0.72)
    with pytest.raises(ValueError, match='values'):
        CompressorTrain(155.0, 3, 0.72, T_cool=[310.0])
    with pytest.raises(ValueError, match='aftercooler'):
        CompressorTrain(155.0, 3, 0.72, T_cool=[310.0, 310.0], aftercooler=True)
    with pytest.raises(ValueError, match='aftercooler'):
        CompressorTrain(155.0, 3, 0.72, T_cool=[310.0, 310.0, None], aftercooler=True)
    assert CompressorTrain(155.0, 1, 0.72).run(feed).coolers == []


def _closed_form(feed, N, eta, Tc, r):
    """Ideal-gas train: Tc[k] is the cooler temperature after stage k (None = no cooler)."""
    a = R_GAS / feed.Cp
    T_in, W, Q, T_out = feed.T, 0.0, 0.0, []
    for k in range(N):
        T2 = T_in * (1.0 + (r ** a - 1.0) / eta)
        W += feed.fn * feed.Cp * (T2 - T_in)
        T_out.append(T2)
        if Tc[k] is not None:
            Q += feed.fn * feed.Cp * (T2 - Tc[k])
            T_in = Tc[k]
        else:
            T_in = T2
    return W, Q, T_out


def test_train_with_skipped_coolers(feed):
    Tc = [None, 318.15, None, 325.0, None]
    res = CompressorTrain(155.0, 5, 0.72, T_cool=Tc).run(feed)
    r = (155.0 / 5.0) ** (1.0 / 5)
    W, Q, T_out = _closed_form(feed, 5, 0.72, Tc, r)
    assert res.cooler_stages == [2, 4] and len(res.coolers) == 2
    assert res.W_ind_kW == pytest.approx(W, rel=1e-9)
    assert res.duty_cooling_kW == pytest.approx(-Q, rel=1e-9)
    for s, T in zip(res.stages, T_out):
        assert s.T_out == pytest.approx(T, abs=1e-6)
    assert 'Stage 5' in res.report()


def test_pressure_drop_only_in_existing_coolers(feed):
    Tc = [318.15, None, 318.15, None, None]
    res = CompressorTrain(155.0, 5, 0.72, T_cool=Tc, dP_cooler=0.5).run(feed)
    assert res.stages[-1].P_out == pytest.approx(155.0)
    for k, (s, nxt) in enumerate(zip(res.stages, res.stages[1:])):
        drop = 0.5 if Tc[k] is not None else 0.0
        assert nxt.P_in == pytest.approx(s.P_out - drop)
        assert s.P_out / s.P_in == pytest.approx(res.ratio_per_stage, rel=1e-9)


def test_list_forms_are_equivalent(feed):
    a = CompressorTrain(155.0, 4, 0.72, T_cool=318.15).run(feed)
    b = CompressorTrain(155.0, 4, 0.72, T_cool=[318.15] * 3).run(feed)
    c = CompressorTrain(155.0, 4, 0.72, T_cool=[318.15] * 3 + [None]).run(feed)
    d = CompressorTrain(155.0, 4, 0.72, T_cool=[318.15] * 4, aftercooler=True).run(feed)
    e = CompressorTrain(155.0, 4, 0.72, T_cool=318.15, aftercooler=True).run(feed)
    assert a.W_ind_kW == b.W_ind_kW == c.W_ind_kW
    assert a.duty_cooling_kW == b.duty_cooling_kW == c.duty_cooling_kW
    assert d.duty_cooling_kW == pytest.approx(e.duty_cooling_kW) and len(d.coolers) == 4
