"""Compressor tests against closed-form ideal-gas results (no database needed)."""

import math
from types import SimpleNamespace

import pytest

from thermo_sim.unitops import Compressor

from thermo_sim.constants import R_GAS


# ---------------------------------------------------------------- ideal gas

class _IdealStream:
    """Ideal gas with constant Cp, same interface as Stream for the compressor."""
    def __init__(self, T, P, fn=1.0, MM=2.016, Cp=29.0, phase='vapor'):
        self.T, self.P, self.fn, self.MM, self.Cp, self.phase = T, P, fn, MM, Cp, phase

    @property
    def fm(self):
        return self.fn * self.MM

    @property
    def props(self):
        return SimpleNamespace(
            H_total_mix=self.Cp * (self.T - 298.15),
            S_total_mix=self.Cp * math.log(self.T / 298.15) - R_GAS * math.log(self.P / 1.01325),
            Cp_total_mix=self.Cp,
            dens_mix=self.P * 100.0 * self.MM / (R_GAS * self.T),
            phase=self.phase)

    def at(self, T=None, P=None, description=None):
        return _IdealStream(self.T if T is None else T, self.P if P is None else P,
                            self.fn, self.MM, self.Cp, self.phase)


@pytest.fixture
def feed():
    return _IdealStream(T=303.15, P=80.0, fn=0.3125)


def test_isentropic_matches_closed_form(feed):
    r = Compressor(P_out=155.0, eta=0.82, eta_mec=0.99).run(feed)
    T2s = feed.T * (155.0 / 80.0) ** (R_GAS / feed.Cp)
    assert r.T_out_is == pytest.approx(T2s, abs=1e-6)
    assert r.T_out == pytest.approx(feed.T + (T2s - feed.T) / 0.82, abs=1e-6)
    assert r.W_ind_kW == pytest.approx(feed.fn * feed.Cp * (r.T_out - feed.T), rel=1e-9)
    assert r.W_brake_kW == pytest.approx(r.W_ind_kW / 0.99, rel=1e-12)
    assert r.eta_is_equiv == pytest.approx(0.82)
    assert r.S_out > r.S_in                       # irreversible: entropy rises


def test_polytropic_matches_closed_form(feed):
    r = Compressor(P_out=155.0, eta=0.82, mode='polytropic', n_steps=20).run(feed)
    T2 = feed.T * (155.0 / 80.0) ** (R_GAS / (feed.Cp * 0.82))
    assert r.T_out == pytest.approx(T2, abs=5e-3)      # Heun, 20 steps: error ~2e-3 K
    assert r.eta_is_equiv < 0.82                  # isentropic eff. below polytropic for compression


def test_polytropic_converges_with_steps(feed):
    T2 = feed.T * (155.0 / 80.0) ** (R_GAS / (feed.Cp * 0.82))
    err = [abs(Compressor(155.0, 0.82, mode='polytropic', n_steps=n).run(feed).T_out - T2)
           for n in (2, 4, 8, 16)]
    assert all(b < a for a, b in zip(err, err[1:]))
    assert err[-1] < 5e-3
    assert err[-2] / err[-1] == pytest.approx(4.0, rel=0.3)     # second-order scheme


def test_rejects_liquid_or_two_phase_feed():
    with pytest.raises(ValueError, match='vapor'):
        Compressor(155.0).run(_IdealStream(300.0, 80.0, phase='two_phase'))


def test_rejects_non_compression(feed):
    with pytest.raises(ValueError, match='exceed'):
        Compressor(P_out=50.0).run(feed)


def test_rejects_bad_arguments():
    with pytest.raises(ValueError):
        Compressor(155.0, eta=1.2)
    with pytest.raises(ValueError):
        Compressor(155.0, mode='adiabatic')


