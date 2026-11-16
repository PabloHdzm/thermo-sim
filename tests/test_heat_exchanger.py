"""HeatExchanger: closed-form checks with constant-Cp liquids and the DWSIM water tutorial.

Part 1 needs no database: two incompressible liquids with constant Cp, for which the effectiveness-NTU
relations are exact. Part 2 compares with the Heat Exchanger Design tutorial of DWSIM 10.2.12.0
(water, Steam Tables IAPWS-IF97), reference_data/dwsim/heat_exchanger_water.json.
"""
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from thermo_sim.unitops.heat_exchanger import HeatExchanger, lmtd, lmtd_correction_factor

REF = Path(__file__).resolve().parents[1] / 'reference_data' / 'dwsim' / 'heat_exchanger_water.json'


class _Liquid:
    """Incompressible liquid with constant Cp [kJ/(kmol K)]; H and S are relative to 298.15 K."""

    def __init__(self, T, P, fn, Cp, phase='liquid'):
        self.T, self.P, self.fn, self.Cp, self.phase = T, P, fn, Cp, phase

    @property
    def props(self):
        return SimpleNamespace(H_total_mix=self.Cp * (self.T - 298.15), S_total_mix=self.Cp * math.log(self.T / 298.15),
                               Cp_total_mix=self.Cp, phase=self.phase)

    def at(self, T=None, P=None):
        return _Liquid(self.T if T is None else T, self.P if P is None else P, self.fn, self.Cp, self.phase)


CP_W = 75.3                                      # kJ/(kmol K), water
HOT = _Liquid(400.0, 3.0, 1.0 / 18.0153, CP_W)
COLD = _Liquid(300.0, 1.0, 2.0 / 18.0153, CP_W)


def _eps_ntu(NTU, Cr, flow):
    if flow == 'parallel':
        return (1.0 - math.exp(-NTU * (1.0 + Cr))) / (1.0 + Cr)
    if abs(Cr - 1.0) < 1e-12:
        return NTU / (1.0 + NTU)
    e = math.exp(-NTU * (1.0 - Cr))
    return (1.0 - e) / (1.0 - Cr * e)


@pytest.mark.parametrize('flow', ['counter', 'parallel'])
@pytest.mark.parametrize('UA', [0.5, 2.5, 20.0])
def test_ua_mode_matches_effectiveness_ntu(flow, UA):
    r = HeatExchanger('ua', flow=flow, UA=UA).run(HOT, COLD)
    C_h, C_c = HOT.fn * CP_W, COLD.fn * CP_W
    C_min, C_max = min(C_h, C_c), max(C_h, C_c)
    Q = _eps_ntu(UA / C_min, C_min / C_max, flow) * C_min * 100.0
    assert r.Q_kW == pytest.approx(Q, rel=1e-9)
    assert r.Q_hot_kW == pytest.approx(r.Q_cold_kW, rel=1e-12)
    assert r.T_hot_out == pytest.approx(400.0 - Q / C_h, abs=1e-9)
    assert r.T_cold_out == pytest.approx(300.0 + Q / C_c, abs=1e-9)
    assert r.NTU == pytest.approx(UA / C_min, rel=1e-9)
    assert r.UA == pytest.approx(UA, rel=1e-9)


def test_lmtd_limits():
    assert lmtd(10.0, 10.0) == 10.0
    assert lmtd(30.0, 10.0) == pytest.approx(20.0 / math.log(3.0))
    assert lmtd(0.0, 10.0) == 0.0 and lmtd(-1.0, 10.0) == 0.0


def test_counter_current_exceeds_co_current_and_uses_the_pinch():
    cc = HeatExchanger('ua', flow='counter', UA=2.5).run(HOT, COLD)
    cp = HeatExchanger('ua', flow='parallel', UA=2.5).run(HOT, COLD)
    assert cc.Q_kW > cp.Q_kW
    assert cc.T_cold_out > cp.T_cold_out and cp.T_cold_out < cp.T_hot_out        # co-current cannot cross
    assert cc.pinch == pytest.approx(min(cc.T_hot_in - cc.T_cold_out, cc.T_hot_out - cc.T_cold_in), abs=1e-9)


@pytest.mark.parametrize('mode,kw', [('hot_out', dict(T_hot_out=350.0)), ('cold_out', dict(T_cold_out=330.0)),
                                     ('duty', dict(duty=150.0)), ('efficiency', dict(efficiency=0.4))])
def test_specified_modes_round_trip_through_ua(mode, kw):
    r = HeatExchanger(mode, **kw).run(HOT, COLD)
    back = HeatExchanger('ua', UA=r.UA).run(HOT, COLD)
    assert back.Q_kW == pytest.approx(r.Q_kW, rel=1e-9)
    assert back.T_hot_out == pytest.approx(r.T_hot_out, abs=1e-8) and back.T_cold_out == pytest.approx(r.T_cold_out, abs=1e-8)
    assert r.balance_error < 1e-12
    if mode == 'efficiency':
        assert r.efficiency == pytest.approx(0.4, rel=1e-12)


def test_area_from_u_and_pressure_drops():
    r = HeatExchanger('hot_out', T_hot_out=350.0, U=0.5, dP_hot=0.1, dP_cold=0.05).run(HOT, COLD)
    assert r.A == pytest.approx(r.UA / 0.5, rel=1e-12)
    assert r.P_hot_out == pytest.approx(2.9) and r.P_cold_out == pytest.approx(0.95)
    ua = HeatExchanger('ua', U=0.5, A=r.A).run(HOT, COLD)
    assert ua.A == r.A and ua.Q_kW == pytest.approx(r.Q_kW, rel=1e-9)


def test_profile_ua_equals_lmtd_ua_for_constant_cp():
    r = HeatExchanger('ua', UA=2.5, n_profile=10).run(HOT, COLD)
    assert r.UA_profile == pytest.approx(2.5, rel=1e-3)                  # trapezoid error of 10 segments
    assert HeatExchanger('ua', UA=2.5, n_profile=200).run(HOT, COLD).UA_profile == pytest.approx(2.5, rel=1e-5)
    assert r.profile[0].T_hot == 400.0 and r.profile[-1].T_cold == 300.0


def test_energy_and_second_law():
    r = HeatExchanger('ua', UA=2.5).run(HOT, COLD)
    assert r.S_gen_kWK > 0 and r.exergy_destroyed_kW == pytest.approx(298.15 * r.S_gen_kWK)
    assert r.Q_kW < r.Q_max_kW and 0 < r.efficiency < 1


def test_phase_change_is_flagged_and_inconsistent_inputs_rejected():
    class _Boil(_Liquid):
        def at(self, T=None, P=None):
            T = self.T if T is None else T
            return _Boil(T, self.P if P is None else P, self.fn, self.Cp, 'vapor' if T > 310.0 else 'liquid')
    assert any('phase' in w for w in HeatExchanger('ua', UA=2.5).run(HOT, _Boil(300.0, 1.0, COLD.fn, CP_W)).warnings)
    with pytest.raises(ValueError, match='hotter'):
        HeatExchanger('ua', UA=1.0).run(COLD, HOT)
    with pytest.raises(ValueError, match='exceeds the maximum'):
        HeatExchanger('duty', duty=1e6).run(HOT, COLD)
    with pytest.raises(ValueError, match='UA'):
        HeatExchanger('ua')
    with pytest.raises(ValueError, match='result'):
        HeatExchanger('duty', duty=10.0, UA=1.0)


def test_f_factor_limits_and_consistency_with_effectiveness_ntu():
    assert lmtd_correction_factor(400.0, 350.0, 300.0, 340.0) < 1.0
    assert lmtd_correction_factor(400.0, 400.0 - 1e-9, 300.0, 340.0) == pytest.approx(1.0, abs=1e-9)   # R -> 0
    # 1-2 exchanger: effectiveness-NTU, eps = 2 / (1 + C + s coth(NTU s / 2)), s = sqrt(1 + C^2)
    C_h, C_c, UA = HOT.fn * CP_W, COLD.fn * CP_W, 2.5
    C_min, Cr = min(C_h, C_c), min(C_h, C_c) / max(C_h, C_c)
    s = math.sqrt(1.0 + Cr * Cr)
    eps = 2.0 / (1.0 + Cr + s / math.tanh(UA / C_min * s / 2.0))
    Q = eps * C_min * 100.0
    T_ho, T_co = 400.0 - Q / C_h, 300.0 + Q / C_c
    F = lmtd_correction_factor(400.0, T_ho, 300.0, T_co)
    assert Q == pytest.approx(UA * F * lmtd(400.0 - T_co, T_ho - 300.0), rel=1e-9)
    r = HeatExchanger('ua', UA=UA, F=F).run(HOT, COLD)
    assert r.Q_kW == pytest.approx(Q, rel=1e-9)


# ---- DWSIM 10.2.12.0, Heat Exchanger Design tutorial (water, IAPWS-IF97) --------------------------

@pytest.fixture(scope='module')
def tutorial():
    from thermo_sim import Stream, load_database
    ref = json.loads(REF.read_text())
    db = load_database()
    if 'water' not in set(db['Component'].astype(str).str.lower()):
        pytest.skip('water missing: run scripts/build_db.py')

    def water(d):
        return Stream('water', fm=d['mass_flow_kg_s'], comp_mass=None, fn=None, comp_mol={'water': 1.0}, V_frac=None,
                      T=d['T_K'], P=d['P_bar'], Tref=298.15, Pref=1.01325, psat_factor=None, gamma=None,
                      component_data=db, kij=None, method='PR', k_mode='gamma', emit_warnings=False)
    inp = ref['inputs']
    hx = HeatExchanger('ua', UA=inp['UA_W_K'] / 1000.0, dP_hot=inp['dP_hot_Pa'] / 1e5, dP_cold=inp['dP_cold_Pa'] / 1e5)
    return ref, hx.run(water(inp['hot_in']), water(inp['cold_in']))


def test_tutorial_temperatures_and_duty(tutorial):
    ref, r = tutorial
    d = ref['results']
    assert r.T_hot_out == pytest.approx(d['hot_out_T_K'], abs=0.35)
    assert r.T_cold_out == pytest.approx(d['cold_out_T_K'], abs=0.35)
    assert r.Q_kW == pytest.approx(d['heat_load_kW'], rel=2e-3)
    assert r.LMTD == pytest.approx(d['LMTD_K'], abs=0.1)
    assert r.F == 1.0


def test_tutorial_efficiency_and_maximum_duty(tutorial):
    ref, r = tutorial
    d = ref['results']
    # the database liquid Cp is 0.3 % above IF97 at 300 K, which moves the 100 K enthalpy span (observed 0.63 %)
    assert r.Q_max_kW == pytest.approx(d['maximum_theoretical_heat_exchange_kW'], rel=1e-2)
    assert 100 * r.efficiency == pytest.approx(d['thermal_efficiency_pct'], abs=0.25)


def test_tutorial_pressures_and_balance(tutorial):
    ref, r = tutorial
    assert r.P_hot_out == pytest.approx(ref['streams']['hot_out']['P_bar'], abs=1e-9)
    assert r.P_cold_out == pytest.approx(ref['streams']['cold_out']['P_bar'], abs=1e-9)
    assert r.balance_error < 1e-6 and not r.warnings

