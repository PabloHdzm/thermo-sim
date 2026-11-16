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


# ---- phase change: pure fluid with a boiling plateau ----------------------------------------------

class _Fluid:
    """Pure fluid, constant Cp in each phase and a constant boiling temperature (any P).
    H [kJ/kmol] from liquid at 298.15 K. State = (T, vapor fraction); inside the plateau T = T_SAT."""
    CPL, CPV, HVAP, T_SAT, T0 = 75.3, 34.0, 40660.0, 373.15, 298.15
    calls = 0

    def __init__(self, T, P, fn, vf=None):
        if vf is None:
            vf = 0.0 if T < self.T_SAT else 1.0
        self.T, self.P, self.fn, self.vf = T, P, fn, vf

    @classmethod
    def _H(cls, T, vf):
        Hl = cls.CPL * (cls.T_SAT - cls.T0)
        if vf >= 1.0 and T > cls.T_SAT:
            return Hl + cls.HVAP + cls.CPV * (T - cls.T_SAT)
        if T < cls.T_SAT:
            return cls.CPL * (T - cls.T0)
        return Hl + vf * cls.HVAP

    @classmethod
    def _S(cls, T, vf):
        Sl = cls.CPL * math.log(cls.T_SAT / cls.T0)
        if T < cls.T_SAT:
            return cls.CPL * math.log(T / cls.T0)
        if vf >= 1.0 and T > cls.T_SAT:
            return Sl + cls.HVAP / cls.T_SAT + cls.CPV * math.log(T / cls.T_SAT)
        return Sl + vf * cls.HVAP / cls.T_SAT

    @property
    def props(self):
        phase = 'liquid' if self.T < self.T_SAT else ('vapor' if self.vf >= 1.0 else 'two_phase')
        Cp = self.CPL if phase == 'liquid' else self.CPV
        return SimpleNamespace(H_total_mix=self._H(self.T, self.vf), S_total_mix=self._S(self.T, self.vf),
                               Cp_total_mix=Cp, phase=phase, V_frac=self.vf)

    def at(self, T=None, P=None, V_frac=None, H_flow=None, T_guess=None):
        type(self).calls += 1
        P = self.P if P is None else P
        if V_frac is not None:
            return _Fluid(self.T_SAT, P, self.fn, V_frac)
        if H_flow is not None:
            H = H_flow / self.fn
            Hl, Hv = self.CPL * (self.T_SAT - self.T0), self.CPL * (self.T_SAT - self.T0) + self.HVAP
            if H <= Hl:
                return _Fluid(self.T0 + H / self.CPL, P, self.fn, 0.0)
            if H >= Hv:
                return _Fluid(self.T_SAT + (H - Hv) / self.CPV, P, self.fn, 1.0)
            return _Fluid(self.T_SAT, P, self.fn, (H - Hl) / self.HVAP)
        return _Fluid(self.T if T is None else T, P, self.fn)


def _condenser_hot():
    return _Fluid(400.0, 1.5, 0.02)                      # superheated vapor


def test_hot_vf_condenses_to_the_requested_vapor_fraction():
    vf = 0.25
    cold = _Liquid(300.0, 1.0, 0.3, CP_W)
    r = HeatExchanger('hot_vf', vf_hot_out=vf).run(_condenser_hot(), cold)
    Q = 0.02 * (_Fluid.CPV * (400.0 - _Fluid.T_SAT) + (1.0 - vf) * _Fluid.HVAP)
    assert r.Q_kW == pytest.approx(Q, rel=1e-12)
    assert r.T_hot_out == _Fluid.T_SAT and r.V_frac_hot_out == pytest.approx(vf)
    assert r.T_cold_out == pytest.approx(300.0 + Q / (0.3 * CP_W), abs=1e-9)
    assert r.balance_error < 1e-9
    L = lmtd(400.0 - r.T_cold_out, _Fluid.T_SAT - 300.0)
    assert r.UA == pytest.approx(Q / L, rel=1e-9)
    assert any('phase' in w for w in r.warnings)
    assert r.pinch > 0 and r.UA_profile > 0


def test_ua_mode_reproduces_the_vapor_fraction_mode():
    cold = _Liquid(300.0, 1.0, 0.3, CP_W)
    ref = HeatExchanger('hot_vf', vf_hot_out=0.25).run(_condenser_hot(), cold)
    back = HeatExchanger('ua', UA=ref.UA).run(_condenser_hot(), cold)
    assert back.Q_kW == pytest.approx(ref.Q_kW, rel=1e-8)
    assert back.T_cold_out == pytest.approx(ref.T_cold_out, abs=1e-6)
    assert back.V_frac_hot_out == pytest.approx(0.25, abs=1e-6)          # PH flash inside the plateau


def test_cold_vf_boils_the_cold_stream():
    cold = _Fluid(300.0, 2.0, 0.05)
    hot = _Liquid(450.0, 3.0, 0.2, CP_W)
    r = HeatExchanger('cold_vf', vf_cold_out=0.5, dP_cold=0.1).run(hot, cold)
    Q = 0.05 * (_Fluid.CPL * (_Fluid.T_SAT - 300.0) + 0.5 * _Fluid.HVAP)
    assert r.Q_kW == pytest.approx(Q, rel=1e-12)
    assert r.T_cold_out == _Fluid.T_SAT and r.V_frac_cold_out == pytest.approx(0.5)
    assert r.T_hot_out == pytest.approx(450.0 - Q / (0.2 * CP_W), abs=1e-9)
    assert r.P_cold_out == pytest.approx(1.9)
    assert r.S_gen_kWK > 0 and r.balance_error < 1e-9
    fine = HeatExchanger('cold_vf', vf_cold_out=0.5, dP_cold=0.1, n_profile=400).run(hot, cold)
    coarse = HeatExchanger('cold_vf', vf_cold_out=0.5, dP_cold=0.1, n_profile=200).run(hot, cold)
    assert fine.UA_profile == pytest.approx(coarse.UA_profile, rel=2e-3)


def test_vapor_fraction_modes_reject_the_wrong_direction():
    cold = _Liquid(300.0, 1.0, 0.3, CP_W)
    wet = _Fluid(_Fluid.T_SAT, 1.5, 0.02, vf=0.5)                                       # half vapor already
    with pytest.raises(ValueError, match='vapor fraction'):
        HeatExchanger('hot_vf', vf_hot_out=0.9).run(wet, cold)                            # cooling cannot add vapor
    with pytest.raises(ValueError, match='vapor fraction'):
        HeatExchanger('cold_vf', vf_cold_out=0.5).run(_Liquid(500.0, 3.0, 0.2, CP_W), _Fluid(400.0, 1.0, 0.05))
    with pytest.raises(ValueError, match='exceeds the maximum'):
        HeatExchanger('hot_vf', vf_hot_out=0.0).run(_Fluid(380.0, 1.5, 5.0), _Liquid(370.0, 1.0, 0.001, CP_W))
    with pytest.raises(ValueError, match=r'\[0, 1\]'):
        HeatExchanger('hot_vf', vf_hot_out=1.5)
    with pytest.raises(ValueError, match='vf_cold_out'):
        HeatExchanger('cold_vf')


def test_solver_needs_few_stream_evaluations():
    """Each Stream.at() is a full Stream build in the real code: the cached curves keep them in the tens."""
    class _Counting(_Liquid):
        n = 0

        def at(self, T=None, P=None):
            _Counting.n += 1
            return _Counting(self.T if T is None else T, self.P if P is None else P, self.fn, self.Cp, self.phase)
    hot, cold = _Counting(400.0, 3.0, 1.0 / 18.0153, CP_W), _Counting(300.0, 1.0, 2.0 / 18.0153, CP_W)
    HeatExchanger('ua', UA=2.5, dP_hot=0.1, dP_cold=0.05).run(hot, cold)
    assert _Counting.n < 100                                                     # was ~330 with one brentq per outlet
    _Fluid.calls = 0
    HeatExchanger('ua', UA=2.0).run(_condenser_hot(), _Liquid(300.0, 1.0, 0.3, CP_W))
    assert _Fluid.calls < 120                                                   # includes finding the plateau once


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
    # the database liquid Cp is about 0.6 % above IAPWS-IF97 over 300-400 K (observed 0.63 % on Qmax)
    assert r.Q_max_kW == pytest.approx(d['maximum_theoretical_heat_exchange_kW'], rel=1e-2)
    assert 100 * r.efficiency == pytest.approx(d['thermal_efficiency_pct'], abs=0.25)


def test_tutorial_pressures_and_balance(tutorial):
    ref, r = tutorial
    assert r.P_hot_out == pytest.approx(ref['streams']['hot_out']['P_bar'], abs=1e-9)
    assert r.P_cold_out == pytest.approx(ref['streams']['cold_out']['P_bar'], abs=1e-9)
    assert r.balance_error < 1e-6 and not r.warnings


def test_tutorial_maximum_duty_case():
    """DWSIM, same streams, vapor fraction mode with an unreachable fraction: it saturates at Q_max
    (thermal efficiency 100 %). Equivalent to mode 'efficiency' with efficiency = 1."""
    from thermo_sim import Stream, load_database
    ref = json.loads(REF.read_text())
    inp, d, s = ref['inputs'], ref['case_max_duty']['results'], ref['case_max_duty']['streams']
    db = load_database()

    def water(x):
        return Stream('water', fm=x['mass_flow_kg_s'], comp_mass=None, fn=None, comp_mol={'water': 1.0}, V_frac=None,
                      T=x['T_K'], P=x['P_bar'], Tref=298.15, Pref=1.01325, psat_factor=None, gamma=None,
                      component_data=db, kij=None, method='PR', k_mode='gamma', emit_warnings=False)
    r = HeatExchanger('efficiency', efficiency=1.0, dP_hot=inp['dP_hot_Pa'] / 1e5,
                      dP_cold=inp['dP_cold_Pa'] / 1e5).run(water(inp['hot_in']), water(inp['cold_in']))
    assert r.efficiency == pytest.approx(1.0, abs=1e-12)
    assert r.T_hot_out == pytest.approx(d['hot_out_T_K'], abs=1e-6)                      # cooled to the cold inlet temperature
    assert r.Q_kW == pytest.approx(d['heat_load_kW'], rel=1e-2)                          # database liquid Cp, see above
    assert r.T_cold_out == pytest.approx(d['cold_out_T_K'], abs=0.6)
    assert r.P_cold_out == pytest.approx(s['cold_out']['P_bar'], abs=1e-9)
    assert r.balance_error < 1e-6


@pytest.mark.parametrize('case', ['case_hot_vf', 'case_cold_vf'])
def test_dwsim_vapor_fraction_modes(case):
    """DWSIM, outlet molar vapor fraction of stream 1 (condenser) and of stream 2 (evaporator), water.
    The gamma-phi enthalpy of vaporization comes from the database correlation (about 0.3 % above IF97 at 1 atm
    once the PR vapor residual is included) and the liquid Cp is about 0.6 % high: Q within 1 %."""
    from thermo_sim import Stream, load_database
    c = json.loads(REF.read_text())[case]
    d, inp, s = c['results'], c['inputs'], c['streams']
    db = load_database()

    def water(x):
        return Stream('water', fm=x['mass_flow_kg_s'], comp_mass=None, fn=None, comp_mol={'water': 1.0}, V_frac=None,
                      T=x['T_K'], P=x['P_bar'], Tref=298.15, Pref=1.01325, psat_factor=None, gamma=None,
                      component_data=db, kij=None, method='PR', k_mode='gamma', emit_warnings=False)
    r = HeatExchanger(c['mode'], **c['spec']).run(water(inp['hot_in']), water(inp['cold_in']))
    spec_side = 'hot' if c['mode'] == 'hot_vf' else 'cold'
    other_side = 'cold' if spec_side == 'hot' else 'hot'
    assert getattr(r, f'V_frac_{spec_side}_out') == pytest.approx(next(iter(c['spec'].values())), abs=1e-9)
    assert getattr(r, f'T_{spec_side}_out') == pytest.approx(d[f'{spec_side}_out_T_K'], abs=0.3)      # saturation temperature
    assert getattr(r, f'T_{other_side}_out') == pytest.approx(d[f'{other_side}_out_T_K'], abs=1.0)
    assert r.Q_kW == pytest.approx(d['heat_load_kW'], rel=1e-2)
    assert r.Q_max_kW == pytest.approx(d['maximum_theoretical_heat_exchange_kW'], rel=1e-2)
    assert 100 * r.efficiency == pytest.approx(d['thermal_efficiency_pct'], abs=1.0)
    assert r.LMTD == pytest.approx(d['LMTD_K'], rel=1e-2)
    assert r.UA == pytest.approx(d['heat_load_kW'] / d['LMTD_K'], rel=1.5e-2)
    assert r.balance_error < 1e-6
    assert r.P_hot_out == pytest.approx(s['hot_out']['P_bar'], abs=1e-9)
    assert r.P_cold_out == pytest.approx(s['cold_out']['P_bar'], abs=1e-9)
    assert any('phase' in w for w in r.warnings)