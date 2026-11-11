"""Multistage compressor train with intercoolers (equal pressure ratio per stage).

Stage k compresses from its suction to its discharge; if stage k has a cooler, a `Cooler`
then brings the gas to that cooler's temperature (with pressure drop `dP_cooler`), and the
result is the next suction. Which stages have a cooler is set with `T_cool` (see
`CompressorTrain`). The ratio r is the same in every stage and is solved so the last
discharge equals `P_out` exactly, also when the coolers have a pressure drop.

Number of stages and intercooler temperatures are inputs here: choosing them is a design
decision, kept out of the model.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .compressor import Compressor, CompressorResult
from .cooler import Cooler, CoolerResult


@dataclass
class TrainResult:
    n_stages: int
    ratio_per_stage: float
    stages: list[CompressorResult] = field(default_factory=list)
    coolers: list[CoolerResult] = field(default_factory=list)
    cooler_stages: list[int] = field(default_factory=list)   # stage index (1-based) of each cooler
    W_ind_kW: float = 0.0
    W_brake_kW: float = 0.0
    duty_cooling_kW: float = 0.0     # sum of cooler duties, negative
    T_max_K: float = 0.0
    outlet: object = None

    def report(self) -> str:
        sep = '-' * 72
        L = [sep, f"Compressor train: {self.n_stages} stages, ratio/stage "
                  f"{self.ratio_per_stage:.4f}", sep,
             f"Indicated power : {self.W_ind_kW:.2f} kW",
             f"Brake power     : {self.W_brake_kW:.2f} kW",
             f"Cooling duty    : {self.duty_cooling_kW:.2f} kW",
             f"T max discharge : {self.T_max_K - 273.15:.1f} C", ""]
        for i, s in enumerate(self.stages):
            L.append(f"Stage {i + 1}: {s.P_in:8.3f} -> {s.P_out:8.3f} bar   "
                     f"T {s.T_in - 273.15:6.1f} -> {s.T_out - 273.15:6.1f} C   "
                     f"W {s.W_ind_kW:8.2f} kW")
            if (i + 1) in self.cooler_stages:
                c = self.coolers[self.cooler_stages.index(i + 1)]
                L.append(f"         cooler -> {c.T_out - 273.15:6.1f} C, dP {c.P_in - c.P_out:.3f} bar, "
                         f"duty {c.duty_kW:9.2f} kW")
        L.append(sep)
        return '\n'.join(L)


class CompressorTrain:
    def __init__(self, P_out: float, n_stages: int, eta: float, eta_mec: float = 1.0,
                 mode: str = 'isentropic', T_cool=None, dP_cooler: float = 0.0,
                 aftercooler: bool = False, n_steps: int = 20):
        """Cooler placement is set by `T_cool` [K]:

        * a number: a cooler after every stage but the last (plus one after the last stage
          if `aftercooler=True`), all at that temperature;
        * a list of N values, one per stage: `None` means no cooler after that stage, and
          the last value is the aftercooler (so `aftercooler` must not contradict it);
        * a list of N-1 values: intercoolers only, same meaning for `None`;
        * `None`: no coolers (only allowed for a single stage).

        dP_cooler [bar] is the pressure drop of every cooler that exists.
        """
        if n_stages < 1:
            raise ValueError("n_stages must be >= 1.")
        if dP_cooler < 0:
            raise ValueError("dP_cooler must be >= 0.")
        self.P_out, self.n, self.dP = float(P_out), int(n_stages), float(dP_cooler)
        self.aftercooler = bool(aftercooler)
        self.T_stage = self._expand_T_cool(T_cool)
        self.n_coolers = sum(t is not None for t in self.T_stage)
        self.eta, self.eta_mec, self.mode, self.n_steps = eta, eta_mec, mode, n_steps

    def _expand_T_cool(self, T_cool) -> list:
        """One entry per stage: a temperature or None."""
        N = self.n
        if T_cool is None:
            if N > 1:
                raise ValueError("T_cool is required for a train with more than one stage.")
            if self.aftercooler:
                raise ValueError("aftercooler=True needs a temperature in T_cool.")
            return [None]
        if not hasattr(T_cool, '__len__'):
            return [float(T_cool)] * (N - 1) + [float(T_cool) if self.aftercooler else None]
        vals = [None if t is None else float(t) for t in T_cool]
        if len(vals) == N - 1:
            if self.aftercooler:
                raise ValueError("aftercooler=True needs N values in T_cool (one per stage).")
            return vals + [None]
        if len(vals) != N:
            raise ValueError(f"T_cool needs N-1 or N values for {N} stages, got {len(vals)}.")
        if self.aftercooler and vals[-1] is None:
            raise ValueError("aftercooler=True but the last value of T_cool is None.")
        return vals

    def _final_pressure(self, P1: float, r: float) -> float:
        """Discharge pressure of the last stage for ratio r; only existing coolers drop dP."""
        P = P1
        for k in range(self.n - 1):
            P = r * P - (self.dP if self.T_stage[k] is not None else 0.0)
        return r * P

    def ratio(self, P1: float) -> float:
        """Equal ratio per stage so that the last discharge equals P_out."""
        if self.P_out <= P1:
            raise ValueError("P_out must exceed the inlet pressure.")
        lo, hi = 1.0, (self.P_out / P1) ** (1.0 / self.n) * 1.5 + 1.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if self._final_pressure(P1, mid) < self.P_out:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def run(self, inlet) -> TrainResult:
        r = self.ratio(float(inlet.P))
        res = TrainResult(n_stages=self.n, ratio_per_stage=r)
        feed = inlet
        for k in range(self.n):
            last = k == self.n - 1
            P_dis = self.P_out if last else r * feed.P
            st = Compressor(P_dis, self.eta, self.eta_mec, self.mode, self.n_steps).run(feed)
            res.stages.append(st)
            res.W_ind_kW += st.W_ind_kW
            res.W_brake_kW += st.W_brake_kW
            res.T_max_K = max(res.T_max_K, st.T_out)
            feed = st.outlet
            if self.T_stage[k] is not None:
                c = Cooler(self.T_stage[k], dP=self.dP).run(feed)
                res.coolers.append(c)
                res.cooler_stages.append(k + 1)
                res.duty_cooling_kW += c.duty_kW
                feed = c.outlet
        res.outlet = feed
        return res
