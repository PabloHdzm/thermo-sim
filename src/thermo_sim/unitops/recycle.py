"""Recycle (tear stream) solver: finds the stream that the flowsheet returns to its own input.

The flowsheet is a function ``loop(recycle) -> calculated_recycle``. Mixing and splitting are done with the
`Stream` algebra (``fresh + recycle``, ``stream.split([...])``), so no Mixer or Splitter block is needed::

    def loop(rec):
        mixed = fresh if rec is None else fresh + rec          # rec is None on the first pass (empty recycle)
        hot = Cooler(350.0, allow_condensation=True).run(mixed).outlet
        product, recycle = hot.split([0.7, 0.3])
        return recycle

    result = Recycle(method='legacy').solve(loop)

Tear variables: molar flow of each component, T and P of the recycle stream. Convergence follows the DWSIM
Recycle block: the difference between the stream that is guessed and the one that the flowsheet calculates must be
below a tolerance for each of temperature [K], pressure [bar] and mass flow [kg/s] (defaults 0.1, 0.1, 0.01).

Methods
-------
'legacy'    direct (successive) substitution with optional smoothing, x_new = (1 - s) x + s g(x). Default.
'wegstein'  bounded Wegstein acceleration, per variable (q limited to [-5, 0]).
'broyden'   global Broyden (secant update of the inverse Jacobian of g(x) - x on all tear variables together).

`smoothing` s in (0, 1]: 1 is plain substitution, smaller values damp the iteration. It only acts in 'legacy'.
thermo-sims' definition; it is not claimed to be the DWSIM smoothing factor.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..stream import Stream

_METHODS = ('legacy', 'wegstein', 'broyden')
_Q_MIN, _Q_MAX = -5.0, 0.0


@dataclass
class RecycleResult:
    stream: object                    # converged recycle stream as calculated by the flowsheet
    converged: bool
    iterations: int
    errors: dict                      # final |calculated - guessed|: 'T' [K], 'P' [bar], 'mass_flow' [kg/s]
    tolerances: dict
    method: str
    history: list = field(default_factory=list)   # (iteration, err_T, err_P, err_mass) per pass
    vector: np.ndarray | None = None  # final tear vector [n_1..n_c (kmol/s), T, P]


class Recycle:
    def __init__(self, tol_T: float = 0.1, tol_P: float = 0.1, tol_mass: float = 0.01, max_iter: int = 50,
                 method: str = 'legacy', smoothing: float = 1.0):
        if method not in _METHODS:
            raise ValueError(f"method must be one of {_METHODS}.")
        if not 0.0 < smoothing <= 1.0:
            raise ValueError("smoothing must be in (0, 1].")
        if max_iter < 1 or min(tol_T, tol_P, tol_mass) <= 0:
            raise ValueError("max_iter >= 1 and positive tolerances are required.")
        self.tol_T, self.tol_P, self.tol_mass = float(tol_T), float(tol_P), float(tol_mass)
        self.max_iter, self.method, self.smoothing = int(max_iter), method, float(smoothing)

    # ------------------------------------------------------------------ stream <-> vector

    @staticmethod
    def _names(s: Stream) -> list[str]:
        return list(s.comp_mol)

    @staticmethod
    def _to_vector(s: Stream, names: list[str]) -> np.ndarray:
        return np.array([s.fn * s.comp_mol.get(n, 0.0) for n in names] + [s.T, s.P], dtype=float)

    @staticmethod
    def _to_stream(ref: Stream, names: list[str], x: np.ndarray, description: str) -> Stream:
        """Stream with the molar flows, T and P of the vector x and the thermodynamic setup of `ref`."""
        n = np.clip(np.asarray(x[:-2], float), 0.0, None)
        fn = float(n.sum())
        if fn <= 0.0:
            raise ValueError("The guessed recycle has no flow.")
        return Stream(description, fm=None, comp_mass=None, fn=fn,
                      comp_mol={nm: float(v) / fn for nm, v in zip(names, n) if v > 0.0}, V_frac=None,
                      T=float(x[-2]), P=float(x[-1]), Tref=ref.Tref, Pref=ref.Pref, gamma=None,
                      component_data=ref.component_data, kij=ref.kij, k_mode=ref.k_mode, method=ref.method,
                      forced_cubic=ref.forced_cubic, solver_options=ref.solver_options,
                      emit_warnings=ref.emit_warnings, psat_factor=ref.gamma_psat_factor)

    def _errors(self, guess: Stream | None, calc: Stream) -> dict:
        if guess is None:                                    # empty guess: the whole flow is the error
            return {'T': abs(calc.T), 'P': abs(calc.P), 'mass_flow': float(calc.fm)}
        return {'T': abs(calc.T - guess.T), 'P': abs(calc.P - guess.P), 'mass_flow': guess.deviation(calc)}

    def _ok(self, e: dict) -> bool:
        return e['T'] <= self.tol_T and e['P'] <= self.tol_P and e['mass_flow'] <= self.tol_mass

    # ------------------------------------------------------------------ vector algorithms

    def _next(self, x, g, state):
        """Next guess from the guess x and the calculated g(x). `state` keeps what the accelerations need."""
        if self.method == 'legacy' or state['k'] == 0:
            s = self.smoothing if self.method == 'legacy' else 1.0
            xn = (1.0 - s) * x + s * g
        elif self.method == 'wegstein':
            dx, dg = x - state['x'], g - state['g']
            with np.errstate(divide='ignore', invalid='ignore'):
                slope = np.where(np.abs(dx) > 1e-14 * (1.0 + np.abs(x)), dg / dx, 0.0)
                q = np.where(slope != 1.0, slope / (slope - 1.0), 0.0)
            q = np.clip(np.nan_to_num(q), _Q_MIN, _Q_MAX)
            xn = q * x + (1.0 - q) * g
        else:                                                # broyden, on scaled variables
            sc, H = state['scale'], state['H']
            u, r = x / sc, (g - x) / sc
            du, dr = u - state['x'] / sc, r - state['r']
            Hdr = H @ dr
            den = float(du @ Hdr)
            if abs(den) > 1e-14:
                H = H + np.outer(du - Hdr, du @ H) / den
            state['H'] = H
            xn = (u - H @ r) * sc
        state.update(x=x.copy(), g=g.copy(), r=(g - x) / state['scale'], k=state['k'] + 1)
        return xn

    # ------------------------------------------------------------------ driver

    def solve(self, loop, guess: Stream | None = None) -> RecycleResult:
        """Iterate ``calc = loop(guess)`` until guess and calc agree within the tolerances.

        guess: initial recycle stream. None (default, like DWSIM) calls ``loop(None)`` on the first pass: the
        flowsheet must then work with an empty recycle."""
        calc = loop(guess)
        names = self._names(calc)
        xg = None if guess is None else self._to_vector(guess, names)
        history, g_stream = [], guess
        state = dict(k=0)
        for it in range(1, self.max_iter + 1):
            err = self._errors(g_stream, calc)
            history.append((it, err['T'], err['P'], err['mass_flow']))
            if g_stream is not None and self._ok(err):
                return RecycleResult(calc, True, it, err, self._tols(), self.method, history, xg)
            g = self._to_vector(calc, names)
            if xg is None:                                   # first pass with an empty guess: take the calculated
                xn = g.copy()
                state.update(k=0, scale=np.where(np.abs(g) > 0, np.abs(g), 1.0), H=-np.eye(g.size))
                state.update(x=np.zeros_like(g), g=g.copy(), r=np.zeros_like(g))
                state['k'] = 0
            else:
                if 'scale' not in state:
                    state.update(scale=np.where(np.abs(g) > 0, np.abs(g), 1.0), H=-np.eye(g.size))
                xn = self._next(xg, g, state)
            xn[:-2] = np.clip(xn[:-2], 0.0, None)            # flows cannot be negative
            xg = xn
            g_stream = self._to_stream(calc, names, xg, 'Recycle guess')
            calc = loop(g_stream)
        err = self._errors(g_stream, calc)
        return RecycleResult(calc, self._ok(err), self.max_iter, err, self._tols(), self.method, history, xg)

    def _tols(self) -> dict:
        return {'T': self.tol_T, 'P': self.tol_P, 'mass_flow': self.tol_mass}

    # ------------------------------------------------------------------ pure vector solver (tests, other uses)

    def solve_vector(self, g, x0, tol: float = 1e-10):
        """Fixed point x = g(x) of a plain function with the selected method; returns (x, iterations, converged).
        Same update rules as `solve`, with one absolute tolerance on max |g(x) - x|."""
        x = np.asarray(x0, float).copy()
        state = dict(k=0, scale=np.where(np.abs(x) > 0, np.abs(x), 1.0), H=-np.eye(x.size))
        for it in range(1, self.max_iter + 1):
            gx = np.asarray(g(x), float)
            if np.max(np.abs(gx - x)) <= tol:
                return gx, it, True
            x = self._next(x, gx, state)
        return x, self.max_iter, False
