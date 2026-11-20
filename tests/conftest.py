import pandas as pd
import pytest


@pytest.fixture
def h2():
    return pd.DataFrame({'Component': ['Hydrogen'], 'Tc (K)': [33.1844], 'Pc (bar)': [13.1171],
                         'w': [-0.20952], 'frac_mol': [1.0]})


@pytest.fixture
def water(h2):
    return h2.assign(Component='Water', **{'Tc (K)': 647.1, 'Pc (bar)': 220.64, 'w': 0.344})


@pytest.fixture
def h2_n2_nh3():
    """Haber-Bosch feed (literature critical constants)."""
    return pd.DataFrame({'Component': ['hydrogen', 'nitrogen', 'ammonia'],
                         'Tc (K)': [33.19, 126.192, 405.56],
                         'Pc (bar)': [13.13, 33.958, 113.63],
                         'w': [-0.216, 0.0372, 0.2526],
                         'frac_mol': [0.726, 0.242, 0.032]})


@pytest.fixture
def nh3():
    return pd.DataFrame({'Component': ['ammonia'], 'Tc (K)': [405.56], 'Pc (bar)': [113.63],
                         'w': [0.2526], 'frac_mol': [1.0]})


@pytest.fixture
def feed_with_argon(h2_n2_nh3):
    """Same feed plus an argon component at z = 0 (must not change the result)."""
    ar = pd.DataFrame({'Component': ['argon'], 'Tc (K)': [150.687], 'Pc (bar)': [48.63],
                       'w': [-0.002], 'frac_mol': [0.0]})
    return pd.concat([h2_n2_nh3, ar], ignore_index=True)

_ROWS = []

@pytest.fixture
def cmp(request):
    """Compares vs expected, then asserts."""
    def _cmp(label, got, ref, abs=None, rel=None):
        tol = max(abs or 0.0, (rel or 0.0) * __builtins__['abs'](ref)) if (abs or rel) else 1e-6 * __builtins__['abs'](ref)
        diff = got - ref
        _ROWS.append((request.node.name, label, got, ref, diff,
                      100 * diff / ref if ref else float('nan'), tol))
        assert got == pytest.approx(ref, abs=abs, rel=rel), f"{label}: {got} vs {ref}"
    return _cmp


def pytest_terminal_summary(terminalreporter):
    if not _ROWS:
        return
    tr = terminalreporter
    tr.section("Obtained vs DWSIM")
    tr.write_line(f"{'test':45s} {'variable':22s} {'thermo-sim':>12s} {'DWSIM':>12s} {'dif':>10s} {'%':>8s} {'tol':>9s} {'tol':>8s}")
    for t, l, g, r, d, p, tol in _ROWS:
        uso = f"{100*__builtins__['abs'](d)/tol:5.0f}%" if tol else "-"
        tr.write_line(f"{t[:45]:45s} {l[:22]:22s} {g:12.5g} {r:12.5g} {d:10.3g} {p:8.3f} {tol:9.2g} {uso:>8s}")