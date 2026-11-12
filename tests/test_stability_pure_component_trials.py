"""The Michelsen stability test must also try each pure component as a trial phase.

With PRSV2 (DWSIM kappa values) the N2 alpha is strongly extrapolated above Tr = 1 and the feed
can split into a vapor and a dense, N2-rich liquid. Trial phases started from Wilson K values do
not converge into that basin at these points; a trial started from pure N2 does. Reference
vapor fractions are DWSIM's (reference_data/dwsim/ammonia_feed_grid_prsv2.csv); the independent
analysis shows that the two-phase state has the lower Gibbs energy at every point.
"""
import pytest

from _dwsim_grid import stream

# (T [K], P [bar], DWSIM vapor fraction)
POINTS = [(350.0, 25.0, 0.81258), (350.0, 100.0, 0.76458), (350.0, 125.0, 0.76058),
          (320.0, 200.0, 0.96859), (350.0, 175.0, 0.75555)]


@pytest.mark.parametrize('T,P,V', POINTS, ids=[f'T{t:g}_P{p:g}' for t, p, _ in POINTS])
def test_dense_n2_liquid_is_found(T, P, V):
    s = stream(T, P, 'prsv2')
    assert s.props.V_frac == pytest.approx(V, abs=3e-4)
