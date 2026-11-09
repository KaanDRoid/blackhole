import numpy as np
import pytest

from blackhole import physics as ph


@pytest.mark.parametrize("eps", [1e-2, 1e-3, 1e-4])
def test_capture_threshold(eps):
    captured_below, _ = ph.trace_impact(ph.B_CRIT * (1 - eps))
    captured_above, _ = ph.trace_impact(ph.B_CRIT * (1 + eps))
    assert captured_below
    assert not captured_above


def test_winding_grows_near_critical():
    # rays just above b_c circle the photon sphere; the swept angle grows like -ln(eps)
    sweeps = [ph.trace_impact(ph.B_CRIT * (1 + eps))[1] for eps in (1e-2, 1e-4, 1e-6)]
    assert sweeps[0] < sweeps[1] < sweeps[2]
    assert sweeps[2] > 4 * np.pi


@pytest.mark.parametrize("b", [100.0, 300.0, 1000.0])
def test_weak_field_deflection(b):
    captured, swept = ph.trace_impact(b)
    assert not captured
    deflection = swept - np.pi
    assert deflection == pytest.approx(2 * ph.R_S / b, rel=0.02)
    # including the second-order term 15 pi M^2 / (4 b^2)
    second = 2 * ph.R_S / b + 15 * np.pi * ph.MASS ** 2 / (4 * b * b)
    assert deflection == pytest.approx(second, rel=1e-3)


def test_photon_sphere_is_circular_orbit():
    u, w = 1 / ph.R_PHOTON, 0.0
    for _ in range(300):  # one full turn with h = 0.02
        u, w = ph.rk4_step(u, w, 0.02)
    assert abs(1 / u - ph.R_PHOTON) < 1e-6


def test_camera_rays_at_shadow_edge():
    r_obs = 30.0
    alpha = ph.shadow_angular_radius(r_obs)
    pos = np.array([r_obs, 0.0, 0.0])
    angles = np.array([alpha * (1 - 1e-3), alpha * (1 + 1e-3)])
    # rays leave towards the hole (-x) at angle alpha, tilted towards +z
    dirs = np.stack([-np.cos(angles), np.zeros(2), np.sin(angles)], axis=1)
    res = ph.trace_rays(pos, dirs, disk=False)
    assert list(res["status"]) == [ph.CAPTURED, ph.ESCAPED]
    assert res["b"] == pytest.approx(ph.impact_from_angle(angles, r_obs), rel=1e-9)
    assert res["b"][0] < ph.B_CRIT < res["b"][1]


def test_escaped_direction_matches_weak_deflection():
    # observer far away, ray passing at b = 200: the outgoing direction is
    # rotated by 2 r_s / b towards the hole relative to a straight line
    r_obs = 1e5
    b = 200.0
    alpha = np.arcsin(b * np.sqrt(1 - 1 / r_obs) / r_obs)
    pos = np.array([r_obs, 0.0, 0.0])
    d = np.array([[-np.cos(alpha), np.sin(alpha), 0.0]])
    res = ph.trace_rays(pos, d, disk=False, h_max=0.005)
    out = res["dir"][0]
    straight = d[0]
    bend = np.arccos(np.clip(out @ straight, -1, 1))
    assert bend == pytest.approx(2 / b, rel=0.03)


def test_novikov_thorne_flux_matches_page_thorne_integral():
    # direct numerical evaluation of the Page-Thorne integral (units M = 1)
    rs = np.linspace(6.0, 80.0, 400001)
    E = (1 - 2 / rs) / np.sqrt(1 - 3 / rs)
    L = np.sqrt(rs) / np.sqrt(1 - 3 / rs)
    Om = rs ** -1.5
    integrand = (E - Om * L) * np.gradient(L, rs)
    integral = np.concatenate([[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(rs))])
    F = -np.gradient(Om, rs) / (rs * (E - Om * L) ** 2) * integral
    closed = ph.novikov_thorne_flux(rs * ph.MASS)
    sel = slice(2000, None, 20000)
    ratio = F[sel] / closed[sel]
    assert np.allclose(ratio, ratio[0], rtol=1e-3)
    assert ph.novikov_thorne_flux(ph.R_ISCO) == 0.0


def test_redshift_face_on_is_purely_gravitational():
    # observer on the axis: every photon has lz = 0, so g has no Doppler part
    r_obs = 30.0
    pos, fwd, right, up = ph.camera_basis(r_obs, 0.0, 0.0)
    dirs = ph.camera_directions(64, 64, 40.0, fwd, right, up)
    res = ph.trace_rays(pos, dirs)
    disk = res["status"] == ph.DISK
    assert disk.sum() > 100
    expected = np.sqrt(1 - 1.5 / res["r"][disk]) / np.sqrt(1 - 1 / r_obs)
    assert np.allclose(res["g"][disk], expected, rtol=1e-6)


def test_approaching_side_is_blueshifted():
    # nearly edge-on view from +x. The disk rotates counter-clockwise about +z,
    # so the gas velocity at azimuth phi is along (-sin phi, cos phi, 0) and
    # gas with a positive x velocity moves towards the camera.
    r_obs = 30.0
    pos, fwd, right, up = ph.camera_basis(r_obs, 85.0, 0.0)
    dirs = ph.camera_directions(160, 90, 30.0, fwd, right, up)
    res = ph.trace_rays(pos, dirs)
    prim = (res["status"] == ph.DISK) & (res["order"] == 0)
    phi = res["phi_disk"][prim]
    g = res["g"][prim]
    # velocity of the gas: v ~ (-sin phi, cos phi, 0); towards the camera if v . e_x > 0
    towards = -np.sin(phi) > 0.5
    away = -np.sin(phi) < -0.5
    assert g[towards].mean() > 1.0 > g[away].mean()
