"""Schwarzschild null geodesics, thin-disk emission and redshift.

This is the NumPy reference implementation. The GLSL tracer in
``shaders/geodesic.glsl`` follows the same algorithm step by step, so this
module is used for the unit tests and to check the GPU output.

Units: G = c = 1 and the Schwarzschild radius r_s = 2M = 1.

Every photon moves in a plane through the centre. Inside that plane the orbit
u(phi) = 1/r(phi) satisfies the Binet equation

    u'' = -u + (3/2) r_s u^2

which is integrated with RK4 in phi. Being second order, it passes through
turning points (periapsis) without special handling, so rays can bend around
the hole, escape to infinity (u -> 0) or fall through the horizon (u -> 1).
"""
from __future__ import annotations

import numpy as np

R_S = 1.0
MASS = 0.5 * R_S
R_PHOTON = 1.5 * R_S
R_ISCO = 3.0 * R_S
B_CRIT = 1.5 * np.sqrt(3.0) * R_S  # critical impact parameter, ~2.598 r_s

ESCAPED, CAPTURED, DISK, UNRESOLVED = 0, 1, 2, 3

# default integrator settings, shared with the shader (see blackhole/gpu.py)
H_MAX = 0.03     # maximum step in phi (rad)
DU_MAX = 0.02    # maximum relative change of u per step (for u >= 0.05)
MAX_STEPS = 4000


# ---------------------------------------------------------------- integrator

def binet_rhs(u, w):
    """Right-hand side of the Binet equation, w = du/dphi."""
    return w, -u + 1.5 * R_S * u * u


def rk4_step(u, w, h):
    k1u, k1w = binet_rhs(u, w)
    k2u, k2w = binet_rhs(u + 0.5 * h * k1u, w + 0.5 * h * k1w)
    k3u, k3w = binet_rhs(u + 0.5 * h * k2u, w + 0.5 * h * k2w)
    k4u, k4w = binet_rhs(u + h * k3u, w + h * k3w)
    return (u + h / 6.0 * (k1u + 2 * k2u + 2 * k3u + k4u),
            w + h / 6.0 * (k1w + 2 * k2w + 2 * k3w + k4w))


def step_size(u, w, h_max=H_MAX, du_max=DU_MAX):
    """Step in phi that limits the change of u to du_max * max(u, 0.05)."""
    return np.minimum(h_max, du_max * np.maximum(u, 0.05) / (np.abs(w) + 1e-9))


def hermite(u0, w0, u1, w1, h, t):
    """Cubic Hermite interpolation of u inside one step, t in [0, 1]."""
    t2 = t * t
    t3 = t2 * t
    return ((2 * t3 - 3 * t2 + 1) * u0 + (t3 - 2 * t2 + t) * h * w0
            + (-2 * t3 + 3 * t2) * u1 + (t3 - t2) * h * w1)


def trace_impact(b, h_max=0.01, du_max=0.005, max_steps=1_000_000):
    """Photon arriving from infinity with impact parameter b.

    Returns ``(captured, swept_phi)``. For an escaping ray the deflection
    angle is ``swept_phi - pi``.
    """
    u, w, phi = 0.0, 1.0 / b, 0.0
    for _ in range(max_steps):
        h = float(step_size(u, w, h_max, du_max))
        u1, w1 = rk4_step(u, w, h)
        if u1 >= 1.0 / R_S:
            return True, phi + h
        if u1 <= 0.0:
            return False, phi + h * u / (u - u1)
        u, w, phi = u1, w1, phi + h
    raise RuntimeError("ray did not terminate")


# ---------------------------------------------------------- disk and redshift

def novikov_thorne_flux(r):
    """Emitted flux of a Novikov-Thorne disk around a Schwarzschild hole.

    Page and Thorne (1974) with spin a = 0, up to a constant factor. Zero at
    the ISCO (r = 3 r_s), falls off as r^-3 far out.
    """
    r = np.asarray(r, dtype=np.float64)
    x = np.sqrt(r / MASS)
    s3 = np.sqrt(3.0)
    x0 = np.sqrt(6.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_term = np.log((x + s3) * (x0 - s3) / ((x - s3) * (x0 + s3)))
        f = (x - x0 + 0.5 * s3 * log_term) / ((x * x - 3.0) * x ** 5)
    return np.where(r > R_ISCO, f, 0.0)


def flux_peak(r_out):
    r = np.linspace(R_ISCO, r_out, 20001)
    f = novikov_thorne_flux(r)
    i = int(np.argmax(f))
    return float(r[i]), float(f[i])


def keplerian_omega(r):
    """Coordinate angular velocity d(phi)/dt of a circular orbit."""
    return np.sqrt(MASS / r ** 3)


def redshift_factor(r, lz, r_obs):
    """g = nu_obs / nu_emit for a photon with E = 1 and axial angular
    momentum lz, emitted by gas on a prograde circular orbit at radius r and
    received by a static observer at r_obs.

        g = sqrt(1 - 3M/r) / ((1 - Omega lz) sqrt(1 - r_s/r_obs))

    It contains both the gravitational shift and the Doppler shift.
    """
    omega = keplerian_omega(r)
    return np.sqrt(1.0 - 1.5 * R_S / r) / ((1.0 - omega * lz) * np.sqrt(1.0 - R_S / r_obs))


def shadow_angular_radius(r_obs):
    """Angular radius of the shadow seen by a static observer at r_obs."""
    return float(np.arcsin(B_CRIT * np.sqrt(1.0 - R_S / r_obs) / r_obs))


def impact_from_angle(alpha, r_obs):
    """Impact parameter of a ray leaving a static observer at angle alpha
    from the radial direction."""
    return r_obs * np.sin(alpha) / np.sqrt(1.0 - R_S / r_obs)


# --------------------------------------------------------------------- camera

def camera_basis(r_obs, inclination_deg, azimuth_deg):
    """Observer position and orthonormal camera axes.

    The disk lies in the z = 0 plane; the inclination is measured from +z.
    The camera looks at the centre.
    """
    inc = np.radians(inclination_deg)
    az = np.radians(azimuth_deg)
    pos = r_obs * np.array([np.sin(inc) * np.cos(az), np.sin(inc) * np.sin(az), np.cos(inc)])
    fwd = -pos / r_obs
    up_world = np.array([0.0, 0.0, 1.0])
    if np.linalg.norm(np.cross(fwd, up_world)) < 1e-6:
        up_world = np.array([np.cos(az), np.sin(az), 0.0])
    right = np.cross(fwd, up_world)
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return pos, fwd, right, up


def camera_directions(width, height, fov_deg, fwd, right, up):
    """Unit ray directions (height, width, 3), row 0 at the top.

    fov_deg is the vertical field of view of a pinhole camera.
    """
    tan_half = np.tan(np.radians(fov_deg) / 2)
    aspect = width / height
    x = (2 * (np.arange(width) + 0.5) / width - 1) * aspect
    y = 1 - 2 * (np.arange(height) + 0.5) / height
    X, Y = np.meshgrid(x, y)
    d = fwd + tan_half * (X[..., None] * right + Y[..., None] * up)
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


# ------------------------------------------------------------- full tracer

def trace_rays(pos, dirs, r_in=R_ISCO, r_out=14.0, disk=True,
               h_max=H_MAX, du_max=DU_MAX, max_steps=MAX_STEPS):
    """Trace camera rays backwards from a static observer at ``pos``.

    ``dirs`` are unit directions in the observer's local orthonormal frame,
    shape (..., 3). The disk is geometrically thin and optically thick, so a
    ray stops at the first crossing of the z = 0 plane with r_in <= r <= r_out.

    Returns a dict of arrays with the input's leading shape:
    status (ESCAPED, CAPTURED, DISK, UNRESOLVED), order (index of the
    equatorial crossing that hit the disk, 0 = primary image), r, g,
    phi_disk (azimuth of the emission point) and dir (asymptotic direction
    of escaped rays).
    """
    shape = dirs.shape[:-1]
    d = dirs.reshape(-1, 3).astype(np.float64)
    n = d.shape[0]
    r0 = float(np.linalg.norm(pos))
    e1 = np.asarray(pos, dtype=np.float64) / r0

    cosa = d @ e1
    perp = d - cosa[:, None] * e1
    sina = np.linalg.norm(perp, axis=1)
    radial = sina < 1e-7
    if radial.any():
        helper = np.array([1.0, 0.0, 0.0]) if abs(e1[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        perp[radial] = np.cross(e1, helper)
        sina[radial] = 1e-7
    e2 = perp / np.linalg.norm(perp, axis=1, keepdims=True)

    u0 = 1.0 / r0
    u = np.full(n, u0)
    w = -u0 * np.sqrt(1.0 - u0) * cosa / sina
    b = r0 * sina / np.sqrt(1.0 - u0)
    lz = -b * np.cross(e1, e2)[:, 2]  # physical photon travels opposite to the traced ray

    # z(phi) is proportional to A cos(phi) + B sin(phi) = R cos(phi - phi_n)
    A = e1[2]
    B = e2[:, 2]
    planar = (A * A + B * B) < 1e-12
    phi_next = np.mod(np.arctan2(B, A) + 0.5 * np.pi, np.pi)
    phi_next[phi_next <= 1e-6] += np.pi
    phi_next[planar] = np.inf

    phi = np.zeros(n)
    crossings = np.zeros(n, dtype=np.int32)
    status = np.full(n, UNRESOLVED, dtype=np.int32)
    order = np.full(n, -1, dtype=np.int32)
    r_hit = np.zeros(n)
    phi_hit = np.zeros(n)
    g = np.zeros(n)
    esc_dir = np.zeros((n, 3))

    idx = np.arange(n)
    for _ in range(max_steps):
        if idx.size == 0:
            break
        ua, wa, pa = u[idx], w[idx], phi[idx]
        h = step_size(ua, wa, h_max, du_max)
        u1, w1 = rk4_step(ua, wa, h)
        p1 = pa + h
        done = np.zeros(idx.size, dtype=bool)

        cross = p1 >= phi_next[idx]
        if cross.any():
            ci = np.nonzero(cross)[0]
            gi = idx[ci]
            t = (phi_next[gi] - pa[ci]) / h[ci]
            uc = hermite(ua[ci], wa[ci], u1[ci], w1[ci], h[ci], t)
            with np.errstate(divide="ignore"):
                rc = np.where(uc > 0, 1.0 / np.maximum(uc, 1e-300), np.inf)
            hit = (uc > 0) & (uc < 1.0) & (rc >= r_in) & (rc <= r_out) & disk
            if hit.any():
                hi = ci[hit]
                gh = idx[hi]
                rh = rc[hit]
                ph = phi_next[gh]
                p = rh[:, None] * (np.cos(ph)[:, None] * e1 + np.sin(ph)[:, None] * e2[gh])
                status[gh] = DISK
                order[gh] = crossings[gh]
                r_hit[gh] = rh
                phi_hit[gh] = np.arctan2(p[:, 1], p[:, 0])
                g[gh] = redshift_factor(rh, lz[gh], r0)
                done[hi] = True
            miss = ci[~hit]
            crossings[idx[miss]] += 1
            phi_next[idx[miss]] += np.pi

        live = ~done
        cap = live & (u1 >= 1.0)
        status[idx[cap]] = CAPTURED
        esc = live & ~cap & (u1 <= 0.0)
        if esc.any():
            ei = np.nonzero(esc)[0]
            gi = idx[ei]
            pinf = pa[ei] + h[ei] * ua[ei] / (ua[ei] - u1[ei])
            esc_dir[gi] = np.cos(pinf)[:, None] * e1 + np.sin(pinf)[:, None] * e2[gi]
            status[gi] = ESCAPED
        done |= cap | esc

        keep = ~done
        u[idx[keep]] = u1[keep]
        w[idx[keep]] = w1[keep]
        phi[idx[keep]] = p1[keep]
        idx = idx[keep]

    return {
        "status": status.reshape(shape),
        "order": order.reshape(shape),
        "r": r_hit.reshape(shape),
        "g": g.reshape(shape),
        "phi_disk": phi_hit.reshape(shape),
        "dir": esc_dir.reshape(shape + (3,)),
        "b": b.reshape(shape),
    }
