"""Numerical checks of the GPU tracer. Prints the numbers quoted in the README.

1. Shadow size: render with the disk switched off, measure the captured region
   in the image and convert it to an impact parameter.
2. GPU vs NumPy reference on the same camera rays.
3. Doppler asymmetry of the primary disk image.
4. Pixel counts of the primary, secondary and higher-order disk images.
"""
from __future__ import annotations

import math
import time

import numpy as np

from blackhole import physics as ph
from blackhole.gpu import GpuTracer, Scene


def tangent_coords(width, height, fov):
    t = math.tan(math.radians(fov) / 2)
    x = (2 * (np.arange(width) + 0.5) / width - 1) * (width / height) * t
    y = (1 - 2 * (np.arange(height) + 0.5) / height) * t
    return np.meshgrid(x, y)


def shadow_check(tracer, r_obs=30.0, size=1024):
    alpha_th = ph.shadow_angular_radius(r_obs)
    fov = math.degrees(2 * alpha_th) * 1.6
    scene = Scene(width=size, height=size, r_obs=r_obs, inclination=60.0, fov=fov, disk=False)
    _, aux = tracer.render(scene)
    captured = aux[..., 0] == ph.CAPTURED
    X, Y = tangent_coords(size, size, fov)
    rho2 = X ** 2 + Y ** 2
    pix = (X[0, 1] - X[0, 0]) ** 2
    # solid angle of each pixel on a pinhole image plane at unit distance
    omega = (pix / (1 + rho2) ** 1.5)[captured].sum()
    alpha_area = math.acos(1 - omega / (2 * math.pi))
    # edge along 720 rays from the centre (sub-pixel bisection on the mask)
    t = math.tan(math.radians(fov) / 2)
    edges = []
    for th in np.linspace(0, 2 * np.pi, 720, endpoint=False):
        lo, hi = 0.0, t * 0.99
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            px = int((mid * math.cos(th) / t + 1) * size / 2)
            py = int((1 - mid * math.sin(th) / t) * size / 2)
            if captured[py, px]:
                lo = mid
            else:
                hi = mid
        edges.append(math.atan(lo))
    edges = np.array(edges)
    pix_angle = math.atan(2 * t / size)
    b_area = ph.impact_from_angle(alpha_area, r_obs)
    b_edge = ph.impact_from_angle(edges.mean(), r_obs)
    return {
        "r_obs": r_obs, "size": size, "pix_angle_deg": math.degrees(pix_angle),
        "alpha_theory_deg": math.degrees(alpha_th),
        "alpha_area_deg": math.degrees(alpha_area),
        "alpha_edge_deg": math.degrees(edges.mean()),
        "alpha_edge_std_deg": math.degrees(edges.std()),
        "b_area": b_area, "b_edge": b_edge, "b_crit": ph.B_CRIT,
        "radius_theory_px": math.tan(alpha_th) / (2 * t / size),
        "radius_area_px": math.tan(alpha_area) / (2 * t / size),
        "relative_b_error": abs(b_area / ph.B_CRIT - 1),
        "unresolved": int((aux[..., 0] == ph.UNRESOLVED).sum()),
    }


def gpu_vs_numpy(tracer, w=192, h=108, inclination=84.0, fov=28.0, r_obs=30.0):
    scene = Scene(width=w, height=h, r_obs=r_obs, inclination=inclination, fov=fov)
    _, aux = tracer.render(scene)
    pos, fwd, right, up = ph.camera_basis(r_obs, inclination, 0.0)
    dirs = ph.camera_directions(w, h, fov, fwd, right, up)
    t0 = time.perf_counter()
    ref = ph.trace_rays(pos, dirs, r_out=scene.r_out)
    t_cpu = time.perf_counter() - t0
    st_gpu = aux[..., 0].astype(int)
    same = st_gpu == ref["status"]
    both = (st_gpu == ph.DISK) & (ref["status"] == ph.DISK)
    dg = np.abs(aux[..., 2][both] - ref["g"][both])
    dr = np.abs(aux[..., 3][both] - ref["r"][both]) / ref["r"][both]
    return {
        "pixels": w * h, "status_agree": same.mean(),
        "order_agree": (aux[..., 1][both].astype(int) == ref["order"][both]).mean(),
        "g_absdiff_median": float(np.median(dg)), "g_absdiff_p99": float(np.percentile(dg, 99)),
        "r_reldiff_median": float(np.median(dr)), "cpu_seconds": t_cpu,
    }


def doppler_and_orders(tracer, w=1920, h=1080, inclination=84.0, fov=28.0, r_obs=30.0):
    scene = Scene(width=w, height=h, r_obs=r_obs, inclination=inclination, fov=fov)
    _, aux = tracer.render(scene)
    status, order, g, r = aux[..., 0], aux[..., 1], aux[..., 2], aux[..., 3]
    _, fmax = ph.flux_peak(scene.r_out)
    lum = g ** 4 * ph.novikov_thorne_flux(r) / fmax
    disk = status == ph.DISK
    prim = disk & (order == 0)
    cols = np.arange(w)[None, :].repeat(h, 0)
    left = prim & (cols < w // 2)
    right = prim & (cols >= w // 2)
    return {
        "g_min": float(g[disk].min()), "g_max": float(g[disk].max()),
        "g_mean_left": float(g[left].mean()), "g_mean_right": float(g[right].mean()),
        "lum_mean_left": float(lum[left].mean()), "lum_mean_right": float(lum[right].mean()),
        "pixels_order0": int(prim.sum()), "pixels_order1": int((disk & (order == 1)).sum()),
        "pixels_order2plus": int((disk & (order >= 2)).sum()),
        "pixels_captured": int((status == ph.CAPTURED).sum()),
        "pixels_unresolved": int((status == ph.UNRESOLVED).sum()),
    }


def main():
    tracer = GpuTracer()
    for name, fn in (("shadow", shadow_check), ("gpu_vs_numpy", gpu_vs_numpy),
                     ("disk", doppler_and_orders)):
        res = fn(tracer)
        print(f"[{name}]")
        for k, v in res.items():
            print(f"  {k:20s} {v:.6g}" if isinstance(v, float) else f"  {k:20s} {v}")


if __name__ == "__main__":
    main()
