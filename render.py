"""Render a Schwarzschild black hole with a thin accretion disk.

Examples:
    python render.py                                   # 1920x1080 still, terminal style
    python render.py --style cyberpunk --out neon.png
    python render.py --style physical --out physical.png
    python render.py --frames 150 --fps 30 --width 1280 --height 720 --out loop.mp4 --webp loop.webp
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
from PIL import Image

from blackhole import frame, post
from blackhole.gpu import GpuTracer, Scene


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--ss", type=int, default=2, help="supersampling factor per axis")
    p.add_argument("--style", choices=["terminal", "cyberpunk", "physical"], default="terminal")
    p.add_argument("--inclination", type=float, default=84.0, help="degrees from the disk normal")
    p.add_argument("--azimuth", type=float, default=0.0)
    p.add_argument("--r-obs", type=float, default=30.0, help="observer radius in r_s")
    p.add_argument("--fov", type=float, default=28.0, help="vertical field of view, degrees")
    p.add_argument("--r-out", type=float, default=14.0, help="outer disk radius in r_s")
    p.add_argument("--t-peak", type=float, default=7000.0, help="physical style: peak disk temperature (K)")
    p.add_argument("--time", type=float, default=0.0, help="loop phase for a still frame, 0..1")
    p.add_argument("--exposure", type=float, default=None, help="default: automatic")
    p.add_argument("--bloom", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--chromatic", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--scanlines", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--frames", type=int, default=0, help="number of animation frames (0 = still image)")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--sway", type=float, default=6.0, help="animation: camera azimuth sway amplitude (deg)")
    p.add_argument("--out", default=None, help="output .png (still) or .mp4 (animation)")
    p.add_argument("--webp", default=None, help="animation: also write an animated WebP")
    p.add_argument("--webp-width", type=int, default=640)
    p.add_argument("--webp-quality", type=int, default=70)
    a = p.parse_args(argv)
    if min(a.width, a.height, a.ss, a.fps, a.webp_width) < 1 or a.frames < 0:
        p.error("dimensions, supersampling and frame rate must be positive; frames must be nonnegative")
    if not 0 <= a.inclination <= 89.5 or not 0 < a.fov < 90:
        p.error("inclination must be in [0, 89.5] degrees and FOV in (0, 90)")
    if a.r_obs <= a.r_out or a.r_out <= 3:
        p.error("require r_obs > r_out > 3 r_s")
    if a.exposure is not None and a.exposure <= 0:
        p.error("exposure must be positive")
    if a.frames and a.out and not a.out.lower().endswith(".mp4"):
        p.error("animation output must end in .mp4")
    if a.frames and (a.width % 2 or a.height % 2):
        p.error("MP4 output dimensions must be even")
    if a.frames and (a.inclination + abs(0.35 * a.sway) > 89.5 or a.inclination - abs(0.35 * a.sway) < 0):
        p.error("camera sway must keep the inclination between 0 and 89.5 degrees")
    cyber = a.style == "cyberpunk"
    for name, default in (("bloom", True), ("chromatic", cyber), ("scanlines", cyber)):  # terminal: frame adds scanlines
        if getattr(a, name) is None:
            setattr(a, name, default)
    if a.out is None:
        a.out = "blackhole.mp4" if a.frames else f"blackhole_{a.style}.png"
    return a


def view_size(a):
    """Size of the traced image; the terminal style traces only its viewport."""
    if a.style == "terminal":
        return frame.layout(a.width, a.height)[-1][2:]
    return a.width, a.height


def make_scene(a, phase=0.0):
    sway = np.sin(2 * np.pi * phase)
    w, h = view_size(a)
    return Scene(width=w, height=h, r_obs=a.r_obs,
                 inclination=a.inclination + 0.35 * a.sway * np.cos(2 * np.pi * phase) * (a.frames > 0),
                 azimuth=a.azimuth + a.sway * sway * (a.frames > 0),
                 fov=a.fov, r_out=a.r_out, style=a.style, time=phase if a.frames else a.time,
                 t_peak=a.t_peak)


def finish(a, color, aux, exposure):
    hdr = post.downsample(color[..., :3], a.ss)
    img = post.finish(hdr, exposure, do_bloom=a.bloom, do_ca=a.chromatic,
                      do_scanlines=a.scanlines, do_vignette=a.style != "physical")
    if a.style != "terminal":
        return img
    primary = (aux[..., 0] == 2) & (aux[..., 1] == 0)
    g = aux[..., 2][primary]
    g_range = f"{g.min():.2f}..{g.max():.2f}" if g.size else "n/a"
    return frame.compose(
        img, a.width, a.height,
        prompt=f"blackhole.exe --metric schwarzschild --incl {a.inclination:.0f} --r-obs {a.r_obs:.0f}",
        readout=[("B_CRIT", f"{1.5 * np.sqrt(3):.4f}"), ("ISCO", "3.000"), ("G", g_range)],
        legend=[("[ PRIMARY ]", frame.AMBER), ("[ SECONDARY+ ]", frame.CYAN)])


def check_resolved(aux):
    count = int(np.count_nonzero(aux[..., 0] == 3))
    if count:
        raise RuntimeError(f"{count} rays exhausted the step budget")


def render_still(a, tracer):
    t0 = time.perf_counter()
    color, aux = tracer.render(make_scene(a), ss=a.ss)
    check_resolved(aux)
    t1 = time.perf_counter()
    exposure = a.exposure or post.auto_exposure(color)
    img = finish(a, color, aux, exposure)
    Image.fromarray(img).save(a.out)
    t2 = time.perf_counter()
    print(f"{a.out}: {a.width}x{a.height} ss={a.ss}  trace {t1 - t0:.2f} s, post {t2 - t1:.2f} s")


def render_animation(a, tracer):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required for animation output")
    tmp = tempfile.mkdtemp(prefix="bh_frames_", dir=os.environ.get("TMPDIR"))
    exposure = a.exposure
    t_start = time.perf_counter()
    for i in range(a.frames):
        color, aux = tracer.render(make_scene(a, i / a.frames), ss=a.ss)
        check_resolved(aux)
        if exposure is None:
            exposure = post.auto_exposure(color)  # fixed from frame 0, no flicker
        img = finish(a, color, aux, exposure)
        Image.fromarray(img).save(os.path.join(tmp, f"f{i:04d}.png"))
        if (i + 1) % 15 == 0 or i == 0:
            print(f"frame {i + 1}/{a.frames}", flush=True)
    print(f"rendered {a.frames} frames in {time.perf_counter() - t_start:.1f} s", flush=True)
    try:
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-framerate", str(a.fps),
                        "-i", os.path.join(tmp, "f%04d.png"), "-c:v", "libx264", "-preset", "slow",
                        "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", a.out], check=True)
        if a.webp:
            subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", a.out,
                            "-vf", f"fps=15,scale={a.webp_width}:-1:flags=lanczos",
                            "-c:v", "libwebp_anim", "-compression_level", "2", "-q:v", str(a.webp_quality),
                            "-loop", "0", a.webp], check=True)
    except BaseException:
        print("encoding failed; frames retained at", tmp)
        raise
    else:
        shutil.rmtree(tmp)
    print(f"wrote {a.out}" + (f" and {a.webp}" if a.webp else ""))
    print(f"total including encoding: {time.perf_counter() - t_start:.1f} s")


def main(argv=None):
    a = parse_args(argv)
    tracer = GpuTracer()
    if a.frames:
        render_animation(a, tracer)
    else:
        render_still(a, tracer)


if __name__ == "__main__":
    sys.exit(main())
