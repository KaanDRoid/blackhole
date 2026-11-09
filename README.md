# blackhole

Small Python experiments with gravitational lensing. There are two separate parts:

1. An interactive GPU demo (`main.py`) that renders thin-lens deflection of a starfield by one or more point masses, using a GLSL compute shader through ModernGL and a GLFW window.
2. Renderers for null geodesics in the equatorial plane of a Schwarzschild black hole, integrated with RK4. There is a fixed-step CPU reference, adaptive-step CPU variants, and a compute-shader port of the adaptive integrator.

## How it works

### Thin-lens demo (`main.py`, `shaders/lensing.comp`)

Each pixel `x` (normalized screen coordinates) is mapped back to a source position

```
src = x - sum_i Re_i^2 * (x - x_i) / |x - x_i|^2
```

where `x_i` and `Re_i` are the position and Einstein radius of lens `i` (up to 8 lenses). The source position is used to sample a procedural starfield texture. The result is written to an image by the compute shader and drawn on a fullscreen quad (`shaders/quad.vert`, `shaders/quad.frag`).

Controls:

| Key | Action |
| --- | --- |
| N / M | single-lens / multi-lens mode |
| Arrow keys | move the active lens |
| C | cycle the active lens (multi mode) |
| Q / E | decrease / increase the Einstein radius of the active lens |
| + / - | add / remove a lens (multi mode, max 8) |
| T / G | scale the background up / down |
| S | save `screenshot.png` |
| Esc | quit |

### Schwarzschild geodesics

All geodesic scripts use the same simplified model (units with `r_s = 1`, `E = 1`):

- the screen x coordinate is mapped to an impact parameter `b` in `[-6, 6]`, with `L = |b|`
- the orbit equation `dr/dphi = r^2 / L * sqrt(E^2 - (1 - r_s/r) * L^2 / r^2)` is integrated with RK4, marching inward from the observer radius
- a ray is captured (black pixel) if it reaches `r <= r_s`; otherwise the accumulated `phi` picks a color from a periodic color wheel

The image only depends on the x coordinate, so it shows vertical bands. The black band in the middle corresponds to captured rays. For a Schwarzschild black hole the critical impact parameter is `3*sqrt(3)/2 * r_s ~ 2.6`, which is about 43% of the `[-6, 6]` range. `run_compute_geodesic.py` gives 43.25% black pixels, which matches.

Limitation: the integration only moves toward smaller `r` and does not continue the ray past the turning point, so the colors of escaping rays are not an accurate deflection map. The width of the capture region is the meaningful part of the output. The coarse quick preset (tolerance `1e-2`) gives 41.7% black pixels and also leaves a few columns near `b = 0` uncaptured.

| Script | What it does | Output |
| --- | --- | --- |
| `geodesic_rk4.py` | fixed-step RK4 reference, 800x400, `r_obs = 100` | `geodesic_out.png` |
| `geodesic_rk4_adaptive.py` | adaptive step (step halving error control), CLI options `--quick`, `--width`, `--height`, `--max-steps` | `geodesic_adaptive_out.png` or `geodesic_adaptive_quick.png` |
| `geodesic_rk4_medium.py` | adaptive, fixed settings: 400x200, `r_obs = 80`, tol `1e-3` | `geodesic_adaptive_medium.png` |
| `geodesic_rk4_quick.py` | adaptive, fixed settings: 240x120, `r_obs = 50`, tol `1e-2` | `geodesic_adaptive_quick.png` |
| `geodesic_adaptive_quick.py` | runs `geodesic_rk4_quick.py` (falls back to the medium script) | same as above |
| `run_compute_geodesic.py` | runs `shaders/geodesic_rk4.comp` (GPU port of the adaptive integrator) in a headless context | `geodesic_compute_out.png` |
| `analyze_image.py` | prints size, sample pixels, channel histograms and the black pixel fraction of a PNG | stdout |

`geodesic_rk4_adaptive.py` is the main CPU version; `run_compute_geodesic.py` is the fast one. The CPU scripts are pure Python per-pixel loops and are slow (several minutes even for the small quick settings), so the GPU version is the practical way to get a full 800x400 image (about 2 seconds on an AMD Radeon 780M).

## Requirements

- Python 3.10+
- An OpenGL 4.3 capable GPU/driver for `main.py` and `run_compute_geodesic.py` (compute shaders)

```bash
pip install -r requirements.txt
```

The CPU renderers only need Pillow; `analyze_image.py` also needs NumPy.

## Running

```bash
# interactive thin-lens demo
python main.py

# Schwarzschild capture image on the GPU
python run_compute_geodesic.py
python analyze_image.py geodesic_compute_out.png

# CPU versions (slow)
python geodesic_rk4_adaptive.py --quick
python geodesic_rk4.py
```

All scripts write their PNG files to the current working directory.

## Layout

```
main.py                     interactive thin-lens demo (GLFW + ModernGL)
run_compute_geodesic.py     GPU geodesic renderer (headless)
geodesic_rk4.py             CPU fixed-step reference
geodesic_rk4_adaptive.py    CPU adaptive renderer with CLI options
geodesic_rk4_medium.py      CPU adaptive, medium preset
geodesic_rk4_quick.py       CPU adaptive, quick preset
geodesic_adaptive_quick.py  wrapper for the quick preset
analyze_image.py            image statistics helper
shaders/
  lensing.comp              thin-lens deflection compute shader
  geodesic_rk4.comp         adaptive RK4 geodesic compute shader
  quad.vert, quad.frag      fullscreen quad
requirements.txt
LICENSE
```

## License

MIT, see [LICENSE](LICENSE).
