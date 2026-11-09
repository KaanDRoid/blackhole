import numpy as np
import pytest

from blackhole import post
from blackhole.physics import camera_basis
from render import make_scene, parse_args


@pytest.mark.parametrize("args", [
    ["--width", "0"], ["--ss", "0"], ["--inclination", "90"],
    ["--r-obs", "10"], ["--frames", "-1"], ["--exposure", "-1"],
    ["--frames", "3", "--width", "101"],
])
def test_cli_rejects_invalid_geometry(args):
    with pytest.raises(SystemExit):
        parse_args(args)


def test_loop_camera_closes():
    a = parse_args(["--frames", "150"])
    first, last = make_scene(a, 0), make_scene(a, 1)
    b0 = camera_basis(first.r_obs, first.inclination, first.azimuth)
    b1 = camera_basis(last.r_obs, last.inclination, last.azimuth)
    assert np.allclose(b0, b1)


def test_gaussian_preserves_uniform_image():
    a = np.full((32, 40, 3), 0.37, dtype=np.float32)
    assert np.allclose(post.gaussian_blur(a, 3), a)


def test_post_is_finite_and_bounded():
    rng = np.random.default_rng(42)
    a = rng.random((40, 64, 3)).astype(np.float32) * 4
    img = post.finish(a, 1, do_ca=True, do_scanlines=True, do_vignette=True)
    assert img.dtype == np.uint8
    assert img.shape == a.shape
