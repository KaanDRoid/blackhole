"""Compare the GLSL tracer with the NumPy reference. Skipped without an
OpenGL 4.3 context."""
import numpy as np
import pytest

from blackhole import physics as ph

moderngl = pytest.importorskip("moderngl")


@pytest.fixture(scope="module")
def tracer():
    from blackhole.gpu import GpuTracer
    try:
        ctx = moderngl.create_standalone_context(require=430)
    except Exception as exc:  # noqa: BLE001 - context backends raise platform-specific exceptions
        pytest.skip(f"no OpenGL 4.3 context: {exc}")
    # Compilation or execution errors must fail, not turn into skips.
    return GpuTracer(context=ctx)


def test_loop_radiance_closes(tracer):
    from blackhole.gpu import Scene
    scene = Scene(width=128, height=72, inclination=84.0, fov=28.0)
    start, _ = tracer.render(scene)
    scene.time = 1.0
    end, _ = tracer.render(scene)
    assert np.allclose(start, end, atol=2e-5)


def test_gpu_matches_reference(tracer):
    from blackhole.gpu import Scene
    scene = Scene(width=160, height=90, inclination=84.0, fov=28.0, r_obs=30.0)
    _, aux = tracer.render(scene)
    pos, fwd, right, up = ph.camera_basis(scene.r_obs, scene.inclination, scene.azimuth)
    ref = ph.trace_rays(pos, ph.camera_directions(160, 90, scene.fov, fwd, right, up), r_out=scene.r_out)
    status = aux[..., 0].astype(int)
    assert (status == ref["status"]).mean() > 0.999
    both = (status == ph.DISK) & (ref["status"] == ph.DISK)
    assert np.median(np.abs(aux[..., 2][both] - ref["g"][both])) < 1e-4


def test_gpu_shadow_radius(tracer):
    from blackhole.gpu import Scene
    r_obs = 30.0
    alpha = ph.shadow_angular_radius(r_obs)
    fov = np.degrees(2 * alpha) * 1.5
    _, aux = tracer.render(Scene(width=256, height=256, r_obs=r_obs, inclination=60.0, fov=fov, disk=False))
    captured = aux[..., 0] == ph.CAPTURED
    t = np.tan(np.radians(fov) / 2)
    pix_area = (2 * t / 256) ** 2
    # small-angle estimate: area on the image plane ~ pi tan(alpha)^2
    measured = np.arctan(np.sqrt(captured.sum() * pix_area / np.pi))
    assert measured == pytest.approx(alpha, rel=0.01)
