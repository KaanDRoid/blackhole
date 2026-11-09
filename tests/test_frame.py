import numpy as np

from blackhole import frame


def test_layout_viewport_fits_and_is_even():
    for w, h in ((1920, 1080), (1280, 720)):
        _, pad, bar, _, _, (x, y, vw, vh) = frame.layout(w, h)
        assert vw % 2 == 0 and vh % 2 == 0
        assert x == pad and y > bar
        assert x + vw <= w and y + vh < h


def test_compose_returns_full_frame():
    w, h = 1280, 720
    vw, vh = frame.layout(w, h)[-1][2:]
    view = np.full((vh, vw, 3), 200, dtype=np.uint8)
    out = frame.compose(view, w, h, prompt="test", readout=[("A", "1")],
                        legend=[("[ X ]", frame.CYAN)])
    assert out.shape == (h, w, 3) and out.dtype == np.uint8
    # title bar is amber, viewport keeps the traced image content
    assert out[5, w // 2, 0] > 150 and out[5, w // 2, 2] < 60
    x, y = frame.layout(w, h)[-1][:2]
    assert out[y + vh // 2, x + vw // 2].mean() > 100
