"""Retro terminal frame for the `terminal` style.

Draws the rendered image inside an amber CRT window (title bar, prompt line,
readout footer, phosphor glow, scanlines). This is presentation only: the
readout numbers are passed in by the caller and come from the ray metadata.
"""
from __future__ import annotations

import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

BG = (10, 7, 0)
AMBER = (255, 176, 0)
AMBER_DIM = (168, 120, 15)
CYAN = (63, 242, 255)
BAR_TEXT = (11, 11, 11)

_FONT_CANDIDATES = (
    "courbd.ttf", "cour.ttf", "consolab.ttf", "consola.ttf",
    "DejaVuSansMono-Bold.ttf", "DejaVuSansMono.ttf", "LiberationMono-Bold.ttf",
)
_FONT_DIRS = (
    os.path.join(os.environ.get("WINDIR", "C:/Windows"), "Fonts"),
    "/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/truetype/liberation",
    "/Library/Fonts", "/System/Library/Fonts",
)


def load_font(size):
    for name in _FONT_CANDIDATES:
        for d in _FONT_DIRS:
            path = os.path.join(d, name)
            if os.path.exists(path):
                return ImageFont.truetype(path, size)
    return ImageFont.load_default(size)


def _text_layer(size, items, font):
    """items: (x, y, text, color). Returns an RGBA layer with glow."""
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    for x, y, text, color in items:
        d.text((x, y), text, font=font, fill=color + (255,))
    glow = layer.filter(ImageFilter.GaussianBlur(radius=max(2, font.size // 6)))
    return Image.alpha_composite(glow, layer)


def layout(width, height):
    """Layout shared by compose() and the renderer.

    Returns (scale, pad, bar, line, top, (view_x, view_y, view_w, view_h)).
    """
    s = height / 1080.0
    pad = round(36 * s)
    bar = round(44 * s)
    line = round(40 * s)
    top = bar + round(18 * s)
    foot = 2 * line + round(20 * s)
    vx0, vy0 = pad, top + line + round(10 * s)
    vw, vh = width - 2 * pad, height - vy0 - foot - round(10 * s)
    vw -= vw % 2
    vh -= vh % 2
    return s, pad, bar, line, top, (vx0, vy0, vw, vh)


def compose(view, width, height, prompt, readout, status="[ ONLINE ]",
            title="KDR-TERM  //  BLACKHOLE.EXE  //  SCHWARZSCHILD", legend=None):
    """Place `view` (H, W, 3 uint8) inside a terminal window of width x height.

    `readout` is a list of (label, value) pairs; `legend` an optional list of
    (text, color) chips drawn on the right of the footer.
    """
    s, pad, bar, line, top, (vx0, vy0, vw, vh) = layout(width, height)
    font = load_font(round(26 * s))
    small = load_font(round(20 * s))

    canvas = Image.new("RGB", (width, height), BG)
    d = ImageDraw.Draw(canvas)
    d.rectangle([0, 0, width, bar], fill=AMBER)
    d.text((pad, bar / 2), title, font=small, fill=BAR_TEXT, anchor="lm")
    d.text((width - pad, bar / 2), status, font=small, fill=BAR_TEXT, anchor="rm")

    view_img = Image.fromarray(view)
    if view_img.size != (vw, vh):
        view_img = view_img.resize((vw, vh), Image.Resampling.LANCZOS)
    canvas.paste(view_img, (vx0, vy0))
    d.rectangle([vx0 - 2, vy0 - 2, vx0 + vw + 1, vy0 + vh + 1], outline=AMBER_DIM, width=max(1, round(2 * s)))

    items = [(pad, top, "C:\\> ", AMBER), (pad + font.getlength("C:\\> "), top, prompt, AMBER)]
    fy = vy0 + vh + round(16 * s)
    x = pad
    for label, value in readout:
        items.append((x, fy, label, AMBER_DIM))
        x += font.getlength(label + " ")
        items.append((x, fy, value, AMBER))
        x += font.getlength(value + "   ")
    if legend:
        x = width - pad
        for text, color in reversed(legend):
            x -= font.getlength(text)
            items.append((x, fy, text, color))
            x -= font.getlength("  ")
    items.append((pad, fy + line, "C:\\> _", AMBER))
    canvas = Image.alpha_composite(canvas.convert("RGBA"), _text_layer((width, height), items, font)).convert("RGB")

    img = np.asarray(canvas).astype(np.float32) / 255.0
    rows = np.arange(height)
    period = max(3, round(3 * s))
    img *= np.where(rows % period == 0, 0.55, 1.0)[:, None, None]
    xs = np.linspace(0.0, 1.0, width)
    edge = np.clip(np.minimum(xs, 1 - xs) / 0.10, 0, 1) * 0.35 + 0.65
    img *= edge[None, :, None]
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)
