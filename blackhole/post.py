"""Post-processing: downsampling, bloom, tone mapping and optional effects.

Everything here is image processing applied after the physics; none of it
changes ray paths or redshift factors.
"""
from __future__ import annotations

import numpy as np


def downsample(img, ss):
    """Box filter for supersampled images (H*ss, W*ss, C) -> (H, W, C)."""
    if ss == 1:
        return img
    h, w, c = img.shape
    return img.reshape(h // ss, ss, w // ss, ss, c).mean(axis=(1, 3))


def gaussian_blur(img, sigma):
    """Gaussian blur of an (H, W, C) float image via FFT with edge padding."""
    pad = int(3 * sigma) + 1
    p = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    h, w = p.shape[:2]
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.rfftfreq(w)[None, :]
    kernel = np.exp(-2 * np.pi ** 2 * sigma ** 2 * (fx ** 2 + fy ** 2))
    out = np.empty_like(p)
    for c in range(p.shape[2]):
        out[..., c] = np.fft.irfft2(np.fft.rfft2(p[..., c]) * kernel, s=(h, w))
    return out[pad:-pad, pad:-pad]


def bloom(img, strength=0.6, sigmas=(3.0, 12.0, 40.0), weights=(0.5, 0.3, 0.2)):
    """Add a sum of Gaussian blurs of the bright part of the image."""
    scale = img.shape[0] / 1080.0
    bright = np.maximum(img - 0.15, 0.0)
    glow = sum(wt * gaussian_blur(bright, s * scale) for s, wt in zip(sigmas, weights))
    return img + strength * glow


def aces(x):
    """Narkowicz fit of the ACES filmic tone curve."""
    return np.clip(x * (2.51 * x + 0.03) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0)


def srgb_encode(x):
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def _resample_scaled(ch, s):
    """Sample one channel at positions scaled by 1/s around the image centre."""
    h, w = ch.shape
    cy, cx = (h - 1) / 2, (w - 1) / 2
    ys = cy + (np.arange(h) - cy) / s
    xs = cx + (np.arange(w) - cx) / s
    y0 = np.clip(np.floor(ys).astype(int), 0, h - 2)
    x0 = np.clip(np.floor(xs).astype(int), 0, w - 2)
    fy = np.clip(ys - y0, 0, 1)[:, None]
    fx = np.clip(xs - x0, 0, 1)[None, :]
    a = ch[y0][:, x0]
    b = ch[y0][:, x0 + 1]
    c = ch[y0 + 1][:, x0]
    d = ch[y0 + 1][:, x0 + 1]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def chromatic_aberration(img, amount=0.003):
    out = img.copy()
    out[..., 0] = _resample_scaled(img[..., 0], 1 + amount)
    out[..., 2] = _resample_scaled(img[..., 2], 1 - amount)
    return out


def scanlines(img, period=3.0, depth=0.12):
    h = img.shape[0]
    y = np.arange(h)[:, None, None]
    return img * (1 - depth * (0.5 + 0.5 * np.cos(2 * np.pi * y / (period * h / 1080.0))))


def vignette(img, strength=0.25):
    h, w = img.shape[:2]
    y = (np.arange(h) - h / 2) / (h / 2)
    x = (np.arange(w) - w / 2) / (h / 2)
    r2 = x[None, :] ** 2 + y[:, None] ** 2
    return img * (1 - strength * np.clip(r2 / 3.2, 0, 1))[..., None]


def auto_exposure(hdr, target=0.9, percentile=99.7):
    lum = hdr[..., :3] @ np.array([0.2126, 0.7152, 0.0722])
    p = np.percentile(lum, percentile)
    return target / max(p, 1e-6)


def finish(hdr, exposure, do_bloom=True, do_ca=False, do_scanlines=False, do_vignette=False):
    """Linear HDR (H, W, 3+) -> uint8 sRGB image."""
    img = hdr[..., :3] * exposure
    if do_bloom:
        img = bloom(img)
    img = aces(img)
    if do_ca:
        img = chromatic_aberration(img)
    if do_vignette:
        img = vignette(img)
    if do_scanlines:
        img = scanlines(img)
    return (srgb_encode(img) * 255 + 0.5).astype(np.uint8)
