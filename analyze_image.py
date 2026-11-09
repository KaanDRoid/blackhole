"""Print basic statistics for a rendered image.

Usage:
    python analyze_image.py [path]

If no path is given, uses geodesic_adaptive_quick.png in the current
directory (where the renderers save their output).
Prints size, sample pixels, channel min/max, 16-bin histograms and the
fraction of pure black (captured) pixels.
"""
import sys

import numpy as np
from PIL import Image

path = sys.argv[1] if len(sys.argv) > 1 else 'geodesic_adaptive_quick.png'
im = Image.open(path).convert('RGB')
w, h = im.size
print('file:', path)
print('size', w, h)
a = np.array(im)
cx, cy = w//2, h//2
print('center', tuple(int(v) for v in a[cy, cx]))
print('left', tuple(int(v) for v in a[cy, w//4]))
print('right', tuple(int(v) for v in a[cy, 3*w//4]))

r = a[:, :, 0].ravel()
g = a[:, :, 1].ravel()
b = a[:, :, 2].ravel()
print('min/max R', int(r.min()), int(r.max()))
print('min/max G', int(g.min()), int(g.max()))
print('min/max B', int(b.min()), int(b.max()))


def hist16(arr):
    hist, _ = np.histogram(arr, bins=16, range=(0, 256))
    return hist.tolist()


print('R 16-bin:', hist16(r))
print('G 16-bin:', hist16(g))
print('B 16-bin:', hist16(b))

black = np.sum((r == 0) & (g == 0) & (b == 0))
print(f'black pixels: {int(black)} ({black / (w*h) * 100:.3f}%)')
