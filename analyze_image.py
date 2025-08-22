"""Image analysis helper for the demo.
Usage:
	python analyze_image.py [path]
If no path is given, defaults to `geodesic_adaptive_quick.png` in the demo folder.
Prints size, sample pixels, channel min/max, 16-bin histograms and black pixel fraction.
"""
import sys
from PIL import Image
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else r'D:\Big Data\Kişisel öğrenim\blackhole_py_demo\geodesic_adaptive_quick.png'
im = Image.open(path).convert('RGB')
w,h = im.size
print('file:', path)
print('size', w, h)
a = np.array(im)
cx,cy = w//2, h//2
print('center', tuple(a[cy,cx]))
print('left', tuple(a[cy,w//4]))
print('right', tuple(a[cy,3*w//4]))

r = a[:,:,0].ravel(); g = a[:,:,1].ravel(); b = a[:,:,2].ravel()
print('min/max R', int(r.min()), int(r.max()))
print('min/max G', int(g.min()), int(g.max()))
print('min/max B', int(b.min()), int(b.max()))

# 16-bin hist
bins = 16
def hist16(arr):
		hist, _ = np.histogram(arr, bins=bins, range=(0,256))
		return hist.tolist()

print('R 16-bin:', hist16(r))
print('G 16-bin:', hist16(g))
print('B 16-bin:', hist16(b))

# black pixel count
black = np.sum((r==0)&(g==0)&(b==0))
print('black pixels:', int(black), '(', round(black/(w*h)*100,3), '%)')

