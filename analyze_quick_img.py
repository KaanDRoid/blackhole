from PIL import Image
import numpy as np
im=Image.open(r'D:\Big Data\Kişisel öğrenim\blackhole_py_demo\geodesic_adaptive_quick.png')
w,h=im.size
print('size',w,h)
a=np.array(im.convert('RGB'))
cx,cy=w//2,h//2
print('center', tuple(a[cy,cx]))
print('left', tuple(a[cy,w//4]))
print('right', tuple(a[cy,3*w//4]))
r=a[:,:,0].ravel(); g=a[:,:,1].ravel(); b=a[:,:,2].ravel()

import numpy as _np

def small_hist(arr):
    h=_np.bincount(arr)
    return [int(h[i]) if i<len(h) else 0 for i in range(10)]

print('r[0..9]', small_hist(r))
print('g[0..9]', small_hist(g))
print('b[0..9]', small_hist(b))
