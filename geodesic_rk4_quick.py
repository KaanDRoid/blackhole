"""Quick low-res adaptive RK4 renderer for fast feedback.
Produces geodesic_adaptive_quick.png
"""
import math
from PIL import Image

# Quick parameters for speed
W, H = 240, 120
r_s = 1.0
r_obs = 50.0
b_scale = 6.0
E = 1.0

h0 = 0.05
min_h = 1e-3
max_h = 0.2
alpha = 5.0
max_steps = 20000

print('Quick render', W, 'x', H)

def sample_background(phi):
    t = (phi / (2*math.pi)) % 1.0
    r = 0.5 + 0.5 * math.cos(2*math.pi*t)
    g = 0.5 + 0.5 * math.cos(2*math.pi*(t + 0.33))
    b = 0.5 + 0.5 * math.cos(2*math.pi*(t + 0.66))
    return (int(255*r), int(255*g), int(255*b))

def dr_dphi(r, L):
    inside = E*E - (1 - r_s / r) * (L*L) / (r*r)
    if inside <= 0:
        return 0.0
    return (r*r / L) * math.sqrt(inside)

def rk4_step(r, h, L):
    k1 = dr_dphi(r, L)
    k2 = dr_dphi(r + 0.5*h*k1, L)
    k3 = dr_dphi(r + 0.5*h*k2, L)
    k4 = dr_dphi(r + h*k3, L)
    return r + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)

img = Image.new('RGB', (W, H))
px = img.load()

for j in range(H):
    for i in range(W):
        x = (i + 0.5) / W * 2.0 - 1.0
        b = x * b_scale
        L = abs(b)
        if L < 1e-6:
            L = 1e-6
        r = r_obs
        phi = 0.0
        captured = False
        steps = 0
        while steps < max_steps:
            if r <= r_s:
                captured = True
                break
            deriv = dr_dphi(r, L)
            if deriv == 0.0:
                break
            h = h0 / (1.0 + alpha * abs(deriv))
            h = max(min_h, min(max_h, h))
            # adaptive RK4 with step halving error control
            r1 = rk4_step(r, -h, L)
            r_half = rk4_step(r, -h*0.5, L)
            r2 = rk4_step(r_half, -h*0.5, L)
            err = abs(r2 - r1)
            tol = 1e-2
            if err <= tol:
                r = r2
                phi += h
                h = min(h * 1.3, max_h)
            else:
                h = max(h * 0.5, min_h)
            if r > r_obs * 0.995 and phi > 0.05:
                break
            steps += 1
        if captured:
            color = (0,0,0)
        else:
            color = sample_background(phi)
        px[i,j] = color

img.save('geodesic_adaptive_quick.png')
print('Saved geodesic_adaptive_quick.png')
