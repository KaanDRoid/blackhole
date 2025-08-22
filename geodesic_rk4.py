"""Simple CPU RK4 equatorial null-geodesic renderer.
Generates `geodesic_out.png` in the demo folder. This is an educational reference, not a highly-optimized renderer.
"""
import math
import numpy as np
from PIL import Image

# Parameters
W, H = 800, 400
r_s = 1.0  # Schwarzschild radius (units)
r_obs = 100.0  # observer radius (far)
b_scale = 6.0  # maps screen half-width to impact parameter
E = 1.0

# Background function: simple azimuthal stripe map
def sample_background(phi):
    # return color as tuple
    t = (phi / (2*math.pi)) % 1.0
    # create simple color wheel
    r = 0.5 + 0.5 * math.cos(2*math.pi*t)
    g = 0.5 + 0.5 * math.cos(2*math.pi*(t + 0.33))
    b = 0.5 + 0.5 * math.cos(2*math.pi*(t + 0.66))
    return (int(255*r), int(255*g), int(255*b))

# ODE dr/dphi = r^2 / L * sqrt(E^2 - (1 - r_s/r) * L^2 / r^2)
def dr_dphi(r, L):
    inside = E*E - (1 - r_s / r) * (L*L) / (r*r)
    if inside <= 0:
        return 0.0
    return (r*r / L) * math.sqrt(inside)

# RK4 step for r(phi)
def rk4_step(r, phi, h, L):
    k1 = dr_dphi(r, L)
    k2 = dr_dphi(r + 0.5*h*k1, L)
    k3 = dr_dphi(r + 0.5*h*k2, L)
    k4 = dr_dphi(r + h*k3, L)
    return r + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)

# For each pixel compute b and integrate r(phi) until escape or capture
img = Image.new('RGB', (W, H))
px = img.load()

for j in range(H):
    for i in range(W):
        # normalized x in [-1,1]
        x = (i + 0.5) / W * 2.0 - 1.0
        # map to impact parameter b
        b = x * b_scale
        L = abs(b)
        if L < 1e-6:
            L = 1e-6
        # integrate phi from 0 outward until r > r_obs*1.01 (escaped) or r <= r_s (captured)
        r = r_obs
        phi = 0.0
        h = 0.01  # phi-step
        captured = False
        escaped = False
        max_steps = 20000
        steps = 0
        # We want to follow ray "inward" so r will decrease; use negative step when dr/dphi>0
        while steps < max_steps:
            # check capture
            if r <= r_s:
                captured = True
                break
            # approximate derivative magnitude
            deriv = dr_dphi(r, L)
            if deriv == 0.0:
                # likely turning point or invalid; break as escaped
                escaped = True
                break
            # Choose sign: for inward travel dr/dphi should be negative; we step phi positive but reduce r
            # We integrate with negative h to march toward smaller r
            r_new = rk4_step(r, phi, -h, L)
            phi += h
            if math.isnan(r_new) or r_new <= 0:
                captured = True
                break
            r = r_new
            # escaped if r increases back above r_obs*0.99
            if r > r_obs * 0.995 and phi > 0.1:
                escaped = True
                break
            steps += 1

        if captured:
            color = (0,0,0)
        else:
            # total deflection angle roughly equals phi reached (approx)
            # map background by phi
            color = sample_background(phi)
        px[i,j] = color

img.save('geodesic_out.png')
print('Saved geodesic_out.png')
