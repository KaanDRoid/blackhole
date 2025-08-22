import argparse

# Parameters (can be overridden via CLI)
parser = argparse.ArgumentParser(description='CPU adaptive RK4 equatorial null-geodesic renderer')
parser.add_argument('--width', type=int, default=800, help='output width')
parser.add_argument('--height', type=int, default=400, help='output height')
parser.add_argument('--quick', action='store_true', help='run quick low-res/low-step test')
parser.add_argument('--max-steps', type=int, default=200000, help='maximum integration steps per ray')
args = parser.parse_args()

W, H = (200, 100) if args.quick else (args.width, args.height)
r_s = 1.0  # Schwarzschild radius
r_obs = 100.0
b_scale = 6.0
E = 1.0

# adaptive parameters
h0 = 0.02
import argparse
import math
from PIL import Image

# Adaptive RK4 equatorial null-geodesic renderer
# Usage: python geodesic_rk4_adaptive.py [--quick] [--width W] [--height H]

parser = argparse.ArgumentParser(description='CPU adaptive RK4 equatorial null-geodesic renderer')
parser.add_argument('--width', type=int, default=800, help='output width')
parser.add_argument('--height', type=int, default=400, help='output height')
parser.add_argument('--quick', action='store_true', help='run quick low-res/fast test')
parser.add_argument('--max-steps', type=int, default=200000, help='maximum integration steps per ray')
args = parser.parse_args()

# geometry / physical params
r_s = 1.0
r_obs = 100.0
b_scale = 6.0
E = 1.0

# adaptive integration defaults
h_init = 0.02
h_min = 1e-5
h_max = 0.1
adapt_alpha = 5.0
max_steps = args.max_steps if not args.quick else 20000

W, H = (200, 100) if args.quick else (args.width, args.height)

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

def integrate_pixel_adaptive(b, r_obs, tol=1e-3):
    L = abs(b) if abs(b) > 1e-8 else 1e-8
    r = r_obs
    phi = 0.0
    h = h_init
    steps = 0
    captured = False
    while steps < max_steps:
        if r <= r_s:
            captured = True
            break
        # fixed strategy: embedded estimate via step halving
        r_full = rk4_step(r, -h, L)
        r_half = rk4_step(r, -h*0.5, L)
        r_half2 = rk4_step(r_half, -h*0.5, L)
        err = abs(r_half2 - r_full)
        if err <= tol:
            r = r_half2
            phi += h
            # increase step slowly
            h = min(h * 1.5, h_max)
        else:
            h = max(h * 0.5, h_min)
            # if h too small, accept to avoid lock
            if h <= h_min + 1e-14:
                r = r_half2
                phi += h
        if math.isnan(r) or r <= 0:
            captured = True
            break
        if r > r_obs * 0.995 and phi > 0.05:
            break
        steps += 1
    if captured:
        return None
    return phi

def run():
    img = Image.new('RGB', (W, H))
    px = img.load()
    print(f'Adaptive renderer {W}x{H} quick={args.quick} max_steps={max_steps}')
    for j in range(H):
        if j % max(1, H//10) == 0:
            print(f'  row {j}/{H}')
        for i in range(W):
            x = (i + 0.5) / W * 2.0 - 1.0
            b = x * b_scale
            phi = integrate_pixel_adaptive(b, r_obs)
            if phi is None:
                color = (0,0,0)
            else:
                color = sample_background(phi)
            px[i,j] = color
    out_name = 'geodesic_adaptive_quick.png' if args.quick else 'geodesic_adaptive_out.png'
    img.save(out_name)
    print(f'Saved {out_name}')

if __name__ == '__main__':
    run()

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

# adaptive per-pixel integration
def integrate_pixel(b, r_obs, tol=1e-3, h_init=0.01, h_min=1e-5, h_max=0.1, max_steps=20000):
    L = abs(b) if abs(b) > 1e-8 else 1e-8
    r = r_obs
    phi = 0.0
    h = h_init
    steps = 0
    captured = False
    escaped = False
    while steps < max_steps:
        if r <= r_s:
            captured = True
            break
        r1 = rk4_step(r, -h, L)
        r_half = rk4_step(r, -h*0.5, L)
        r2 = rk4_step(r_half, -h*0.5, L)
        err = abs(r2 - r1)
        if err <= tol:
            r = r2
            phi += h
            h = min(h*1.5, h_max)
        else:
            h = max(h*0.5, h_min)
            if h <= h_min + 1e-12:
                r = r2
                phi += h
        if math.isnan(r) or r <= 0:
            captured = True
            break
        if r > r_obs * 0.995 and phi > 0.1:
            escaped = True
            break
        steps += 1
    if captured:
        return None
    return phi

img = Image.new('RGB', (W,H))
px = img.load()

for j in range(H):
    for i in range(W):
        x = (i + 0.5) / W * 2.0 - 1.0
        b = x * b_scale
        phi = integrate_pixel(b, r_obs)
        if phi is None:
            color = (0,0,0)
        else:
            color = sample_background(phi)
        px[i,j] = color

img.save('geodesic_adaptive_out.png')
print('Saved geodesic_adaptive_out.png')
