"""Run shaders/geodesic_rk4.comp in a headless moderngl context and save
geodesic_compute_out.png (GPU version of geodesic_rk4_adaptive.py).
"""
import moderngl
from PIL import Image
import numpy as np
import os

DIR = os.path.dirname(__file__)
SHADER = os.path.join(DIR, 'shaders', 'geodesic_rk4.comp')

W,H = 800,400

ctx = moderngl.create_standalone_context()
with open(SHADER, 'r', encoding='utf-8') as f:
    src = f.read()
comp = ctx.compute_shader(src)

tex = ctx.texture((W,H), 4, dtype='f4')
tex.bind_to_image(0, read=False, write=True)

comp['u_r_s'] = 1.0
comp['u_r_obs'] = 100.0
comp['u_b_scale'] = 6.0
comp['u_step_phi'] = 0.01
# adaptive step control; these must be set, otherwise they default to 0
# and the step size collapses to zero
comp['u_tol'] = 1e-3
comp['u_min_step'] = 1e-5
comp['u_max_step'] = 0.1
comp['u_max_steps'] = 20000

gx = (W + 7)//8
gy = (H + 7)//8
comp.run(group_x=gx, group_y=gy, group_z=1)
ctx.memory_barrier()

# the texture is RGBA float32, convert to 8-bit RGB before saving
data = np.frombuffer(tex.read(), dtype='f4').reshape(H, W, 4)
rgb = (np.clip(data[:, :, :3], 0.0, 1.0) * 255).astype(np.uint8)
img = Image.fromarray(rgb)
img.save('geodesic_compute_out.png')
print('Saved geodesic_compute_out.png')
