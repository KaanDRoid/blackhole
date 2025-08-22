"""Run the compute shader `geodesic_rk4.comp` using moderngl offscreen context and save output.
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
ctx.bind_image(0, tex, read=False, write=True)

comp['u_r_s'] = 1.0
comp['u_r_obs'] = 100.0
comp['u_b_scale'] = 6.0
comp['u_step_phi'] = 0.01
comp['u_max_steps'] = 20000

gx = (W + 7)//8
gy = (H + 7)//8
comp.run(group_x=gx, group_y=gy, group_z=1)
ctx.memory_barrier()

data = tex.read()
img = Image.frombytes('RGBA', (W,H), data)
img = img.convert('RGB')
img.save('geodesic_compute_out.png')
print('Saved geodesic_compute_out.png')
