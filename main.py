"""Binary-lens compute shader demo (GLFW + moderngl)
Keys:
- N: single-lens mode
- M: multi-lens mode
- Arrow keys: move active lens
- C: cycle active lens (multi mode)
- Q/E: decrease/increase Einstein radius (Re) of active lens
- + / - : add / remove lens (multi mode, up to 8)
- T/G: background scale up/down
- S: save screenshot
- ESC: exit
"""
import sys, os
import glfw
import moderngl
import numpy as np
from PIL import Image

DEMO_DIR = os.path.dirname(__file__)
SHADER_DIR = os.path.join(DEMO_DIR, 'shaders')

WIDTH, HEIGHT = 800, 600
MAX_LENSES = 8  # must match lensing.comp

def load_shader(ctx, path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def make_background(size=(1024,512)):
    # procedural starfield
    W,H = size
    img = Image.new('RGB', (W,H), (0,0,10))
    px = img.load()
    import random
    for _ in range(8000):
        x = random.randrange(W)
        y = random.randrange(H)
        b = random.randint(150,255)
        px[x,y] = (b,b,b)
    return img

class Demo:
    def __init__(self):
        if not glfw.init():
            raise RuntimeError('glfw init failed')
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        self.win = glfw.create_window(WIDTH, HEIGHT, 'Binary Lens Demo', None, None)
        if not self.win:
            glfw.terminate(); raise RuntimeError('window')
        glfw.make_context_current(self.win)
        self.ctx = moderngl.create_context()

        # create compute shader
        comp_src = load_shader(self.ctx, os.path.join(SHADER_DIR, 'lensing.comp'))
        self.comp = self.ctx.compute_shader(comp_src)

        # fullscreen quad program
        vert = load_shader(self.ctx, os.path.join(SHADER_DIR, 'quad.vert'))
        frag = load_shader(self.ctx, os.path.join(SHADER_DIR, 'quad.frag'))
        self.prog = self.ctx.program(vertex_shader=vert, fragment_shader=frag)

        # quad vbo
        quad = np.array([ -1.0,-1.0,  1.0,-1.0, -1.0,1.0,  1.0,1.0 ], dtype='f4')
        self.vbo = self.ctx.buffer(quad.tobytes())
        self.vao = self.ctx.simple_vertex_array(self.prog, self.vbo, 'in_pos')

        # create image texture
        self.tex = self.ctx.texture((WIDTH, HEIGHT), 4, dtype='f4')
        self.tex.filter = (moderngl.LINEAR, moderngl.LINEAR)

        # background texture
        bg_img = make_background((1024,512))
        self.bg = self.ctx.texture(bg_img.size, 3, bg_img.tobytes())
        self.bg.build_mipmaps()
        self.bg.use(location=0)

        # lens params: start with two lenses
        self.lens_pos = [np.array([0.45,0.5], dtype='f4'), np.array([0.55,0.5], dtype='f4')]
        self.lens_re  = [0.06, 0.06]
        self.bg_scale = 1.0
        self.mode = 'multi'  # 'single' or 'multi'
        self.active_idx = 0

        glfw.set_key_callback(self.win, self.on_key)

        print('Controls: N single, M multi, C cycle active, +/- add/remove, arrows move active, Q/E Re -,/+, T/G bg scale, S screenshot')

    def on_key(self, win, key, sc, action, mods):
        if action == glfw.PRESS or action == glfw.REPEAT:
            step = 0.01
            # global
            if key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(self.win, True)
                return
            if key == glfw.KEY_N:
                self.mode = 'single'
                if len(self.lens_pos) == 0:
                    self.lens_pos.append(np.array([0.5,0.5], dtype='f4'))
                    self.lens_re.append(0.05)
                self.active_idx = 0
                print('Mode -> SINGLE')
                return
            if key == glfw.KEY_M:
                self.mode = 'multi'
                self.active_idx = 0
                print('Mode -> MULTI')
                return
            # cycle active lens
            if key == glfw.KEY_C:
                if len(self.lens_pos) > 0:
                    self.active_idx = (self.active_idx + 1) % len(self.lens_pos)
                    print(f'Active lens -> {self.active_idx}')
                return
            # add / remove lens (multi mode)
            if key == glfw.KEY_KP_ADD or key == glfw.KEY_EQUAL:
                if len(self.lens_pos) < MAX_LENSES:
                    self.lens_pos.append(np.array([0.5,0.5], dtype='f4'))
                    self.lens_re.append(0.02)
                    self.active_idx = len(self.lens_pos)-1
                    print('Lens added, count=', len(self.lens_pos))
                return
            if key == glfw.KEY_KP_SUBTRACT or key == glfw.KEY_MINUS:
                if len(self.lens_pos) > 1:
                    removed = len(self.lens_pos)-1
                    self.lens_pos.pop()
                    self.lens_re.pop()
                    self.active_idx = max(0, len(self.lens_pos)-1)
                    print('Lens removed, count=', len(self.lens_pos))
                return
            # screenshot
            if key == glfw.KEY_S:
                # texture is RGBA float32 with row 0 at the bottom of the window
                data = np.frombuffer(self.tex.read(), dtype='f4').reshape(HEIGHT, WIDTH, 4)
                rgb = (np.clip(data[:, :, :3], 0.0, 1.0) * 255).astype(np.uint8)
                Image.fromarray(np.flipud(rgb)).save('screenshot.png')
                print('Saved screenshot.png')
                return

            # movement keys apply to active lens (multi) or lens 0 (single)
            idx = 0 if self.mode == 'single' else self.active_idx
            if key == glfw.KEY_LEFT:
                self.lens_pos[idx][0] -= step
            if key == glfw.KEY_RIGHT:
                self.lens_pos[idx][0] += step
            # uv.y = 0 is the bottom of the window
            if key == glfw.KEY_UP:
                self.lens_pos[idx][1] += step
            if key == glfw.KEY_DOWN:
                self.lens_pos[idx][1] -= step

            # Einstein radius adjust for active lens
            if key == glfw.KEY_Q:
                self.lens_re[idx] = max(0.0, self.lens_re[idx]-0.005)
            if key == glfw.KEY_E:
                self.lens_re[idx] = self.lens_re[idx]+0.005

            # background scale
            if key == glfw.KEY_T:
                self.bg_scale *= 1.1
            if key == glfw.KEY_G:
                self.bg_scale /= 1.1

    def update_uniforms(self):
        # single mode only uses the first lens
        count = 1 if self.mode == 'single' else len(self.lens_pos)
        pos = [(float(p[0]), float(p[1])) for p in self.lens_pos]
        re_ = [float(r) for r in self.lens_re]
        # pad to MAX_LENSES as declared in lensing.comp; moderngl exposes
        # uniform arrays by base name, so the whole array is written at once
        pos += [(0.0, 0.0)] * (MAX_LENSES - len(pos))
        re_ += [0.0] * (MAX_LENSES - len(re_))
        self.comp['u_lens_count'].value = count
        self.comp['u_lens_pos_flat'].value = pos
        self.comp['u_lens_re_flat'].value = re_
        self.comp['u_bg_scale'].value = float(self.bg_scale)

    def run(self):
        while not glfw.window_should_close(self.win):
            glfw.poll_events()

            # update uniforms
            self.update_uniforms()

            # bind image and background
            # write to image unit 0
            self.tex.bind_to_image(0, read=False, write=True)
            # background in location 0
            self.bg.use(location=0)

            # dispatch compute
            gx = (WIDTH + 7)//8
            gy = (HEIGHT + 7)//8
            self.comp.run(group_x=gx, group_y=gy, group_z=1)

            # memory barrier
            self.ctx.memory_barrier()

            # draw quad
            self.tex.use(location=0)
            self.prog['tex'] = 0
            self.ctx.clear(0.0,0.0,0.0)
            self.vao.render(moderngl.TRIANGLE_STRIP)

            glfw.swap_buffers(self.win)

        glfw.terminate()

if __name__ == '__main__':
    d = Demo()
    d.run()
