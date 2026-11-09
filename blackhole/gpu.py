"""GPU renderer: runs shaders/raytrace.comp in a headless ModernGL context."""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass

import numpy as np

from . import physics

SHADER_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "shaders")
STYLES = {"physical": 0, "cyberpunk": 1, "terminal": 2}


def load_shader(name="raytrace.comp"):
    """Read a shader and expand ``#include "file"`` lines."""
    with open(os.path.join(SHADER_DIR, name), encoding="utf-8") as f:
        src = f.read()

    def include(m):
        return load_shader(m.group(1))

    return re.sub(r'^#include "([^"]+)"\s*$', include, src, flags=re.MULTILINE)


@dataclass
class Scene:
    width: int = 1920
    height: int = 1080
    r_obs: float = 30.0
    inclination: float = 80.0   # degrees from the disk normal
    azimuth: float = 0.0        # degrees
    fov: float = 30.0           # vertical, degrees
    r_in: float = physics.R_ISCO
    r_out: float = 14.0
    disk: bool = True
    style: str = "cyberpunk"
    time: float = 0.0           # loop phase in [0, 1)
    t_peak: float = 7000.0      # physical style, K
    flow: float = 14.0
    tex_amp: float | None = None
    horizon: float = -0.10      # rad, cyberpunk floor horizon
    max_steps: int = physics.MAX_STEPS
    h_max: float = physics.H_MAX
    du_max: float = physics.DU_MAX


class GpuTracer:
    def __init__(self, tile_rows=64, context=None):
        import moderngl

        if tile_rows < 8 or tile_rows % 8:
            raise ValueError("tile_rows must be a positive multiple of 8")
        self.ctx = context if context is not None else moderngl.create_standalone_context(require=430)
        self.prog = self.ctx.compute_shader(load_shader())
        self.tile_rows = tile_rows
        self._tex = {}

    def _set(self, name, value):
        if name in self.prog:
            self.prog[name].value = value

    def _textures(self, w, h):
        key = (w, h)
        if key not in self._tex:
            for old in self._tex.values():
                for t in old:
                    t.release()
            self._tex = {key: (self.ctx.texture((w, h), 4, dtype="f4"),
                               self.ctx.texture((w, h), 4, dtype="f4"))}
        return self._tex[key]

    def render(self, scene: Scene, ss: int = 1):
        """Return (color, aux) as float32 arrays of shape (H*ss, W*ss, 4).

        color.rgb is linear HDR radiance; aux = (status, order, g, r).
        """
        w, h = scene.width * ss, scene.height * ss
        tex_c, tex_a = self._textures(w, h)
        tex_c.bind_to_image(0, read=False, write=True)
        tex_a.bind_to_image(1, read=False, write=True)

        pos, fwd, right, up = physics.camera_basis(scene.r_obs, scene.inclination, scene.azimuth)
        _, fmax = physics.flux_peak(scene.r_out)
        tan_half = math.tan(math.radians(scene.fov) / 2)
        style = STYLES[scene.style]
        tex_amp = scene.tex_amp if scene.tex_amp is not None else (0.0 if style == 0 else (0.45 if style == 2 else 0.85))

        s = self._set
        s("u_size", (w, h))
        s("u_cam_pos", tuple(pos))
        s("u_cam_fwd", tuple(fwd))
        s("u_cam_right", tuple(right))
        s("u_cam_up", tuple(up))
        s("u_tan_half_fov", tan_half)
        s("u_r_obs", scene.r_obs)
        s("u_r_in", scene.r_in)
        s("u_r_out", scene.r_out)
        s("u_disk_on", 1 if scene.disk else 0)
        s("u_max_steps", scene.max_steps)
        s("u_h_max", scene.h_max)
        s("u_du_max", scene.du_max)
        s("u_style", style)
        s("u_time", scene.time)
        s("u_pix_angle", 2 * tan_half / scene.height)
        s("u_flux_max", fmax)
        s("u_t_peak", scene.t_peak)
        s("u_flow", scene.flow)
        s("u_tex_amp", tex_amp)
        s("u_horizon", scene.horizon)

        gx = (w + 7) // 8
        for y0 in range(0, h, self.tile_rows):
            rows = min(self.tile_rows, h - y0)
            s("u_row_offset", y0)
            self.prog.run(group_x=gx, group_y=(rows + 7) // 8, group_z=1)
            self.ctx.finish()

        color = np.frombuffer(tex_c.read(), dtype=np.float32).reshape(h, w, 4)
        aux = np.frombuffer(tex_a.read(), dtype=np.float32).reshape(h, w, 4)
        return color, aux
