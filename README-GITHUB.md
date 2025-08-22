Blackhole Py Demo

This repo contains a small GPU/CPU demo for gravitational lensing and Schwarzschild null geodesics.

Run (Windows PowerShell):

```powershell
python -m venv .venv; .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py    # GPU demo (requires OpenGL 4.3+ and drivers)
python geodesic_rk4_adaptive.py  # CPU reference
```

Files:
- `main.py` - GLFW + ModernGL demo (GPU compute shader lensing)
- `shaders/` - compute and draw shaders
- `geodesic_rk4_adaptive.py` - CPU adaptive RK4 reference
- `geodesic_rk4_quick.py` - quick low-res renderer

