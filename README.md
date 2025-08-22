
Binary lensing demo and Schwarzschild geodesic RK4 reference

Contents:
- `main.py` : GLFW + moderngl demo that runs a binary (multi-)lens compute shader and displays the result. Use keys to move lenses and change Einstein radii.
- `shaders/lensing.comp` : compute shader implementing N-point-mass (binary) thin-lens deflection.
- `shaders/quad.vert`, `shaders/quad.frag` : fullscreen quad shaders to present the compute result.
- `shaders/geodesic_schwarzschild.comp` : commented skeleton for a compute-shader RK4 integrator (guide + starting implementation).
- `geodesic_rk4.py` : complete CPU RK4 integrator for equatorial null geodesics that renders a reference PNG.

Requirements:
python 3.10+, pip install glfw moderngl numpy Pillow

Run the GPU demo:
python main.py

Run CPU geodesic reference (generates `geodesic_out.png`):
python geodesic_rk4.py
=======
# blackhole
GPU accelerated gravitational lensing demo (Python + ModernGL) with adaptive RK4 Schwarzschild geodesics

