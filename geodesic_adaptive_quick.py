"""Run the quick renderer, or the medium one if the quick script is missing.

Usage: python geodesic_adaptive_quick.py
"""
import os
import sys
from subprocess import run

here = os.path.dirname(os.path.abspath(__file__))
quick = os.path.join(here, 'geodesic_rk4_quick.py')
medium = os.path.join(here, 'geodesic_rk4_medium.py')

if os.path.exists(quick):
    print('Running quick renderer...')
    run([sys.executable, quick], check=True)
else:
    print('Quick renderer not found, running medium renderer...')
    run([sys.executable, medium], check=True)
