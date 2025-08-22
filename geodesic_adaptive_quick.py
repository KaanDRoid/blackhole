"""Thin wrapper to run the quick adaptive renderer or fall back to the medium one if missing.
Usage: python geodesic_adaptive_quick.py
"""
import os
from subprocess import run

here = os.path.dirname(__file__)
quick = os.path.join(here, 'geodesic_rk4_quick.py')
medium = os.path.join(here, 'geodesic_rk4_medium.py')

if os.path.exists(quick):
	print('Running quick renderer...')
	run(['python', quick], check=True)
else:
	print('Quick renderer not found, running medium renderer...')
	run(['python', medium], check=True)

