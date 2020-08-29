"""
Design
======

This module is dedicated to design of experiments (DOE).
"""

# factorial / factorial-like designs
from lind.design import factorial
from lind.design import plackett_burman
# response surface designs
from lind.design import box_wilson
from lind.design import box_behnken
# randomized designs
from lind.design import latin_hyper_cube
