"""
validated_partial_factorial_designs:

References:
    * Section 5.3.3.4.7 of the Engineering Statistics Handbook by NIST
    * Statistics For Experimentors by BOX, HUNTER & HUNTER
    * Systems Of Experimental Design, VOL. 2 by TAGUCHI
"""

import pandas as pd

# 2**3-1 Res III Design (Taguchi L4 DESIGN)
design_2_3_1 = pd.DataFrame({
    'x0': {0: -1.0, 1: -1.0, 2: 1.0, 3: 1.0},
    'x1': {0: -1.0, 1: 1.0, 2: -1.0, 3: 1.0},
    'x2': {0: 1.0, 1: -1.0, 2: -1.0, 3: 1.0}
})


# 2**4-1 Res IV Design
design_2_4_1 = pd.DataFrame({
    'x0': {0: -1.0, 1: 1.0, 2: -1.0, 3: 1.0, 4: -1.0, 5: 1.0, 6: -1.0, 7: 1.0},
    'x1': {0: -1.0, 1: -1.0, 2: -1.0, 3: -1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0},
    'x2': {0: -1.0, 1: -1.0, 2: 1.0, 3: 1.0, 4: -1.0, 5: -1.0, 6: 1.0, 7: 1.0},
    'x3': {0: -1.0, 1: 1.0, 2: 1.0, 3: -1.0, 4: 1.0, 5: -1.0, 6: -1.0, 7: 1.0}
})


# 2**5-1 Res V Design
design_2_5_1 = pd.DataFrame({
    'x0': {0: -1.0, 1: 1.0, 2: -1.0, 3: 1.0, 4: -1.0, 5: 1.0, 6: -1.0, 7: 1.0, 8: -1.0, 9: 1.0,
           10: -1.0, 11: 1.0, 12: -1.0, 13: 1.0, 14: -1.0, 15: 1.0},
    'x1': {0: -1.0, 1: -1.0, 2: 1.0, 3: 1.0, 4: -1.0, 5: -1.0, 6: 1.0, 7: 1.0, 8: -1.0, 9: -1.0,
           10: 1.0, 11: 1.0, 12: -1.0, 13: -1.0, 14: 1.0, 15: 1.0},
    'x2': {0: -1.0, 1: -1.0, 2: -1.0, 3: -1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: -1.0, 9: -1.0,
           10: -1.0, 11: -1.0, 12: 1.0, 13: 1.0, 14: 1.0, 15: 1.0},
    'x3': {0: -1.0, 1: -1.0, 2: -1.0, 3: -1.0, 4: -1.0, 5: -1.0, 6: -1.0, 7: -1.0, 8: 1.0, 9: 1.0,
           10: 1.0, 11: 1.0, 12: 1.0, 13: 1.0, 14: 1.0, 15: 1.0},
    'x4': {0: 1.0, 1: -1.0, 2: -1.0, 3: 1.0, 4: -1.0, 5: 1.0, 6: 1.0, 7: -1.0, 8: -1.0, 9: 1.0,
           10: 1.0, 11: -1.0, 12: 1.0, 13: -1.0, 14: -1.0, 15: 1.0}
})


# 2**5-2 Res III Design
design_2_5_2 = pd.DataFrame({
    'x0': {0: -1.0, 1: 1.0, 2: -1.0, 3: 1.0, 4: -1.0, 5: 1.0, 6: -1.0, 7: 1.0},
    'x1': {0: -1.0, 1: -1.0, 2: 1.0, 3: 1.0, 4: -1.0, 5: -1.0, 6: 1.0, 7: 1.0},
    'x2': {0: -1.0, 1: -1.0, 2: -1.0, 3: -1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0},
    'x3': {0: 1.0, 1: -1.0, 2: -1.0, 3: 1.0, 4: 1.0, 5: -1.0, 6: -1.0, 7: 1.0},
    'x4': {0: 1.0, 1: -1.0, 2: 1.0, 3: -1.0, 4: -1.0, 5: 1.0, 6: -1.0, 7: 1.0}
})


# 2**6-1 Res VI Design
design_2_6_1 = pd.DataFrame({
    'x0': {0: -1.0, 1: 1.0, 2: -1.0, 3: 1.0, 4: -1.0, 5: 1.0, 6: -1.0, 7: 1.0, 8: -1.0, 9: 1.0,
           10: -1.0, 11: 1.0, 12: -1.0, 13: 1.0, 14: -1.0, 15: 1.0, 16: -1.0, 17: 1.0, 18: -1.0,
           19: 1.0, 20: -1.0, 21: 1.0, 22: -1.0, 23: 1.0, 24: -1.0, 25: 1.0, 26: -1.0, 27: 1.0,
           28: -1.0, 29: 1.0, 30: -1.0, 31: 1.0},
    'x1': {0: -1.0, 1: -1.0, 2: 1.0, 3: 1.0, 4: -1.0, 5: -1.0, 6: 1.0, 7: 1.0, 8: -1.0, 9: -1.0,
           10: 1.0, 11: 1.0, 12: -1.0, 13: -1.0, 14: 1.0, 15: 1.0, 16: -1.0, 17: -1.0, 18: 1.0,
           19: 1.0, 20: -1.0, 21: -1.0, 22: 1.0, 23: 1.0, 24: -1.0, 25: -1.0, 26: 1.0, 27: 1.0,
           28: -1.0, 29: -1.0, 30: 1.0, 31: 1.0},
    'x2': {0: -1.0, 1: -1.0, 2: -1.0, 3: -1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: -1.0, 9: -1.0,
           10: -1.0, 11: -1.0, 12: 1.0, 13: 1.0, 14: 1.0, 15: 1.0, 16: -1.0, 17: -1.0, 18: -1.0,
           19: -1.0, 20: 1.0, 21: 1.0, 22: 1.0, 23: 1.0, 24: -1.0, 25: -1.0, 26: -1.0, 27: -1.0,
           28: 1.0, 29: 1.0, 30: 1.0, 31: 1.0},
    'x3': {0: -1.0, 1: -1.0, 2: -1.0, 3: -1.0, 4: -1.0, 5: -1.0, 6: -1.0, 7: -1.0, 8: 1.0, 9: 1.0,
           10: 1.0, 11: 1.0, 12: 1.0, 13: 1.0, 14: 1.0, 15: 1.0, 16: -1.0, 17: -1.0, 18: -1.0,
           19: -1.0, 20: -1.0, 21: -1.0, 22: -1.0, 23: -1.0, 24: 1.0, 25: 1.0, 26: 1.0, 27: 1.0,
           28: 1.0, 29: 1.0, 30: 1.0, 31: 1.0},
    'x4': {0: -1.0, 1: -1.0, 2: -1.0, 3: -1.0, 4: -1.0, 5: -1.0, 6: -1.0, 7: -1.0, 8: -1.0, 9: -1.0,
           10: -1.0, 11: -1.0, 12: -1.0, 13: -1.0, 14: -1.0, 15: -1.0, 16: 1.0, 17: 1.0, 18: 1.0,
           19: 1.0, 20: 1.0, 21: 1.0, 22: 1.0, 23: 1.0, 24: 1.0, 25: 1.0, 26: 1.0, 27: 1.0, 28: 1.0,
           29: 1.0, 30: 1.0, 31: 1.0},
    'x5': {0: -1.0, 1: 1.0, 2: 1.0, 3: -1.0, 4: 1.0, 5: -1.0, 6: -1.0, 7: 1.0, 8: 1.0, 9: -1.0,
           10: -1.0, 11: 1.0, 12: -1.0, 13: 1.0, 14: 1.0, 15: -1.0, 16: 1.0, 17: -1.0, 18: -1.0,
           19: 1.0, 20: -1.0, 21: 1.0, 22: 1.0, 23: -1.0, 24: -1.0, 25: 1.0, 26: 1.0, 27: -1.0,
           28: 1.0, 29: -1.0, 30: -1.0, 31: 1.0}
})


# 2**6-1 Res IV Design
design_2_6_2 = pd.DataFrame({
})



# 2**6-1 Res III Design
design_2_6_3 = pd.DataFrame({
})