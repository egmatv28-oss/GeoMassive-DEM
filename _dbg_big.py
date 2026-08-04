import sys
import numpy as np


def log(*a):
    print(*a, flush=True)


log("import")
import render as m
from render import *

apply_gravity(1.0)
apply_rock(10.0, 4.0, 0.62)
apply_water(12.0)
P[None].fixedRows = 2
P[None].walls = 1
P[None].substeps = 48
P[None].depth = 60.0

generate(2, 0)
log("bcount", bcount[None], "bonds", bondCount[None], "maxSize", P[None].maxSize)

for f in range(25):
    step_physics(2, DT)
    if f % 5 == 4 or f < 3:
        st = diag_strain()
        log("f=%d broken=%d intact=%d strain=%.4f" % (f, broken[None], count_intact(), st))
log("done")
