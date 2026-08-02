import sys
sys.path.insert(0, ".")
import mineudec as m
import taichi as ti
import random

@ti.kernel
def _spread_lo() -> ti.f32:
    lo = 1e9
    hi = -1e9
    for i in range(m.bcount[None]):
        if m.bfixed[i] == 0 and m.bx[i] > 20.0 and m.bx[i] < 50.0:
            if m.bx[i] < lo:
                lo = m.bx[i]
            if m.bx[i] > hi:
                hi = m.bx[i]
    return hi - lo

@ti.kernel
def _count_lo() -> ti.i32:
    n = 0
    for i in range(m.bcount[None]):
        if m.bfixed[i] == 0 and m.bx[i] > 20.0 and m.bx[i] < 50.0:
            n += 1
    return n

@ti.kernel
def _miny_lo() -> ti.f32:
    my = 1e9
    for i in range(m.bcount[None]):
        if m.bfixed[i] == 0 and m.bx[i] > 20.0 and m.bx[i] < 50.0:
            if m.by[i] < my:
                my = m.by[i]
    return my

m.spread_lo = _spread_lo
m.count_lo = _count_lo
m.miny_lo = _miny_lo

m.apply_gravity(1.0)
m.apply_rock(10.0, 4.0, 0.62)
m.apply_water(12.0)
m.P[None].fixedRows = 2
m.P[None].walls = 1
m.P[None].substeps = 48
m.P[None].depth = 60.0
m.generate(0, 0)
print("world: blocks=%d intact=%d" % (m.bcount[None], m.count_intact()))

# drop a dense heap of SINGLE loose cubes near the surface (cells 34..38)
random.seed(7)
for k in range(60):
    cx = random.choice([34, 35, 36, 37, 38])
    cy = random.choice([37, 38, 39, 40])
    m.stamp_rock(cx + 0.5, cy + 0.5, 0.35)
print("dropped heap: count=%d" % m.count_lo())

import os
for f in range(300):
    m.step_physics(0, m.DT)
    if f % 50 == 0:
        print("  f=%d count=%d spread=%.1f miny=%.1f"
              % (f, m.count_lo(), m.spread_lo(), m.miny_lo()))
print("final: count=%d spread=%.1f miny=%.1f" % (m.count_lo(), m.spread_lo(), m.miny_lo()))
