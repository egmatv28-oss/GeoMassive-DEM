import sys
sys.path.insert(0, ".")
import mineudec as m
import taichi as ti
import random

@ti.kernel
def _pile_spread() -> ti.f32:
    lo = 1e9
    hi = -1e9
    n = 0
    for i in range(m.bcount[None]):
        if m.bfixed[i] == 0 and m.bhome[i] // m.COLS < 39:
            n += 1
            ti.atomic_min(lo, m.bx[i])
            ti.atomic_max(hi, m.bx[i])
    return ti.select(n == 0, 0.0, hi - lo)

@ti.kernel
def _pile_count() -> ti.i32:
    n = 0
    for i in range(m.bcount[None]):
        if m.bfixed[i] == 0 and m.bhome[i] // m.COLS < 39:
            n += 1
    return n

@ti.kernel
def _pile_maxy() -> ti.f32:
    mx = 0.0
    for i in range(m.bcount[None]):
        if m.bfixed[i] == 0 and m.bhome[i] // m.COLS < 39:
            ti.atomic_max(mx, m.by[i])
    return mx

m.pile_spread = _pile_spread
m.pile_count = _pile_count
m.pile_maxy = _pile_maxy

m.apply_gravity(1.0)
m.apply_rock(10.0, 4.0, 0.62)
m.P[None].fixedRows = 2
m.P[None].walls = 1
m.P[None].substeps = 48
m.P[None].depth = 60.0
m.generate_solid()

# leave the mass y=40..43 as the "floor"; carve the rest below
for yy in range(0, 40):
    for xx in range(0, m.COLS):
        m.stamp_erase(xx + 0.5, yy + 0.5, 0.5)
print("floor: blocks=%d" % m.bcount[None])

# drop 40 single loose cubes just below the floor surface, tight 10x4 block
random.seed(11)
dropped = 0
for f in range(300):
    for k in range(3):
        if dropped >= 40:
            break
        cx = 30 + (dropped % 10)
        cy = 35 + (dropped // 10)
        m.stamp_rock(cx + 0.5, cy + 0.5, 0.35)
        dropped += 1
    m.step_physics(0, m.DT)
    if f % 50 == 0:
        print("  f=%d count=%d spread=%.1f maxy=%.1f"
              % (f, m.pile_count(), m.pile_spread(), m.pile_maxy()))
print("final: count=%d spread=%.1f maxy=%.1f"
      % (m.pile_count(), m.pile_spread(), m.pile_maxy()))
