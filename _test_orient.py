import sys
sys.path.insert(0, ".")
import mineudec as m
import taichi as ti

m.apply_gravity(1.0)
m.apply_rock(10.0, 4.0, 0.62)
m.P[None].substeps = 48
m.generate_solid()

# carve a vertical shaft so we can watch a cube fall
for yy in range(0, m.ROWS):
    for xx in range(30, 34):
        m.stamp_erase(xx + 0.5, yy + 0.5, 0.6)

# drop one free cube inside the shaft, away from any wall
m.stamp_rock(31.5, 30.5, 0.35)

@ti.kernel
def _last() -> ti.i32:
    return m.bcount[None] - 1
m.last = _last
@ti.kernel
def _pos(i: ti.i32) -> ti.f32:
    return m.by[i]
m.pos = _pos

i = m.last()
print("start: idx=%d y=%.3f" % (i, m.pos(i)))
for f in range(30):
    m.step_physics(0, m.DT)
    if f % 3 == 0:
        print("  f=%d y=%.3f" % (f, m.pos(i)))
