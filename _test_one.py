import sys
sys.path.insert(0, ".")
import mineudec as m
import taichi as ti

m.apply_gravity(1.0)
m.apply_rock(10.0, 4.0, 0.62)
m.P[None].fixedRows = 2
m.P[None].walls = 1
m.P[None].substeps = 48
m.P[None].depth = 60.0
m.generate_solid()

for yy in range(31, m.ROWS):
    for xx in range(0, m.COLS):
        m.stamp_erase(xx + 0.5, yy + 0.5, 0.5)

# single cube, drop height ~2 cells above the floor
m.stamp_rock(35.5, 32.5, 0.35)

@ti.kernel
def _last() -> ti.i32:
    return m.bcount[None] - 1
m.last = _last
@ti.kernel
def _state(i: ti.i32) -> ti.f32:
    return m.by[i]
m.state = _state
@ti.kernel
def _bw(i: ti.i32) -> ti.f32:
    return m.bw[i]
m.bwv = _bw
@ti.kernel
def _bx(i: ti.i32) -> ti.f32:
    return m.bx[i]
m.bxv = _bx
@ti.kernel
def _nb(i: ti.i32) -> ti.i32:
    return m.nbond[i]
m.nbv = _nb
@ti.kernel
def _tor(i: ti.i32) -> ti.f32:
    return m.btor[i]
m.torv = _tor

i = m.last()
print("start: y=%.3f" % m.state(i))
for f in range(120):
    m.step_physics(0, m.DT)
    if f % 5 == 0:
        print("  f=%d y=%.3f x=%.3f bw=%.3f btor=%.3f nbond=%d" % (f, m.state(i), m.bxv(i), m.bwv(i), m.torv(i), m.nbv(i)))
