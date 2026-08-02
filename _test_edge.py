import sys
sys.path.insert(0, ".")
import mineudec as m
import taichi as ti
import random

@ti.kernel
def _spilled() -> ti.i32:
    n = 0
    for i in range(m.bcount[None]):
        if m.bfixed[i] == 0 and (m.bhome[i] // m.COLS) >= 39 and m.bx[i] > 30.2:
            n += 1
    return n

@ti.kernel
def _on_ledge() -> ti.i32:
    n = 0
    for i in range(m.bcount[None]):
        if m.bfixed[i] == 0 and (m.bhome[i] // m.COLS) >= 39 and m.bx[i] < 30.2:
            n += 1
    return n

@ti.kernel
def _ledge_spread() -> ti.f32:
    lo = 1e9
    hi = -1e9
    n = 0
    for i in range(m.bcount[None]):
        if m.bfixed[i] == 0 and (m.bhome[i] // m.COLS) >= 39 and m.bx[i] < 30.2:
            n += 1
            ti.atomic_min(lo, m.bx[i])
            ti.atomic_max(hi, m.bx[i])
    return ti.select(n == 0, 0.0, hi - lo)

m.spilled = _spilled
m.on_ledge = _on_ledge
m.ledge_spread = _ledge_spread

m.apply_gravity(1.0)
m.apply_rock(10.0, 4.0, 0.62)
m.P[None].fixedRows = 0
m.P[None].walls = 1
m.P[None].substeps = 48
m.P[None].depth = 60.0
m.generate_solid()

# leave only a thin ledge x<30 at the top (y=42..43); everything else is void
for yy in range(0, m.ROWS):
    for xx in range(0, m.COLS):
        if not (xx < 30 and yy >= 42):
            m.stamp_erase(xx + 0.5, yy + 0.5, 0.5)
print("ledge: blocks=%d" % m.bcount[None])

# drop a pile of loose cubes near the edge (x=25..30) just under the ledge
random.seed(5)
dropped = 0
for f in range(400):
    for k in range(2):
        if dropped >= 50:
            break
        cx = 25 + (dropped % 6)
        cy = 39 + (dropped // 6)
        m.stamp_rock(cx + 0.5, cy + 0.5, 0.35)
        dropped += 1
    m.step_physics(0, m.DT)
    if f % 50 == 0:
        print("  f=%d on_ledge=%d spilled=%d spread=%.1f"
              % (f, m.on_ledge(), m.spilled(), m.ledge_spread()))
print("final: on_ledge=%d spilled=%d spread=%.1f"
      % (m.on_ledge(), m.spilled(), m.ledge_spread()))
