import sys
sys.path.insert(0, ".")
import mineudec as m
import taichi as ti

m.apply_gravity(1.0)
m.apply_rock(10.0, 4.0, 0.62)
m.P[None].fixedRows = 2
m.P[None].depth = 60.0
m.generate_solid()
print("after gen:", m.bcount[None])

@ti.kernel
def _fixed_count() -> ti.i32:
    n = 0
    for i in range(m.bcount[None]):
        if m.bfixed[i] == 1:
            n += 1
    return n
m.fixed_count = _fixed_count

@ti.kernel
def _row_count(ro: ti.i32) -> ti.i32:
    n = 0
    for i in range(m.bcount[None]):
        if m.bhome[i] // m.COLS == ro:
            n += 1
    return n
m.row_count = _row_count

for ro in [38, 39, 40, 41, 42, 43]:
    print("row %d: %d" % (ro, m.row_count(ro)))

for yy in range(0, 40):
    for xx in range(0, m.COLS):
        m.stamp_erase(xx + 0.5, yy + 0.5, 0.5)
print("after erase:", m.bcount[None], "fixed:", m.fixed_count())
for ro in [38, 39, 40, 41, 42, 43]:
    print("row %d: %d" % (ro, m.row_count(ro)))
