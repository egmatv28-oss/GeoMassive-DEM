import sys
sys.path.insert(0, ".")
import mineudec as m
import taichi as ti

@ti.kernel
def _cf() -> ti.i32:
    n = 0
    for i in range(m.bcount[None]):
        if m.nbond[i] == 0:
            n += 1
    return n
m.count_free = _cf

m.apply_gravity(1.0)
m.apply_rock(10.0, 4.0, 0.62)
m.apply_water(12.0)
m.P[None].fixedRows = 2
m.P[None].walls = 1
m.P[None].substeps = 48
m.P[None].depth = 60.0
m.generate(0, 0)
for f in range(60):
    m.step_physics(0, m.DT)
print("settled intact=%d blocks=%d" % (m.count_intact(), m.bcount[None]))

# leave a big 12-wide, 20-tall chunk hanging by ONE bond at x=10
for yy in range(0, 14):
    for xx in range(1, 30):
        m.stamp_erase(xx + 0.5, yy + 0.5, 0.4)
# keep only x=10,y=14 as the anchor bond to chunk x=9..20, y=14..33
for yy in range(14, 34):
    for xx in range(11, 22):
        m.stamp_erase(xx + 0.5, yy + 0.5, 0.4)
print("after carve blocks=%d intact=%d" % (m.bcount[None], m.count_intact()))

for f in range(120):
    m.step_physics(0, m.DT)
    if f % 20 == 0:
        print("  f=%d intact=%d free=%d maxdisp=%.3f blocks=%d"
              % (f, m.count_intact(), m.count_free(), m.diag_stats(), m.bcount[None]))
print("done")
