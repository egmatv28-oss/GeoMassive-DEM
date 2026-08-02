import sys
sys.path.insert(0, ".")
import mineudec as m

m.apply_gravity(1.0)
m.apply_rock(10.0, 4.0, 0.62)
m.apply_water(12.0)
m.P[None].fixedRows = 2
m.P[None].walls = 1
m.P[None].substeps = 48
m.P[None].depth = 60.0
m.generate(0, 0)
for f in range(3):
    m.dbgCnt[None] = 0
    m.dbgTor[None] = 0.0
    m.dbgDx[None] = 0.0
    m.dbgDy[None] = 0.0
    m.step_physics(0, m.DT)
    print("f=%d contacts=%d |dx|=%.4f |dy|=%.4f free_tor=%.1f intact=%d"
          % (f, m.dbgCnt[None], m.dbgDx[None], m.dbgDy[None], m.dbgTor[None], m.count_intact()))
