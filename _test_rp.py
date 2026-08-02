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

import subprocess, json

for rp in [4.0, 2.0, 1.0, 0.5]:
    m.apply_gravity(1.0)
    m.apply_rock(10.0, rp, 0.62)
    m.apply_water(12.0)
    m.P[None].fixedRows = 2
    m.P[None].walls = 1
    m.P[None].substeps = 48
    m.P[None].depth = 60.0
    m.generate(0, 0)
    for f in range(60):
        m.step_physics(0, m.DT)
    ok = m.count_intact() == m.bondCount[None]
    for yy in range(0, 12):
        for xx in range(8, 16):
            m.stamp_erase(xx + 0.5, yy + 0.5, 0.4)
    for f in range(120):
        m.step_physics(0, m.DT)
    print("rp=%.1f settleOK=%s intact=%d free=%d maxdisp=%.3f"
          % (rp, ok, m.count_intact(), m.count_free(), m.diag_stats()))
    m.clear_scene()
