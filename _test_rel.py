import sys
sys.path.insert(0, ".")
import mineudec as m
import taichi as ti

@ti.kernel
def _maxrel() -> ti.f32:
    mx = 0.0
    for bd in range(m.bondCount[None]):
        if m.bondIntact[bd] == 1:
            a = m.bondA[bd]
            b = m.bondB[bd]
            r = ti.abs(m.brot[b] - m.brot[a])
            ti.atomic_max(mx, r)
    return mx
m.maxrel = _maxrel

@ti.kernel
def _maxbw() -> ti.f32:
    mx = 0.0
    for i in range(m.bcount[None]):
        ti.atomic_max(mx, ti.abs(m.bw[i]))
    return mx
m.maxbw = _maxbw

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
    if f % 10 == 0:
        print("  settle f=%d intact=%d maxrel=%.4f maxbw=%.4f" % (f, m.count_intact(), m.maxrel(), m.maxbw()))
print("settled intact=%d maxrel=%.4f" % (m.count_intact(), m.maxrel()))
