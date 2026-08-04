import sys
sys.path.insert(0, ".")
import render as m
import taichi as ti

m.apply_gravity(1.0)
m.apply_rock(10.0, 4.0, 0.62)
m.apply_water(12.0)
m.P[None].fixedRows = 2
m.P[None].walls = 1
m.P[None].substeps = 48
m.P[None].depth = 40.0
m.generate(0, 0)

@ti.kernel
def _maxrot() -> ti.f32:
    mx = 0.0
    for i in range(m.bcount[None]):
        if m.nbond[i] > 0:
            ti.atomic_max(mx, ti.abs(m.brot[i]))
    return mx
m.maxrot = _maxrot

@ti.kernel
def _totrot() -> ti.f32:
    s = 0.0
    n = 0
    for i in range(1, m.bcount[None]):
        if m.nbond[i] > 0:
            s += m.brot[i] * m.brot[i]
            n += 1
    return s if n == 0 else s / n
m.avgsq = _totrot

print("bcount=%d bonds=%d cav=(%.1f,%.1f)" % (m.bcount[None], m.bondCount[None], m.cavX[None], m.cavY[None]))
for f in range(200):
    m.step_physics(0, m.DT)
    if f % 25 == 0:
        print("  f=%d intact=%d max|brot|bonded=%.5f avgsq=%.6f"
              % (f, m.count_intact(), m.maxrot(), m.avgsq()))
print("RESULT intact=%d max|brot|bonded=%.5f" % (m.count_intact(), m.maxrot()))