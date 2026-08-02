import sys, time
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
m.P[None].k0 = 0.0

m.generate(0, 0)
print("generated blocks=%d intact=%d" % (m.bcount[None], m.count_intact()))

for f in range(60):
    m.step_physics(0, m.DT)
print("settled intact=%d broken=%d" % (m.count_intact(), m.broken[None]))

for yy in range(0, 12):
    for xx in range(8, 16):
        m.stamp_erase(xx + 0.5, yy + 0.5, 0.4)
print("carved blocks=%d intact=%d" % (m.bcount[None], m.count_intact()))

free0 = m.count_free()
print("free blocks at start=%d" % free0)

mF = 0
fmin = 0.0
bmax = 0.0
for f in range(120):
    m.step_physics(0, m.DT)
    fr = m.count_free()
    if fr > mF:
        mF = fr
    d = m.diag_stats()
    if d < fmin:
        fmin = d
    dd = m.diag_strain()
    if dd > bmax:
        bmax = dd
    if f % 20 == 0:
        print("  f=%d intact=%d free=%d maxdisp=%.4f maxst=%.4f blocks=%d"
              % (f, m.count_intact(), fr, d, dd, m.bcount[None]))
print("RESULT free0=%d maxfree=%d min_maxdisp=%.4f maxStrain=%.4f broken=%d"
      % (free0, mF, fmin, bmax, m.broken[None]))
