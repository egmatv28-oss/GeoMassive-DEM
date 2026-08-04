import sys
sys.path.insert(0, ".")
import render as m
import taichi as ti

# Монолит без полости: каждый внутренний блок имеет 4 связи -> вращение
# должно быть полностью защемлено (блок упирается в стенки соседей).
m.apply_gravity(1.0)
m.apply_rock(10.0, 4.0, 0.62)
m.apply_water(0.0)
m.P[None].fixedRows = 2
m.P[None].walls = 1
m.P[None].substeps = 48
m.P[None].depth = 15.0
m.gen_solid()

# максимальное |brot| только у блоков, у которых есть связи
@ti.kernel
def _maxrot_bonded() -> ti.f32:
    mx = 0.0
    for i in range(m.bcount[None]):
        if m.nbond[i] > 0:
            ti.atomic_max(mx, ti.abs(m.brot[i]))
    return mx
m.maxrot_bonded = _maxrot_bonded

# максимальное угловое проникновение: |dx| и |dy| между связанными парами
@ti.kernel
def _maxpenetration() -> ti.f32:
    mx = 0.0
    for bd in range(m.bondCount[None]):
        if m.bondIntact[bd] == 1:
            a = m.bondA[bd]
            b = m.bondB[bd]
            dx = m.bx[b] - m.bx[a]
            dy = m.by[b] - m.by[a]
            ov = 0.0
            if ti.abs(dx) >= ti.abs(dy):
                ov = 1.0 - ti.abs(dx)
            else:
                ov = 1.0 - ti.abs(dy)
            ti.atomic_max(mx, ti.max(0.0, ov))
    return mx
m.maxpen = _maxpenetration

print("bcount=%d bonds=%d" % (m.bcount[None], m.bondCount[None]))
for f in range(300):
    m.step_physics(0, m.DT)
    if f % 50 == 0:
        print("  f=%d intact=%d max|brot|bonded=%.5f maxpen=%.5f strain=%.6f"
              % (f, m.count_intact(), m.maxrot_bonded(), m.maxpen(), m.diag_strain()))
print("RESULT intact=%d max|brot|bonded=%.5f maxpen=%.5f"
      % (m.count_intact(), m.maxrot_bonded(), m.maxpen()))