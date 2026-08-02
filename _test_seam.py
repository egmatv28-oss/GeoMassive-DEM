import sys
sys.path.insert(0, ".")
import mineudec as m
import taichi as ti

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

for yy in range(0, 12):
    for xx in range(8, 16):
        m.stamp_erase(xx + 0.5, yy + 0.5, 0.4)
print("carved blocks=%d intact=%d" % (m.bcount[None], m.count_intact()))

# measure max tension, max shear offset, max shear force across ALL bonds
@ti.kernel
def diag_bonds():
    m.tmx[None] = 0.0
    m.smx[None] = 0.0
    m.offmax[None] = 0.0
    for bd in range(m.bondCount[None]):
        if m.bondIntact[bd] == 1:
            a = m.bondA[bd]
            b = m.bondB[bd]
            if m.bondAxis[bd] == 0:
                st = m.bx[b] - m.bx[a] - 1.0
                t = m.P[None].kb * st
                off = m.by[b] - m.by[a]
                sf = m.P[None].ks * off
                ti.atomic_max(m.tmx[None], t)
                ti.atomic_max(m.smx[None], ti.abs(sf))
                ti.atomic_max(m.offmax[None], ti.abs(off))
            else:
                st = m.by[b] - m.by[a] - 1.0
                t = m.P[None].kb * st
                off = m.bx[b] - m.bx[a]
                sf = m.P[None].ks * off
                ti.atomic_max(m.tmx[None], t)
                ti.atomic_max(m.smx[None], ti.abs(sf))
                ti.atomic_max(m.offmax[None], ti.abs(off))
m.tmx = ti.field(ti.f32, shape=())
m.smx = ti.field(ti.f32, shape=())
m.offmax = ti.field(ti.f32, shape=())

# what's the overhang weight? count blocks x=8..15
@ti.kernel
def count_overhang() -> ti.i32:
    n = 0
    for i in range(m.bcount[None]):
        if m.bhome[i] % m.COLS >= 8 and m.bhome[i] % m.COLS <= 15:
            n += 1
    return n

print("overhang blocks=%d  weight_units=%.0f" % (count_overhang(), count_overhang() * m.P[None].g))

diag_bonds()
print("post-carve: maxTension=%.1f maxShearF=%.1f maxOff=%.4f  (Ftmax=%.0f Fsmax=%.0f)"
      % (m.tmx[None], m.smx[None], m.offmax[None], m.P[None].rp * m.P[None].ftScale,
         m.P[None].rp * m.P[None].ftScale * m.P[None].fsFactor))

for f in range(120):
    m.step_physics(0, m.DT)
    if f % 30 == 0:
        diag_bonds()
        print("  f=%d intact=%d maxdisp=%.4f maxT=%.0f maxFs=%.0f off=%.4f"
              % (f, m.count_intact(), m.diag_stats(), m.tmx[None], m.smx[None], m.offmax[None]))
