import mineudec as m
m.apply_gravity(1.0); m.apply_rock(10.0, 4.0, 0.62); m.apply_water(12.0)
m.P[None].fixedRows = 2; m.P[None].walls = 1; m.P[None].substeps = 48
m.generate(0, 0)
for f in range(60):
    m.step_physics(0, m.DT)
for yy in range(0, 12):
    for xx in range(8, 16):
        m.stamp_erase(xx + 0.5, yy + 0.5, 0.4)
print("carved blocks=%d" % m.bcount[None])
# значения прочности
print("rp=%d ftScale=%.0f Ftmax=%.0f Fsmax=%.0f" % (m.P[None].rp, m.P[None].ftScale, m.P[None].rp*m.P[None].ftScale, m.P[None].rp*m.P[None].ftScale*m.P[None].fsFactor))
# вес блока: g
print("g_sim=%.1f (ускорение на блок)" % m.P[None].g)
for f in range(1, 121):
    m.step_physics(0, m.DT)
    if f == 120:
        # найдём макс сдвиг/растяжение связей у консоли (x=8..15, y>=12)
        mx_shear = 0.0; mx_strain = 0.0; nchk = 0
        for bd in range(m.bondCount[None]):
            if m.bondIntact[bd] == 1:
                a = m.bondA[bd]; b = m.bondB[bd]
                if m.bondAxis[bd] == 0:
                    sh = abs(m.by[b]-m.by[a]); st = m.bx[b]-m.bx[a]-1.0
                else:
                    sh = abs(m.bx[b]-m.bx[a]); st = m.by[b]-m.by[a]-1.0
                if sh > mx_shear: mx_shear = sh
                if abs(st) > mx_strain: mx_strain = abs(st)
        print("f=120: maxShearOff=%.4f maxStrain=%.4f" % (mx_shear, mx_strain))
