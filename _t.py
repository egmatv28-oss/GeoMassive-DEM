import sys,os
sys.path.insert(0,'.')
import mineudec as m
m.apply_gravity(1.0); m.apply_rock(10.0,4.0,0.62); m.apply_water(12.0)
m.P[None].fixedRows=2; m.P[None].walls=1; m.P[None].substeps=48
depth=float(sys.argv[1]); k0=float(sys.argv[2])
m.P[None].depth=depth; m.P[None].k0=k0
m.cavX[None]=0.5; m.cavY[None]=0.5; m.cavRX[None]=0.06; m.cavRY[None]=0.9
m.generate(0,0)
def g():
    d={}
    for i in range(m.bcount[None]):
        c=m.bhome[i]; cx=c%m.COLS; cy=c//m.COLS
        if cx>=0 and cx<m.COLS-1 and m.homeFilled[c+1]==0 and cx>5:
            d.setdefault(cy,[m.bx[i],1e9])[0]=m.bx[i]
        if cx>0 and m.homeFilled[c-1]==0 and cx<m.COLS-5:
            d.setdefault(cy,[1e9,m.bx[i]])[1]=m.bx[i]
    return d
g0=g()
for f in range(900): m.step_physics(0, m.DT)
g1=g()
tot=0;n=0
for y in set(g0)&set(g1):
    l0,r0=g0[y]; l1,r1=g1[y]; tot+=(l1-l0)+(r0-r1); n+=1
print('depth=%.0f k0=%.1f: avg shrink=%.4f broken=%d' % (depth, k0, tot/max(n,1), m.broken[None]))
