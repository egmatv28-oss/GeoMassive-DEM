import sys
sys.path.insert(0, ".")
import render as m
import taichi as ti

decay = float(sys.argv[1]) if len(sys.argv) > 1 else 0.85

m.apply_gravity(1.0)
m.apply_rock(10.0, 4.0, 0.62)
m.apply_water(8.0)
m.P[None].dmgDecay = decay
m.P[None].fixedRows = 2
m.P[None].walls = 1
m.P[None].substeps = 48
m.P[None].depth = 40.0
m.generate(0, 0)

for f in range(90):
    m.step_physics(0, m.DT)

# подрываем обширную область над выработкой -> она обрушается
m.crack_at(m.cavX[None], m.cavY[None] + 6.0, 8.0)

broken0 = m.broken[None]
for f in range(500):
    m.step_physics(0, m.DT)

@ti.kernel
def _maxrot() -> ti.f32:
    mx = 0.0
    for i in range(m.bcount[None]):
        if m.nbond[i] > 0:
            ti.atomic_max(mx, ti.abs(m.brot[i]))
    return mx
m.maxrot = _maxrot

print("dmgDecay=%.2f broken_total=%d (до подрыва %d) intact_after=%d/%d maxrot_bonded=%.5f"
      % (decay, m.broken[None], broken0, m.count_intact(), m.bondCount[None], m.maxrot()))