import sys
import numpy as np


def log(*a):
    print(*a, flush=True)


log("import render...")
import render
from render import *
log("imported ok")

generate(0, 0)
log("bcount", bcount[None], "bondCount", bondCount[None])
log("--- full frame 0 ---")
step_physics(0, DT)
log("frame0 done, broken=", broken[None], "intact=", count_intact())

DT_F = DT_FLUID


def klog(name, fn):
    log("  [", name, "] ...", flush=True)
    fn()
    log("  [", name, "] ok", flush=True)


# === replicate frame 1 step by step ===
frameNo[None] += 1
frameBrk[None] = 0
klog("decay_bond_damage", decay_bond_damage)
klog("emit_sources", emit_sources)
klog("build_hash", build_hash)
klog("fluid_frame_start", fluid_frame_start)
klog("fluid_step#1", lambda: fluid_step(DT_F))
klog("collide_water(1)", lambda: collide_water(1))
klog("fluid_step#2", lambda: fluid_step(DT_F))
klog("collide_water(0)", lambda: collide_water(0))
klog("fluid_step#3", lambda: fluid_step(DT_F))
klog("collide_water(0)#2", lambda: collide_water(0))
klog("fluid_clamp_disp", fluid_clamp_disp)
klog("compute_top_load", lambda: compute_top_load(0))
klog("count_bonds", count_bonds)
for s in range(SUBSTEPS):
    if s % 12 == 0:
        log("  substep", s, flush=True)
    phys_reset()
    phys_bonds()
    phys_contact(DT)
    phys_integrate(DT)
klog("cleanup_fallen", cleanup_fallen)
log("OK")
