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
log("--- frames ---")
for f in range(4):
    step_physics(0, DT)
    bx_np = bx.to_numpy()[:bcount[None]]
    by_np = by.to_numpy()[:bcount[None]]
    bad = int(np.isnan(bx_np).sum() + np.isnan(by_np).sum())
    log(f"f={f} broken={broken[None]} intact={count_intact()} maxby={float(by_np.max()):.3f} minby={float(by_np.min()):.3f} nan={bad}")
log("OK")
