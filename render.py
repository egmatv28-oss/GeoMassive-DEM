"""ГЕОМАССИВ/2D · DEM — модель, инструменты и графика (render)

Здесь строится МОДЕЛЬ (генерация сцен, полости, ослабленные плоскости,
произвольный мозаичный массив), живут инструменты редактирования
(порода/трещина/ластик), штрих мыши, пыль и вся отрисовка. Модель передаётся
в физический движок solver.py (примитивами add_block_func/add_bond_func либо
bulk-загрузкой build_model). Решатель ничего не знает про сцены и GUI.
Точка входа — mineudec.py:  python mineudec.py [--selftest]
"""

import argparse
import math
import random
import time

import taichi as ti

from solver import *

# =================================================== модель: служебные поля генерации
# (сетка занятости клеток и пометки разрезов используются только при построении
# модели на регулярной сетке; в физике их нет)
cutR = ti.field(ti.i32, N)
cutD = ti.field(ti.i32, N)
noisePts = ti.field(ti.f32, 128)
n1Arr = ti.field(ti.f32, COLS)
n2Arr = ti.field(ti.f32, COLS)
hgtArr = ti.field(ti.f32, COLS)
r_filled = ti.field(ti.i32, N)
r_block = ti.field(ti.i32, N)

img = ti.Vector.field(3, ti.f32, (W, H))

# ================================================================== отрисовка
BG = ti.Vector([0.0431, 0.0667, 0.0941])
GRID = ti.Vector([0.49, 0.608, 0.765])
AMBER = ti.Vector([0.961, 0.647, 0.141])
CYAN = ti.Vector([0.239, 0.784, 1.0])
FIXED = ti.Vector([0.169, 0.212, 0.267])
CRACK = ti.Vector([0.0167, 0.0267, 0.04])
ORANGE = ti.Vector([0.75, 0.312, 0.179])
BLUE = ti.Vector([0.261, 0.395, 0.546])
JOINT = ti.Vector([0.34, 0.41, 0.48])
WHITE = ti.Vector([1.0, 1.0, 1.0])


@ti.func
def fill_rect(x0: ti.i32, y0: ti.i32, x1: ti.i32, y1: ti.i32, col: ti.template()):
    if x0 < 0:
        x0 = 0
    if y0 < 0:
        y0 = 0
    if x1 >= W:
        x1 = W - 1
    if y1 >= H:
        y1 = H - 1
    X = x0
    while X <= x1:
        Y = y0
        while Y <= y1:
            img[X, Y] = col
            Y += 1
        X += 1


@ti.func
def fill_rot_rect(cx: ti.f32, cy: ti.f32, ang: ti.f32, half: ti.f32, col: ti.template()):
    # заливка квадрата со стороной 2*half, повёрнутого на ang вокруг (cx, cy)
    c = ti.cos(ang)
    s = ti.sin(ang)
    # половинки повёрнутого квадрата в проекциях на оси
    hx = half * (ti.abs(c) + ti.abs(s))
    hy = hx
    X = ti.max(0, int(cx - hx))
    while X <= ti.min(W - 1, int(cx + hx)):
        Y = ti.max(0, int(cy - hy))
        while Y <= ti.min(H - 1, int(cy + hy)):
            dx = X - cx
            dy = Y - cy
            ux = c * dx + s * dy
            uy = -s * dx + c * dy
            if ti.abs(ux) <= half and ti.abs(uy) <= half:
                img[X, Y] = col
            Y += 1
        X += 1


@ti.func
def dist_to_seg(px: ti.f32, py: ti.f32, x1: ti.f32, y1: ti.f32, x2: ti.f32, y2: ti.f32) -> ti.f32:
    dx = x2 - x1
    dy = y2 - y1
    l2 = dx * dx + dy * dy
    d = 0.0
    if l2 < 1e-9:
        d = ti.sqrt((px - x1) * (px - x1) + (py - y1) * (py - y1))
    else:
        t = ((px - x1) * dx + (py - y1) * dy) / l2
        if t < 0.0:
            t = 0.0
        if t > 1.0:
            t = 1.0
        cx = x1 + t * dx
        cy = y1 + t * dy
        d = ti.sqrt((px - cx) * (px - cx) + (py - cy) * (py - cy))
    return d


@ti.func
def draw_line(x1: ti.f32, y1: ti.f32, x2: ti.f32, y2: ti.f32, col: ti.template(), w: ti.f32):
    lo_x = int(ti.min(x1, x2) - w)
    hi_x = int(ti.max(x1, x2) + w)
    lo_y = int(ti.min(y1, y2) - w)
    hi_y = int(ti.max(y1, y2) + w)
    X = ti.max(0, lo_x)
    while X <= ti.min(W - 1, hi_x):
        Y = ti.max(0, lo_y)
        while Y <= ti.min(H - 1, hi_y):
            d = dist_to_seg(X + 0.5, Y + 0.5, x1, y1, x2, y2)
            if d <= w * 0.5 + 0.5:
                img[X, Y] = col
            Y += 1
        X += 1


@ti.func
def draw_circle_outline(cx: ti.f32, cy: ti.f32, r: ti.f32, col: ti.template(), w: ti.f32):
    X = int(cx - r - w)
    while X <= int(cx + r + w):
        Y = int(cy - r - w)
        while Y <= int(cy + r + w):
            d = ti.sqrt((X - cx) * (X - cx) + (Y - cy) * (Y - cy))
            if ti.abs(d - r) <= w * 0.5 + 0.5:
                if X >= 0 and X < W and Y >= 0 and Y < H:
                    img[X, Y] = col
            Y += 1
        X += 1


@ti.func
def fill_diamond(cx: ti.f32, cy: ti.f32, rad: ti.f32, col: ti.template()):
    X = int(cx - rad)
    while X <= int(cx + rad):
        Y = int(cy - rad)
        while Y <= int(cy + rad):
            if ti.abs(X - cx) + ti.abs(Y - cy) <= rad:
                if X >= 0 and X < W and Y >= 0 and Y < H:
                    img[X, Y] = col
            Y += 1
        X += 1


@ti.kernel
def render_base():
    for X in range(W):
        for Y in range(H):
            col = BG
            if X % CELL == 0 or Y % CELL == 0:
                a = 0.045
                col = col * (1.0 - a) + GRID * a
            img[X, Y] = col


@ti.kernel
def render_triangles(t: ti.f32, on: ti.i32):
    if on == 1:
        a = 0.35 + 0.2 * ti.sin(t * 3.0)
        col = BG * (1.0 - a) + AMBER * a
        x = 2
        while x < COLS:
            X = x * CELL + CELL // 2
            bob = ti.sin(t * 3.0 + x) * 1.5
            y = int(4.0 + bob)
            while y <= int(12.0 + bob):
                tt = (y - (4.0 + bob)) / (12.0 + bob - (4.0 + bob))
                half = 4.0 * (1.0 - tt)
                px = int(X - half)
                while px <= int(X + half):
                    if px >= 0 and px < W and y >= 0 and y < H:
                        img[px, H - 1 - y] = col
                    px += 1
                y += 1
            x += 4


@ti.kernel
def render_blocks():
    for i in range(bcount[None]):
        ri = bsz[i] * CELL * 0.5
        px = int(bsz[i] * CELL)
        X = int(bx[i] * CELL - ri)
        Y = H - 1 - int(by[i] * CELL + ri)
        if bfixed[i] == 1:
            fill_rect(X, Y, X + px - 1, Y + px - 1, FIXED)
            diag = FIXED * (1.0 - 0.35) + AMBER * 0.35
            draw_line(X + 3, Y + 3, X + px - 3, Y + px - 3, diag, 1.0)
            strip = FIXED * (1.0 - 0.14) + AMBER * 0.14
            fill_rect(X, Y + px - 3, X + px - 1, Y + px - 1, strip)
        else:
            s = bshade[i]
            r = 112.0 + s * 26.0
            g = 121.0 + s * 24.0
            b = 134.0 + s * 22.0
            tq = bstress[i]
            cq = bstressC[i]
            if tq > 0.03:
                u = tq if tq > 1.0 else tq * tq * (3.0 - 2.0 * tq)
                r = r + (255.0 - r) * u
                g = g + (101.0 - g) * u
                b = b + (58.0 - b) * u
            elif cq > 0.05:
                u = cq if cq > 1.0 else cq * cq * (3.0 - 2.0 * cq)
                r = r + (96.0 - r) * u
                g = g + (152.0 - g) * u
                b = b + (228.0 - b) * u
            col = ti.Vector([r / 255.0, g / 255.0, b / 255.0])
            if ti.abs(brot[i]) > 0.02:
                fill_rot_rect(bx[i] * CELL, H - 1 - by[i] * CELL, brot[i], ri, col)
            else:
                fill_rect(X, Y, X + px - 1, Y + px - 1, col)
                hl = col * (1.0 - 0.07) + WHITE * 0.07
                fill_rect(X, Y + px - 2, X + px - 1, Y + px - 1, hl)


@ti.kernel
def render_bonds(show: ti.i32):
    if show == 1:
        for bd in range(bondCount[None]):
            if bondIntact[bd] == 1:
                r = bondR[bd]
                a = bondA[bd]
                b = bondB[bd]
                if r > 0.12:
                    draw_line(bx[a] * CELL, H - 1 - by[a] * CELL, bx[b] * CELL, H - 1 - by[b] * CELL, ORANGE, 2.0)
                elif r < -0.12:
                    draw_line(bx[a] * CELL, H - 1 - by[a] * CELL, bx[b] * CELL, H - 1 - by[b] * CELL, BLUE, 2.0)


@ti.kernel
def render_joints():
    # швы между связанными блоками (линия по общему ребру, поперёк связи)
    for bd in range(bondCount[None]):
        if bondIntact[bd] == 0:
            continue
        a = bondA[bd]
        b = bondB[bd]
        mx = (bx[a] + bx[b]) * 0.5
        my = (by[a] + by[b]) * 0.5
        dx = bx[b] - bx[a]
        dy = by[b] - by[a]
        d = ti.sqrt(dx * dx + dy * dy)
        if d < 1e-6:
            continue
        ux = dx / d
        uy = dy / d
        w = ti.min(bsz[a], bsz[b]) * 0.5
        px = -uy * w * CELL
        py = ux * w * CELL
        cx0 = mx * CELL
        cy0 = H - 1 - my * CELL
        draw_line(cx0 - px, cy0 - py, cx0 + px, cy0 + py, JOINT, 1.0)


@ti.kernel
def render_cracks():
    for bd in range(bondCount[None]):
        if bondIntact[bd] == 1:
            continue
        a = bondA[bd]
        b = bondB[bd]
        dx = bx[b] - bx[a]
        dy = by[b] - by[a]
        if dx * dx + dy * dy > 1.7:
            continue
        mx = (bx[a] + bx[b]) * 0.5 * CELL
        my = H - 1 - (by[a] + by[b]) * 0.5 * CELL
        j = bondJ[bd] * CELL
        hl = ti.min(bsz[a], bsz[b]) * CELL * 0.55
        if ti.abs(dx) >= ti.abs(dy):
            draw_line(mx + j * 0.4, my - hl, mx - j * 0.4, my + hl, CRACK, 2.0)
        else:
            draw_line(mx - hl, my + j * 0.4, mx + hl, my - j * 0.4, CRACK, 2.0)


@ti.kernel
def render_water():
    s = RP_SPH * 2.0 * CELL * 0.95
    core = CYAN
    for i in range(pCount[None]):
        cx = wx[i] * CELL - s * 0.5
        cy = H - 1 - wy[i] * CELL - s * 0.5
        fill_rect(int(cx), int(cy), int(cx + s), int(cy + s), core)


@ti.kernel
def render_dust():
    for i in range(dustCount[None]):
        if i >= MAXDUST:
            continue
        a = dustLife[i] * 0.5
        c = ti.Vector([0.745, 0.784, 0.843])
        if dustCol[i] == 1:
            c = ti.Vector([0.471, 0.667, 0.784])
        cc = BG * (1.0 - a) + c * a
        cx = int(dustX[i] * CELL - 2)
        cy = H - 1 - int(dustY[i] * CELL - 2)
        fill_rect(cx, cy, cx + 3, cy + 3, cc)


@ti.kernel
def render_sources(t: ti.f32):
    for s in range(srcCount[None]):
        X = srcX[s] * CELL
        Y = H - 1 - srcY[s] * CELL
        pulse = (t * 1.1 + srcPh[s]) % 1.0
        r = (0.35 + pulse * 0.95) * CELL
        draw_circle_outline(X, Y, r, CYAN * (0.55 * (1.0 - pulse)) + BG * (1.0 - 0.55 * (1.0 - pulse)), 2.0)
        fill_diamond(X, Y, 5.0, CYAN)


@ti.kernel
def render_mouse(mx: ti.f32, my: ti.f32, brush: ti.f32, is_water: ti.i32):
    if mx > -900.0:
        X = mx * CELL
        Y = H - 1 - my * CELL
        if is_water == 1:
            draw_circle_outline(X, Y, 8.0, CYAN * 0.7, 1.5)
        else:
            r = (brush / 2.0 + 0.3) * CELL
            draw_circle_outline(X, Y, r, AMBER * 0.7, 1.5)


# ================================================================== генерация модели
@ti.func
def cut_between(ax: ti.i32, ay: ti.i32, bxx: ti.i32, byy: ti.i32):
    ok = (ax >= 0 and ay >= 0 and bxx >= 0 and byy >= 0
          and ax < COLS and ay < ROWS and bxx < COLS and byy < ROWS)
    if ok:
        dx = bxx - ax
        dy = byy - ay
        if dy == 0 and ti.abs(dx) == 1:
            cutR[ay * COLS + ti.min(ax, bxx)] = 1
        elif dx == 0 and ti.abs(dy) == 1:
            cutD[ti.min(ay, byy) * COLS + ax] = 1
        elif ti.abs(dx) == 1 and ti.abs(dy) == 1:
            x0 = ti.min(ax, bxx)
            y0 = ti.min(ay, byy)
            cutR[y0 * COLS + x0] = 1
            cutD[y0 * COLS + x0] = 1


@ti.func
def noise_line(step: ti.f32):
    m = int(ti.ceil(COLS / step)) + 2
    i = 0
    while i < m:
        noisePts[i] = ti.random()
        i += 1
    x = 0
    while x < COLS:
        t = x / step
        i0 = int(t)
        f = t - i0
        u = (1.0 - ti.cos(f * PI)) * 0.5
        hgtArr[x] = noisePts[i0] * (1.0 - u) + noisePts[i0 + 1] * u
        x += 1


@ti.kernel
def gen_tunnel(weak: ti.i32):
    for c in range(N):
        r_filled[c] = 0
        r_block[c] = -1
        cutR[c] = 0
        cutD[c] = 0
    tx = COLS * 0.5
    ty = ROWS * 0.6
    trx = 7.0
    try_ = 5.0
    cavX[None] = tx
    cavY[None] = ty
    cavRX[None] = trx
    cavRY[None] = try_
    y = 0
    while y < ROWS:
        x = 0
        while x < COLS:
            cell = y * COLS + x
            fixed = 1
            if y < ROWS - 2:
                dx = (x + 0.5 - tx) / trx
                dy = (y + 0.5 - ty) / try_
                if dx * dx + dy * dy < 1.0:
                    x += 1
                    continue
                fixed = 1 if y >= ROWS - P[None].fixedRows else 0
            i = add_block_func(x + 0.5, y + 0.5, 1.0, fixed)
            if i >= 0:
                r_filled[cell] = 1
                r_block[cell] = i
            x += 1
        y += 1
    if weak == 1:
        b = 0
        while b < 2:
            base = 13 if b == 0 else 33
            ph = ti.random() * 6.0
            x = 6
            while x < COLS - 6:
                yb = base + int(ti.floor(ti.sin(x * 0.22 + ph) * 1.5 + 0.5))
                if yb > 1 and yb < ROWS - 3:
                    cutD[yb * COLS + x] = 1
                x += 1
            b += 1
        jn = 0
        while jn < 3:
            jx = int(8.0 + ti.random() * (COLS - 16))
            ang = PI * 0.5 + (ti.random() - 0.5) * 0.5
            cx2 = jx
            cy2 = 1
            fx = jx + 0.5
            fy = 1.5
            s = 0
            while s < 40:
                fx += ti.cos(ang) * 0.9
                fy += ti.sin(ang) * 0.9
                ang += (ti.random() - 0.5) * 0.3
                nx = int(fx)
                ny = int(fy)
                if nx < 1 or nx >= COLS - 1 or ny < 1 or ny >= ROWS - 2:
                    break
                if nx != cx2 or ny != cy2:
                    cut_between(cx2, cy2, nx, ny)
                    cx2 = nx
                    cy2 = ny
                s += 1
            jn += 1
    c = 0
    while c < N:
        a = r_block[c]
        if a >= 0:
            if (c % COLS) < COLS - 1 and r_block[c + 1] >= 0:
                add_bond_func(a, r_block[c + 1], 1.0, 1 if cutR[c] == 0 else 0)
            if (c // COLS) < ROWS - 1 and r_block[c + COLS] >= 0:
                add_bond_func(a, r_block[c + COLS], 1.0, 1 if cutD[c] == 0 else 0)
        c += 1


@ti.kernel
def gen_slope(weak: ti.i32):
    for c in range(N):
        r_filled[c] = 0
        r_block[c] = -1
        cutR[c] = 0
        cutD[c] = 0
    noise_line(11.0)
    for x in range(COLS):
        n1Arr[x] = hgtArr[x]
    noise_line(4.0)
    for x in range(COLS):
        n2Arr[x] = hgtArr[x]
    for x in range(COLS):
        c1 = x - COLS * 0.44
        c2 = x - COLS * 0.8
        m = 17.0 * ti.exp(-c1 * c1 / (2.0 * 15.0 * 15.0)) + 8.0 * ti.exp(-c2 * c2 / (2.0 * 8.0 * 8.0))
        hgtArr[x] = ROWS * 0.66 - m - n1Arr[x] * 5.0 - n2Arr[x] * 2.2
    cavX[None] = COLS * (0.4 + ti.random() * 0.12)
    cavY[None] = ROWS * 0.74
    cavRX[None] = 5.0 + ti.random() * 2.5
    cavRY[None] = 2.8 + ti.random() * 1.4
    y = 0
    while y < ROWS:
        x = 0
        while x < COLS:
            cell = y * COLS + x
            fixed = 1
            if y < ROWS - 2:
                cutoff = ti.max(4, int(hgtArr[x] + 0.5))
                if y < cutoff:
                    x += 1
                    continue
                dx = (x + 0.5 - cavX[None]) / cavRX[None]
                dy = (y + 0.5 - cavY[None]) / cavRY[None]
                if dx * dx + dy * dy < 1.0:
                    x += 1
                    continue
                fixed = 1 if y >= ROWS - P[None].fixedRows else 0
            i = add_block_func(x + 0.5, y + 0.5, 1.0, fixed)
            if i >= 0:
                r_filled[cell] = 1
                r_block[cell] = i
            x += 1
        y += 1
    if weak == 1:
        nj = 4 + int(ti.random() * 3)
        jn = 0
        while jn < nj:
            jx = int(4.0 + ti.random() * (COLS - 8))
            jy = 0
            while jy < ROWS - 1 and r_filled[jy * COLS + jx] == 0:
                jy += 1
            ang = PI * 0.5 + (ti.random() - 0.5) * 1.1
            cx2 = jx
            cy2 = jy
            fx = jx + 0.5
            fy = jy + 0.5
            ln = int(7.0 + ti.random() * 11)
            s = 0
            while s < ln:
                fx += ti.cos(ang) * 0.9
                fy += ti.sin(ang) * 0.9
                ang += (ti.random() - 0.5) * 0.55
                nx = int(fx)
                ny = int(fy)
                if nx < 1 or nx >= COLS - 1 or ny < 1 or ny >= ROWS - 2 or r_filled[ny * COLS + nx] == 0:
                    break
                if nx != cx2 or ny != cy2:
                    cut_between(cx2, cy2, nx, ny)
                    cx2 = nx
                    cy2 = ny
                s += 1
            jn += 1
    c = 0
    while c < N:
        a = r_block[c]
        if a >= 0:
            if (c % COLS) < COLS - 1 and r_block[c + 1] >= 0:
                add_bond_func(a, r_block[c + 1], 1.0, 1 if cutR[c] == 0 else 0)
            if (c // COLS) < ROWS - 1 and r_block[c + COLS] >= 0:
                add_bond_func(a, r_block[c + COLS], 1.0, 1 if cutD[c] == 0 else 0)
        c += 1
    k = 0
    while k < 240:
        a = ti.random() * 2.0 * PI
        rr = ti.sqrt(ti.random())
        spawn_particle(
            cavX[None] + ti.cos(a) * rr * cavRX[None] * 0.85,
            cavY[None] + ti.sin(a) * rr * cavRY[None] * 0.6 + cavRY[None] * 0.25,
            0.02,
            PI * 0.5,
        )
        k += 1


@ti.kernel
def gen_solid():
    """Полностью заполненный массив породой (монолит без полостей)."""
    for c in range(N):
        r_filled[c] = 0
        r_block[c] = -1
    c = 0
    while c < N:
        fixed = 1 if c // COLS >= ROWS - P[None].fixedRows else 0
        i = add_block_func((c % COLS) + 0.5, (c // COLS) + 0.5, 1.0, fixed)
        if i >= 0:
            r_filled[c] = 1
            r_block[c] = i
        c += 1
    c = 0
    while c < N:
        a = r_block[c]
        if a >= 0:
            if (c % COLS) < COLS - 1 and r_block[c + 1] >= 0:
                add_bond_func(a, r_block[c + 1], 1.0, 1)
            if (c // COLS) < ROWS - 1 and r_block[c + COLS] >= 0:
                add_bond_func(a, r_block[c + COLS], 1.0, 1)
        c += 1


def build_bigblocks():
    """Сцена с блоками ПРОИЗВОЛЬНОГО размера (мозаика 1..3 клетки).

    Модель строится на CPU (списки) и передаётся в решатель через build_model —
    демонстрация, что решатель не привязан ни к сетке, ни к размеру блока.
    """
    cx = COLS * 0.5
    cy = ROWS * 0.6
    rx = 7.0
    ry = 5.0
    occ = [[0] * COLS for _ in range(ROWS)]
    for y in range(ROWS):
        for x in range(COLS):
            dx = (x + 0.5 - cx) / rx
            dy = (y + 0.5 - cy) / ry
            if dx * dx + dy * dy < 1.0:
                occ[y][x] = -1   # полость выработки
    blocks = []
    for y in range(ROWS):
        for x in range(COLS):
            if occ[y][x] != 0:
                continue
            maxs = min(3, COLS - x, ROWS - y)
            r = random.random()
            s = 3 if r < 0.2 else (2 if r < 0.55 else 1)
            if s > maxs:
                s = maxs
            fits = True
            for yy in range(y, y + s):
                for xx in range(x, x + s):
                    if occ[yy][xx] != 0:
                        fits = False
                        break
                if not fits:
                    break
            if not fits:
                occ[y][x] = 1
                continue
            fixed = 1 if y >= ROWS - P[None].fixedRows else 0
            idx = len(blocks)
            blocks.append((x + s / 2.0, y + s / 2.0, float(s), fixed))
            for yy in range(y, y + s):
                for xx in range(x, x + s):
                    occ[yy][xx] = idx + 1
    bonds = []
    seen = set()
    for y in range(ROWS):
        for x in range(COLS):
            b = occ[y][x]
            if b <= 0:
                continue
            a0 = b - 1
            for (xx, yy) in ((x + 1, y), (x, y + 1)):
                if xx >= COLS or yy >= ROWS:
                    continue
                c = occ[yy][xx]
                if c <= 0 or c == b:
                    continue
                a1 = c - 1
                if (a0, a1) in seen or (a1, a0) in seen:
                    continue
                dx = blocks[a1][0] - blocks[a0][0]
                dy = blocks[a1][1] - blocks[a0][1]
                rest = abs(dx) if abs(dx) >= abs(dy) else abs(dy)
                bonds.append((a0, a1, rest, 1))
                seen.add((a0, a1))
    water = []
    for _ in range(200):
        a = random.uniform(0.0, 2.0 * math.pi)
        rr = math.sqrt(random.random())
        water.append((cx + math.cos(a) * rr * rx * 0.85,
                      cy + math.sin(a) * rr * ry * 0.6 + ry * 0.25,
                      0.02, math.pi * 0.5))
    build_model(blocks, bonds, water, (cx, cy, rx, ry))


def generate(mode, weak):
    """Построить модель сцены и передать её в решатель.

    mode 0 — выработка (единичные блоки), mode 1 — склон, mode 2 — мозаика
    произвольных размеров (build_model). weak — ослабленные плоскости."""
    if mode == 2:
        build_bigblocks()
        return
    reset_all()
    if mode == 0:
        gen_tunnel(weak)
    else:
        gen_slope(weak)


@ti.kernel
def rebuild_occ():
    for c in range(N):
        r_block[c] = -1
    for i in range(bcount[None]):
        cxi = clamp_cell(bx[i])
        cyi = clamp_cell(by[i])
        r_block[cyi * COLS + cxi] = i


# ================================================================== инструменты
@ti.kernel
def stamp_rock(x: ti.f32, y: ti.f32, r: ti.f32):
    cx = int(x)
    cy = int(y)
    rr = int(ti.ceil(r))
    oy = -rr
    while oy <= rr:
        ox = -rr
        while ox <= rr:
            ax = cx + ox
            ay = cy + oy
            if ax >= 0 and ax < COLS and ay >= 0 and ay < ROWS - 1:
                if ox * ox + oy * oy <= r * r:
                    cell = ay * COLS + ax
                    if r_block[cell] < 0:
                        fixed = 1 if ay >= ROWS - P[None].fixedRows else 0
                        i = add_block_func(ax + 0.5, ay + 0.5, 1.0, fixed)
                        if i >= 0:
                            if ax > 0 and r_block[cell - 1] >= 0:
                                add_bond_func(r_block[cell - 1], i, 1.0, 1)
                            if ax < COLS - 1 and r_block[cell + 1] >= 0:
                                add_bond_func(i, r_block[cell + 1], 1.0, 1)
                            if ay > 0 and r_block[cell - COLS] >= 0:
                                add_bond_func(r_block[cell - COLS], i, 1.0, 1)
                            if ay < ROWS - 1 and r_block[cell + COLS] >= 0:
                                add_bond_func(i, r_block[cell + COLS], 1.0, 1)
                            r_block[cell] = i
            ox += 1
        oy += 1


@ti.kernel
def stamp_erase(x: ti.f32, y: ti.f32, r: ti.f32):
    i = bcount[None] - 1
    while i >= 0:
        if bfixed[i] == 0:
            dx = bx[i] - x
            dy = by[i] - y
            if dx * dx + dy * dy < r * r:
                remove_block_func(i)
        i -= 1


@ti.kernel
def crack_at(x: ti.f32, y: ti.f32, r: ti.f32):
    r2 = r * r
    for bd in range(bondCount[None]):
        if bondIntact[bd] == 1:
            mx = (bx[bondA[bd]] + bx[bondB[bd]]) * 0.5 - x
            my = (by[bondA[bd]] + by[bondB[bd]]) * 0.5 - y
            if mx * mx + my * my < r2:
                a = bondA[bd]
                b = bondB[bd]
                bondIntact[bd] = 0
                clear_bond_bit(a, b)
                ti.atomic_add(broken[None], 1)
                breakCrush[None] = 0
                dx = bx[b] - bx[a]
                dy = by[b] - by[a]
                d = ti.sqrt(dx * dx + dy * dy)
                kx = 0.15
                ky = 0.0
                if d > 1e-9:
                    kx = dx / d * 0.15
                    ky = dy / d * 0.15
                bvx[a] -= kx
                bvy[a] -= ky
                bvx[b] += kx
                bvy[b] += ky


@ti.kernel
def clear_scene():
    """Оставить только закреплённые блоки, связи пересобрать по геометрии."""
    n = 0
    i = 0
    while i < bcount[None]:
        if bfixed[i] == 1:
            if i != n:
                bx[n] = bx[i]
                by[n] = by[i]
                bvx[n] = bvx[i]
                bvy[n] = bvy[i]
                bfx[n] = bfx[i]
                bfy[n] = bfy[i]
                wfx[n] = wfx[i]
                wfy[n] = wfy[i]
                ovf[n] = ovf[i]
                bfixed[n] = bfixed[i]
                bshade[n] = bshade[i]
                bstress[n] = bstress[i]
                bstressC[n] = bstressC[i]
                brot[n] = brot[i]
                bw[n] = bw[i]
                btor[n] = btor[i]
                nbond[n] = nbond[i]
                bsz[n] = bsz[i]
                hx[n] = hx[i]
                hy[n] = hy[i]
            n += 1
        i += 1
    bcount[None] = n
    bondCount[None] = 0
    i = 0
    while i < n:
        j = i + 1
        while j < n:
            ri = bsz[i] * 0.5
            rj = bsz[j] * 0.5
            dx = bx[j] - bx[i]
            dy = by[j] - by[i]
            axx = ti.abs(dx)
            ayy = ti.abs(dy)
            rs = ri + rj
            if axx > rs - 0.05 and axx < rs + 0.05 and ayy < ti.min(ri, rj):
                add_bond_func(i, j, ti.sqrt(dx * dx + dy * dy), 1)
            elif ayy > rs - 0.05 and ayy < rs + 0.05 and axx < ti.min(ri, rj):
                add_bond_func(i, j, ti.sqrt(dx * dx + dy * dy), 1)
            j += 1
        i += 1
    pCount[None] = 0
    dustCount[None] = 0
    broken[None] = 0
    for s in range(FRIC_SLOTS):
        fricKey[s] = -1
        fricS[s] = 0.0
        fricF[s] = 0


@ti.kernel
def dust_step():
    for i in range(dustCount[None]):
        if i >= MAXDUST:
            continue
        dustX[i] += dustVX[i]
        dustY[i] += dustVY[i]
        dustVY[i] += 0.004
        dustLife[i] -= 0.04


@ti.kernel
def dust_compact():
    w = 0
    i = 0
    while i < dustCount[None]:
        if i < MAXDUST and dustLife[i] > 0.0:
            dustX[w] = dustX[i]
            dustY[w] = dustY[i]
            dustVX[w] = dustVX[i]
            dustVY[w] = dustVY[i]
            dustLife[w] = dustLife[i]
            dustCol[w] = dustCol[i]
            w += 1
        i += 1
    dustCount[None] = w


# ================================================================== ввод
def do_stroke(mx, my, last_mx, last_my, tool, brush):
    dx = mx - last_mx
    dy = my - last_my
    dist = math.hypot(dx, dy)
    steps = max(1, int(math.ceil(dist / 0.35)))
    r_rock = brush / 2.0 + 0.4
    r_crack = brush / 2.0 + 0.25
    if tool == 0:
        rebuild_occ()
    for s in range(1, steps + 1):
        t = s / steps
        x = last_mx + dx * t
        y = last_my + dy * t
        if tool == 0:
            stamp_rock(x, y, r_rock)
        elif tool == 1:
            crack_at(x, y, r_crack)
        elif tool == 3:
            stamp_erase(x, y, r_rock)
    return mx, my


# ================================================================== selftest / GUI
def run_selftest(frames=120):
    apply_gravity(1.0)
    apply_rock(10.0, 4.0, 0.62)
    apply_water(12.0)
    P[None].fixedRows = 2
    P[None].walls = 1
    P[None].substeps = 48
    P[None].depth = 60.0

    t0 = time.perf_counter()
    generate(0, 0)
    t_gen = time.perf_counter() - t0
    nbi = count_intact()
    print("selftest: generated mode=tunnel blocks=%d bonds=%d intact=%d (%.2fs)" % (bcount[None], bondCount[None], nbi, t_gen))

    t0 = time.perf_counter()
    for f in range(frames):
        step_physics(0, DT)
        if f < 20 or f % 20 == 0:
            print(
                "  f=%d intact=%d broken=%d maxdisp=%.3f maxst=%.3f blocks=%d"
                % (f, count_intact(), broken[None], diag_stats(), diag_strain(), bcount[None])
            )
    t_phys = time.perf_counter() - t0
    fill = fill_pct()
    print(
        "selftest: %d frames in %.2fs (%.1f fps) blocks=%d particles=%d broken=%d fill=%.0f%%"
        % (frames, t_phys, frames / t_phys, bcount[None], pCount[None], broken[None], fill)
    )
    if broken[None] > 0:
        print("selftest: WARNING — bonds broke during settle (broken=%d)" % broken[None])

    t0 = time.perf_counter()
    crack_at(COLS * 0.5, ROWS * 0.2, 2.5)
    nb1 = count_intact()
    n0 = bcount[None]
    stamp_rock(COLS * 0.5, ROWS * 0.6, 4.0)
    nb2 = bcount[None] - n0
    stamp_erase(COLS * 0.5, ROWS * 0.6, 4.0)
    nb3 = n0 - bcount[None]
    for f in range(10):
        step_physics(0, DT)
    clear_scene()
    nb4 = bcount[None]
    t_tools = time.perf_counter() - t0
    print(
        "selftest: tools crack->%d bonds, stamp->+%d blocks, erase->-%d, clear->%d blocks (%.2fs)"
        % (nb1, nb2, nb3, nb4, t_tools)
    )

    t0 = time.perf_counter()
    generate(0, 0)
    spawn_water(1200, 0.03, COLS * 0.5, ROWS * 0.3, PI * 0.5)
    for f in range(40):
        step_physics(0, DT)
    fp = fill_pct()
    t_wat = time.perf_counter() - t0
    print("selftest: water pcount=%d fill=%.0f%% broken=%d (%.2fs)" % (pCount[None], fp, broken[None], t_wat))
    print("selftest: WARNING — water not contained" if fp < 5.0 else "")

    # мозаика произвольных размеров
    t0 = time.perf_counter()
    generate(2, 0)
    ms = P[None].maxSize
    for f in range(20):
        step_physics(2, DT)
    t_big = time.perf_counter() - t0
    print(
        "selftest: bigblocks n=%d maxSize=%d intact=%d broken=%d (%.2fs)"
        % (bcount[None], ms, count_intact(), broken[None], t_big)
    )
    if broken[None] > bcount[None] // 4:
        print("selftest: WARNING — bigblocks collapsed (broken=%d)" % broken[None])

    # рендер одного кадра (без окна)
    t0 = time.perf_counter()
    render_base()
    render_triangles(0.0, 1)
    render_blocks()
    render_joints()
    render_bonds(1)
    render_cracks()
    render_water()
    render_dust()
    render_sources(0.0)
    render_mouse(-999.0, 0.0, 2.0, 0)
    t_render = time.perf_counter() - t0
    print("selftest: one render pass in %.1f ms" % (t_render * 1000.0))
    print("selftest OK")


def run_gui():
    apply_gravity(1.0)
    apply_rock(10.0, 4.0, 0.62)
    apply_water(12.0)
    P[None].fixedRows = 2
    P[None].walls = 1
    P[None].substeps = 48
    P[None].depth = 60.0

    mode = 0
    tool = 0
    brush = 2
    paused = True
    weak = 0
    show_bonds = 1
    E_GPa = 10
    rho = RHO_R
    scene_names = {0: "vyrabotka", 1: "sklon", 2: "krupnye bloki"}

    window = ti.ui.Window("ГЕОМАССИВ/2D · DEM", (W, H), vsync=True)
    canvas = window.get_canvas()
    gui = window.get_gui()

    generate(0, 0)

    last = time.perf_counter()
    fps = 60.0
    hud_t = 0.0
    last_mx = -1.0
    last_my = -1.0
    rain_until = 0.0
    flash_msg = "Nazhmite СТАРТ - simulyatsiya nachnetsya"
    flash_t = 0.0
    last_broken = 0
    last_hud_broken = 0

    while window.running:
        now = time.perf_counter()
        dt_real = min(0.05, now - last)
        last = now
        t_sim = now

        for e in window.get_events(ti.ui.PRESS):
            k = e.key
            if k == ti.ui.ESCAPE:
                window.running = False
            elif k == ti.ui.SPACE:
                paused = not paused
            elif k in ("1", "2", "3", "4"):
                tool = int(k) - 1
            elif k == ti.ui.LMB:
                pos = window.get_cursor_pos()
                mx = pos[0] * COLS
                my = (1.0 - pos[1]) * ROWS
                if pos[0] > 0.71:
                    continue
                if tool == 2:
                    flash_msg = place_source(mx, my)
                    flash_t = t_sim
                else:
                    last_mx = mx
                    last_my = my
                    last_mx, last_my = do_stroke(mx, my, last_mx, last_my, tool, brush)
            elif k == ti.ui.RMB:
                pos = window.get_cursor_pos()
                if remove_source(pos[0] * COLS, (1.0 - pos[1]) * ROWS):
                    flash_msg = "Istochnik ubran"
                    flash_t = t_sim

        if window.is_pressed(ti.ui.LMB):
            pos = window.get_cursor_pos()
            if pos[0] <= 0.71:
                mx = pos[0] * COLS
                my = (1.0 - pos[1]) * ROWS
                if tool != 2 and last_mx >= 0:
                    last_mx, last_my = do_stroke(mx, my, last_mx, last_my, tool, brush)
                if tool == 2 and last_mx < 0:
                    pass

        # ---- GUI
        gui.begin("Upravlenie", 0.72, 0.02, 0.27, 0.96)
        if gui.button("СТАРТ" if paused else "СТОП"):
            paused = not paused
            flash_msg = "Симуляция запущена" if not paused else "Пауза"
            flash_t = t_sim
        gui.text("Scena: %s" % scene_names[mode])
        gui.text("Poroda: E=%d GPa, Rt=%d MPa, Rc=%d MPa" % (E_GPa, int(P[None].rp), int(P[None].rp * P[None].rcFactor)))
        gui.text("Napor: %.2f MPa na porodu" % (RHO_W * G_PHYS * P[None].head / 1e6))
        gui.text("LKM - instrument * PKM - istochnik")
        if gui.button("Poroda (1)"):
            tool = 0
        if gui.button("Treshchina (2)"):
            tool = 1
        if gui.button("Istochnik vody (3)"):
            tool = 2
        if gui.button("Lastik (4)"):
            tool = 3
        brush = gui.slider_int("Kist", brush, 1, 6)
        E_new = gui.slider_int("Modul E, GPa", E_GPa, 1, 20)
        rp_new = gui.slider_int("Rp (otriv) MPa", int(P[None].rp), 1, 15)
        mu_new = gui.slider_int("Trenie mu %%", int(P[None].mu * 100), 20, 90)
        if E_new != E_GPa or rp_new != int(P[None].rp) or mu_new != int(P[None].mu * 100):
            E_GPa = E_new
            apply_rock(E_GPa, float(rp_new), mu_new / 100.0, rho)
        P[None].depth = gui.slider_int("Poroda nad skhemoy, m", int(P[None].depth), 0, 3000)
        rho_new = gui.slider_int("Plotnost porody, kg/m3", int(rho), 1000, 3500)
        if rho_new != rho:
            rho = float(rho_new)
            apply_rock(E_GPa, float(P[None].rp), P[None].mu, rho)
            apply_water(P[None].head)
        P[None].k0 = gui.slider_float("K0 bokovogo davleniya", P[None].k0, 0.0, 2.0)
        P[None].fixedRows = gui.slider_int("Zakrepl. ryadov", P[None].fixedRows, 1, 4)
        P[None].brkCap = gui.slider_int("Limit razryvov/kadr", P[None].brkCap, 1, 200)
        P[None].dmgMax = gui.slider_int("Tr. povrezhdeniya, kadr", int(P[None].dmgMax), 1, 60)
        walls = gui.checkbox("Bokovye stenki", P[None].walls == 1)
        P[None].walls = 1 if walls else 0
        weak_new = gui.checkbox("Oslablennye ploskosti", weak == 1)
        if weak_new != weak:
            weak = 1 if weak_new else 0
            srcCount[None] = 0
            generate(mode, weak)
            flash_msg = "Massiv %s: svyazey %d" % ("oslablen" if weak else "monolitny", count_intact())
            flash_t = t_sim
        gv = gui.slider_int("Gravitatsiya x0.1g", int(P[None].g / G_SIM * 10.0), 0, 20)
        apply_gravity(gv / 10.0)
        if gui.button("Vyrabotka"):
            if mode != 0:
                mode = 0
                srcCount[None] = 0
                generate(mode, weak)
        if gui.button("Sklon"):
            if mode != 1:
                mode = 1
                srcCount[None] = 0
                generate(mode, weak)
        if gui.button("Krupnye bloki"):
            if mode != 2:
                mode = 2
                srcCount[None] = 0
                generate(mode, weak)
                flash_msg = "Mozaika: blokov %d, max razmer %d" % (bcount[None], P[None].maxSize)
                flash_t = t_sim
        if gui.button("Zapolnit massiv"):
            srcCount[None] = 0
            reset_all()
            gen_solid()
            flash_msg = "Massiv zapolnen: blokov %d" % bcount[None]
            flash_t = t_sim
        if gui.button("Peregenerirovat"):
            srcCount[None] = 0
            generate(mode, weak)
            flash_msg = "Massiv: svyazey %d" % count_intact()
            flash_t = t_sim
        if gui.button("Dozhd"):
            rain_until = t_sim + 2.5
            flash_msg = "Liven - voda ishchet treshchiny"
            flash_t = t_sim
        if gui.button("Ochistit porodu"):
            clear_scene()
            flash_msg = "Poroda ubrana"
            flash_t = t_sim
        show_bonds = 1 if gui.checkbox("Pokazyvat svyazi", show_bonds == 1) else 0
        apply_water(float(gui.slider_int("Napor H, m", int(P[None].head), 1, 40)))
        if gui.button("Slit vodu"):
            pCount[None] = 0
            filledFlag[None] = 0
            flash_msg = "Voda slita"
            flash_t = t_sim
        if gui.button("Ubrat istochniki"):
            srcCount[None] = 0
            flash_msg = "Istochniki ubrany"
            flash_t = t_sim
        if gui.button("Pauza / Pusk (Space)"):
            paused = not paused
            if not paused:
                flash_msg = "Симуляция запущена"
            flash_t = t_sim
        fp = fill_pct()
        gui.text("fps %.0f * blokov %d" % (fps, bcount[None]))
        gui.text("chastits h2o %d%s * svyazey %d" % (pCount[None], " (MAX)" if pCount[None] >= PMAX else "", count_intact()))
        gui.text("razryvov %d * zapolnenie %.0f%%" % (broken[None], fp))
        if flash_msg and (t_sim - flash_t < 2.0):
            gui.text(flash_msg)
        gui.end()

        # ---- симуляция
        if not paused:
            if t_sim < rain_until:
                rx = 1.0 + (math.sin(t_sim * 13.7) * 0.5 + 0.5) * (COLS - 2)
                spawn_water(2, 0.03, rx, 1.2, PI * 0.5)
            step_physics(mode, DT)
            dust_step()
            dust_compact()
            if broken[None] != last_broken:
                if broken[None] % 8 == 1:
                    flash_msg = "Smyatie porody (Rc)" if breakCrush[None] == 1 else "Razryv svyazi (Rp)"
                    flash_t = t_sim
                last_broken = broken[None]
            if filledFlag[None] == 0 and fp >= 95:
                filledFlag[None] = 1
                flash_msg = "Vyrabotka zapolnena vodoy"
                flash_t = t_sim
        else:
            if last_broken != broken[None]:
                last_broken = broken[None]

        # ---- отрисовка
        render_base()
        render_triangles(t_sim, 1 if (mode == 0 and P[None].depth > 0) else 0)
        render_blocks()
        render_joints()
        render_bonds(show_bonds)
        render_cracks()
        render_water()
        render_dust()
        render_sources(t_sim)

        pos = window.get_cursor_pos()
        if pos[0] <= 0.71:
            render_mouse(pos[0] * COLS, (1.0 - pos[1]) * ROWS, brush, 1 if tool == 2 else 0)
        else:
            render_mouse(-999.0, 0.0, brush, 0)

        canvas.set_image(img)
        window.show()

        fps += (1.0 / max(1e-6, dt_real) - fps) * 0.06
        if now - hud_t > 0.25:
            hud_t = now


def main():
    ap = argparse.ArgumentParser(description="ГЕОМАССИВ/2D · DEM на Taichi")
    ap.add_argument("--selftest", action="store_true", help="прогон физики без окна")
    ap.add_argument("--frames", type=int, default=120)
    args = ap.parse_args()
    if args.selftest:
        run_selftest(args.frames)
    else:
        run_gui()


if __name__ == "__main__":
    main()
