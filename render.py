"""ГЕОМАССИВ/2D · DEM — графика отображение (render)

Рендер на GPU (Vulkan), GUI на imgui. Подключает решатель solver.py.
Точка входа — mineudec.py:  python mineudec.py [--selftest]
"""

import argparse
import math
import time

from solver import *

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
        X = int((bx[i] - 0.5) * CELL)
        Y = H - 1 - int((by[i] - 0.5) * CELL)
        if bfixed[i] == 1:
            fill_rect(X, Y, X + CELL - 1, Y + CELL - 1, FIXED)
            diag = FIXED * (1.0 - 0.35) + AMBER * 0.35
            draw_line(X + 3, Y + 3, X + CELL - 3, Y + CELL - 3, diag, 1.0)
            strip = FIXED * (1.0 - 0.14) + AMBER * 0.14
            fill_rect(X, Y + CELL - 3, X + CELL - 1, Y + CELL - 1, strip)
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
                fill_rot_rect(bx[i] * CELL, H - 1 - by[i] * CELL, brot[i], CELL * 0.5, col)
            else:
                fill_rect(X, Y, X + CELL - 1, Y + CELL - 1, col)
                hl = col * (1.0 - 0.07) + WHITE * 0.07
                fill_rect(X, Y + CELL - 2, X + CELL - 1, Y + CELL - 1, hl)


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
    # серые линии по стыкам соединённых блоков — видно, где массив цельный
    for bd in range(bondCount[None]):
        if bondIntact[bd] == 0:
            continue
        a = bondA[bd]
        b = bondB[bd]
        # рисуем по фиксированной сетке клеток-домов, а не по смещённым позициям
        ha = bhome[a]
        hb = bhome[b]
        hax = ha % COLS
        hay = ha // COLS
        hbx = hb % COLS
        hby = hb // COLS
        if bondAxis[bd] == 0:
            x = (ti.max(hax, hbx) + 1) * CELL
            y_top = H - 1 - (ti.min(hay, hby) + 1) * CELL
            y_bot = H - 1 - ti.min(hay, hby) * CELL
            fill_rect(x, y_top, x, y_bot, JOINT)
        else:
            y = H - 1 - (ti.max(hay, hby) + 1) * CELL
            x1 = ti.min(hax, hbx) * CELL
            x2 = (ti.min(hax, hbx) + 1) * CELL
            fill_rect(x1, y, x2, y, JOINT)


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
        if bondAxis[bd] == 0:
            draw_line(mx + j * 0.4, my - CELL * 0.55, mx - j * 0.4, my + CELL * 0.55, CRACK, 2.0)
        else:
            draw_line(mx - CELL * 0.55, my + j * 0.4, mx + CELL * 0.55, my - j * 0.4, CRACK, 2.0)


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

    # инструменты
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

    # вода
    t0 = time.perf_counter()
    generate(0, 0)
    spawn_water(1200, 0.03, COLS * 0.5, ROWS * 0.3, PI * 0.5)
    for f in range(40):
        step_physics(0, DT)
    fp = fill_pct()
    t_wat = time.perf_counter() - t0
    print("selftest: water pcount=%d fill=%.0f%% broken=%d (%.2fs)" % (pCount[None], fp, broken[None], t_wat))
    print("selftest: WARNING — water not contained" if fp < 5.0 else "")

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
    paused = False
    weak = 0
    show_bonds = 1
    E_GPa = 10

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
    flash_msg = "Massiv: svyazey %d" % count_intact()
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
        gui.text("Scena: %s" % ("vyrabotka" if mode == 0 else "sklon"))
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
            apply_rock(E_GPa, float(rp_new), mu_new / 100.0)
        P[None].depth = gui.slider_int("Poroda nad skhemoy, m", int(P[None].depth), 0, 400)
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
        if gui.button("Zapolnit massiv"):
            srcCount[None] = 0
            generate_solid()
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
