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
# Позиции блоков/воды/пыли в решателе — в МЕТРАХ (см. solver). Чтобы нарисовать их
# в пикселях (CELL пикселей на клетку), нужен масштаб PX = CELL / LCELL.
PX = CELL / LCELL

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

# камера: CAMX/CAMY — смещение картинки в пикселях окна, CAMZ — масштаб
# (во сколько раз увеличен рисунок; 1.0 = как раньше). Хранятся в полях,
# чтобы их видели GPU-ядра отрисовки.
CAMX = ti.field(ti.f32, ())
CAMY = ti.field(ti.f32, ())
CAMZ = ti.field(ti.f32, ())
CAMX[None] = 0.0
CAMY[None] = 0.0
CAMZ[None] = 1.0
CAM_ZMIN = 0.25
CAM_ZMAX = 8.0

# python-зеркало камеры: хост-код (курсор/зум/пан) работает с ним, а не читает
# поля — каждый read скалярного поля из Python это синхронизация CPU<->GPU.
# GPU-ядра рендера по-прежнему читают поля CAMX/CAMY/CAMZ.
_CAM = [0.0, 0.0, 1.0]


def cam_write(ox, oy, oz):
    _CAM[0] = ox
    _CAM[1] = oy
    _CAM[2] = oz
    CAMX[None] = ox
    CAMY[None] = oy
    CAMZ[None] = oz


# =================================================== ввод символов с клавиатуры
# Taichi GGUI (button_id_to_name в taichi/ui/utils/utils.h) отдаёт события
# ТОЛЬКО для букв и спецклавиш (Shift/Return/Escape/BackSpace/Space/стрелки):
# GLFW-коды цифр, минуса и точки кидают std::runtime_error и событие ТЕРЯЕТСЯ
# в window_base.cpp. То есть цифры через window.get_events() не приходят
# НИКОГДА — старый «ввод числа угла» был мёртв по этой причине. Символы
# читаем напрямую из состояния клавиатуры (user32.GetAsyncKeyState), ловя
# фронт нажатия up->down (авто-повтор ОС не дублирует ввод). Опрос идёт
# только когда окно Mineudec активно.
import ctypes

_user32 = ctypes.windll.user32
_user32.GetActiveWindow.restype = ctypes.c_void_p
_user32.GetForegroundWindow.restype = ctypes.c_void_p
_user32.GetWindowTextW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_int]
_vk_chars = {}
for _d in range(10):
    _vk_chars[0x30 + _d] = str(_d)     # основной ряд цифр
    _vk_chars[0x60 + _d] = str(_d)     # numpad
_vk_chars[0xBD] = "-"                  # VK_OEM_MINUS
_vk_chars[0xBE] = "."                  # VK_OEM_PERIOD
_vk_chars[0xBC] = "."                  # VK_OEM_COMMA (запятая = десятичная точка)
_vk_chars[0x6E] = "."                  # VK_DECIMAL (numpad)
_vk_prev = set()
_our_hwnd = None
WIN_TITLE = "Геомассив/2D и DEM"


def _app_focused():
    """Активно ли окно Mineudec: опрос клавиш — только для своего окна."""
    global _our_hwnd
    if _our_hwnd is None:
        _our_hwnd = _user32.GetActiveWindow()
        if not _our_hwnd:
            hwnd = _user32.GetForegroundWindow()
            buf = ctypes.create_unicode_buffer(256)
            if hwnd:
                _user32.GetWindowTextW(hwnd, buf, 256)
            if buf.value.startswith(WIN_TITLE):
                _our_hwnd = hwnd
        if not _our_hwnd:
            _our_hwnd = -1   # окно не нашли — работаем без проверки фокуса
    if _our_hwnd == -1:
        return True
    return _user32.GetForegroundWindow() == _our_hwnd


def poll_type_chars():
    """Символы для набора числа: цифры, '-', '.' (',' и numpad-точка = '.').

    Список символов по фронтам нажатия (up->down); пусто, если окно не активно."""
    out = []
    if not _app_focused():
        _vk_prev.clear()
        return out
    for vk, ch in _vk_chars.items():
        down = (_user32.GetAsyncKeyState(vk) & 0x8000) != 0
        was = vk in _vk_prev
        if down and not was:
            out.append(ch)
        if down:
            _vk_prev.add(vk)
        else:
            _vk_prev.discard(vk)
    return out


def parse_deg(buf):
    try:
        return float(buf.replace(",", "."))
    except ValueError:
        return None

# ================================================================== отрисовка
BG = ti.Vector([0.0431, 0.0667, 0.0941])
GRID = ti.Vector([0.49, 0.608, 0.765])
AMBER = ti.Vector([0.961, 0.647, 0.141])
CYAN = ti.Vector([0.239, 0.784, 1.0])
FIXED = ti.Vector([0.169, 0.212, 0.267])
CRACK = ti.Vector([0.0167, 0.0267, 0.04])
ORANGE = ti.Vector([0.75, 0.312, 0.179])
BLUE = ti.Vector([0.261, 0.395, 0.546])
PURPLE = ti.Vector([0.72, 0.32, 0.9])
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
    # ang в соглашении физики: brot>0 = по часовой на экране (мир y вниз),
    # буфер же Y вверх, поэтому тест R(+ang)·d даёт фигуру R(-ang) — по часовой
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
            ux = c * dx - s * dy
            uy = s * dx + c * dy
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
    z = CAMZ[None]
    k = PX * z
    ox = CAMX[None]
    oy = CAMY[None]
    d = 0.5 / k
    for X in range(W):
        for Y in range(H):
            col = BG
            wx = (X + 0.5 - ox) / k
            wy = (H - 1.5 - Y - oy) / k
            if ti.abs(wx - ti.round(wx)) < d or ti.abs(wy - ti.round(wy)) < d:
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


@ti.func
def stress_u(q: ti.f32) -> ti.f32:
    """Интенсивность цвета напряжения (0..1): гладкая кривая до предела
    прочности (q=1), выше — насыщенный. Та же функция для блоков и легенды."""
    u = q if q > 1.0 else q * q * (3.0 - 2.0 * q)
    if u > 1.0:
        u = 1.0
    return u


@ti.func
def stress_band(q: ti.f32, pk: ti.f32) -> ti.i32:
    """Номер цветового интервала 0..9 для напряжения q (нормировано на прочность)
    при пике шкалы pk: шкала разбита на 10 равных интервалов 0..pk."""
    qq = 0.0
    if pk > 1e-6:
        qq = q / pk
    if qq > 1.0:
        qq = 1.0
    k = int(qq * 10.0)
    if k > 9:
        k = 9
    return k


@ti.func
def legend_band_color(scale: ti.i32, k: ti.i32):
    """Цвет интервала k (0..9) шкалы легенды: scale 0 — растяжение
    (серый->красный), 1 — сжатие (серый->синий). Одна функция и для квадратов
    легенды, и для блоков — поэтому цвет блока всегда совпадает с квадратом
    своего интервала на шкале."""
    uu = stress_u((k + 1.0) / 10.0)
    r = 112.0
    g = 121.0
    b = 134.0
    if scale == 0:
        r = r + (255.0 - r) * uu
        g = g + (101.0 - g) * uu
        b = b + (58.0 - b) * uu
    else:
        r = r + (96.0 - r) * uu
        g = g + (152.0 - g) * uu
        b = b + (228.0 - b) * uu
    return ti.Vector([r / 255.0, g / 255.0, b / 255.0])


@ti.kernel
def stress_peaks():
    """Пики напряжений по сцене для динамической шкалы легенды (нормировано на
    прочность: 0 — растяжение, 1 — сжатие). Пик растёт мгновенно, спадает
    медленно — значения на шкале меняются плавно, без дёрганья."""
    stressPeakRaw[0] = 0.0
    stressPeakRaw[1] = 0.0
    for i in range(bcount[None]):
        ti.atomic_max(stressPeakRaw[0], bstress[i])
        ti.atomic_max(stressPeakRaw[1], bstressC[i])
    for k in range(2):
        if stressPeakRaw[k] > stressPeak[k]:
            stressPeak[k] = stressPeakRaw[k]
        else:
            stressPeak[k] = stressPeak[k] * 0.97 + stressPeakRaw[k] * 0.03


@ti.kernel
def render_blocks(show: ti.i32):
    z = CAMZ[None]
    ox = CAMX[None]
    oy = CAMY[None]
    for i in range(bcount[None]):
        ri = bsz[i] * PX * z * 0.5
        px = int(bsz[i] * PX * z)
        X = int(bx[i] * PX * z + ox - ri)
        Y = H - 1 - int(by[i] * PX * z + oy + ri)
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
            # СРАВНЕНИЕ В ФИЗИЧЕСКИХ ЕДИНИЦАХ: bstress нормирован на Ftmax,
            # а bstressC — на Fcmax = rcFactor*Ftmax, поэтому сжатие,
            # приведённое к шкале Ftmax, равно cq*rcFactor. Раньше пороги были
            # 0.03 против 0.05 в РАЗНЫХ шкалах (сжатие требовало в 16.7 раз
            # большего напряжения) и растяжение всегда "побеждало": блок с
            # доминирующим сжатием подсвечивался красным, и картина изгиба
            # выглядела перевёрнутой (сверху растяжение / снизу сжатие).
            tphy = tq
            cphy = cq * P[None].rcFactor
            # �� �����񭭮� ������� (show=1) ����� �� ���訢����� �� 誠��:
            # 梥� 誠�� ����� ����� �痢� (render_bonds, scale=1)
            if show == 0 and (tphy > 0.03 or cphy > 0.03):
                if tphy >= cphy:
                    u = tq if tq > 1.0 else tq * tq * (3.0 - 2.0 * tq)
                    r = r + (255.0 - r) * u
                    g = g + (101.0 - g) * u
                    b = b + (58.0 - b) * u
                else:
                    u = cq if cq > 1.0 else cq * cq * (3.0 - 2.0 * cq)
                    r = r + (96.0 - r) * u
                    g = g + (152.0 - g) * u
                    b = b + (228.0 - b) * u
            col = ti.Vector([r / 255.0, g / 255.0, b / 255.0])
            if ti.abs(brot[i]) > 0.02:
                fill_rot_rect(bx[i] * PX * z + ox, H - 1 - (by[i] * PX * z + oy), brot[i], ri, col)
            else:
                fill_rect(X, Y, X + px - 1, Y + px - 1, col)
                hl = col * (1.0 - 0.07) + WHITE * 0.07
                fill_rect(X, Y + px - 2, X + px - 1, Y + px - 1, hl)


# =================================================== легенда напряжений (шрифт 3x5)
# ti.ui не умеет рисовать текст в канвасе — цифры шкал рисуем пикселями:
# глиф 3x5 (маска 15 бит), масштаб s пикселей на точку.

@ti.func
def digit_mask(d: ti.i32) -> ti.i32:
    """Маска глифа 3x5 (15 бит, старший бит — левый верхний пиксель).
    0..9 — цифры, 10 — точка, 11 — минус, 12..14 — буквы М, П, а (МПа)."""
    m = 0
    if d == 0:
        m = 0b111101101101111
    elif d == 1:
        m = 0b010110010010111
    elif d == 2:
        m = 0b111001111100111
    elif d == 3:
        m = 0b111001111001111
    elif d == 4:
        m = 0b101101111001001
    elif d == 5:
        m = 0b111100111001111
    elif d == 6:
        m = 0b111100111101111
    elif d == 7:
        m = 0b111001001001001
    elif d == 8:
        m = 0b111101111101111
    elif d == 9:
        m = 0b111101111001111
    elif d == 10:
        m = 0b000000000000010
    elif d == 11:
        m = 0b000001110000000
    elif d == 12:
        m = 0b101111111101101   # М
    elif d == 13:
        m = 0b111101101101101   # П
    else:
        m = 0b010101111101101   # а (А-образный)
    return m


@ti.func
def draw_digit(x: ti.i32, y: ti.i32, d: ti.i32, s: ti.i32, col: ti.template()):
    """Один глиф 3x5 масштаба s; y — НИЗ глифа. Канвас img на экране идёт
    снизу вверх (img[.,0] — низ экрана), поэтому строки глифа рисуются вверх
    от y — текст на экране читается, а не перевёрнут."""
    m = digit_mask(d)
    for py in range(5):
        for px in range(3):
            if ((m >> (14 - (py * 3 + px))) & 1) == 1:
                fill_rect(x + px * s, y + (4 - py) * s,
                          x + px * s + s - 1, y + (4 - py) * s + s - 1, col)


@ti.func
def draw_num(x: ti.i32, y: ti.i32, val: ti.f32, s: ti.i32, col: ti.template(), dec: ti.i32):
    """Число (МПа): целая часть + dec знаков после точки (0..2), y — низ цифр.
    Значение округляется до dec знаков, хвостовые нули при dec=1 не пишутся."""
    v = val
    x0 = x
    if v < 0.0:
        draw_digit(x0, y, 11, s, col)
        x0 += 4 * s
        v = -v
    scale = 1
    if dec == 1:
        scale = 10
    elif dec == 2:
        scale = 100
    num = int(v * float(scale) + 0.5)   # значение в долях 1/scale
    whole = num // scale
    frac = num - whole * scale
    if whole >= 100:
        draw_digit(x0, y, (whole // 100) % 10, s, col)
        x0 += 4 * s
    if whole >= 10:
        draw_digit(x0, y, (whole // 10) % 10, s, col)
        x0 += 4 * s
    draw_digit(x0, y, whole % 10, s, col)
    x0 += 4 * s
    if dec == 1 and frac != 0:
        draw_digit(x0, y, 10, s, col)
        x0 += 2 * s
        draw_digit(x0, y, frac, s, col)
    elif dec == 2:
        draw_digit(x0, y, 10, s, col)
        x0 += 2 * s
        draw_digit(x0, y, (frac // 10) % 10, s, col)
        x0 += 4 * s
        draw_digit(x0, y, frac % 10, s, col)


@ti.kernel
def render_stress_legend(rp: ti.f32, rc: ti.f32):
    """Динамическая легенда напряжений (левый верхний угол экрана): по 10
    цветных квадратов на шкалу — красная (растяжение) и синяя (сжатие),
    значения в МПа от 0 до ПИКА напряжений в сцене (stressPeak), поэтому
    числа меняются по мере роста/спада напряжений. Цвет квадрата —
    legend_band_color: ровно тот же, что у линий связей своего интервала
    (render_bonds). Каждый квадрат подписан верхней границей интервала.
    Канвас: y ВВЕРХ (img[.,0] — низ экрана) — вся геометрия снизу вверх,
    глифы рисуются от нижнего края, чтобы текст читался, а не был вверх ногами."""
    s = 3                      # масштаб шрифта (пикселей в точке)
    n = 10                     # квадратов (цветовых интервалов) на шкалу
    sw_w = 18                  # квадрат: ширина
    sw_h = 20                  # квадрат: высота
    stack_h = n * sw_h
    x_t = 26                   # шкала растяжения (красная)
    x_c = 168                  # шкала сжатия (синяя)
    y_base = H - 280           # низ стопки квадратов (y канваса вверх)
    # подложка: числа читаются на любом фоне сцены
    fill_rect(x_t - 10, y_base - 24, x_c + 116, y_base + stack_h + 30,
              ti.Vector([0.05, 0.06, 0.08]))
    for bi in range(2):
        xb = x_t if bi == 0 else x_c
        pk = stressPeak[0] if bi == 0 else stressPeak[1]
        vmax = rp if bi == 0 else rc
        if pk < 1e-4:
            pk = 1.0            # напряжений ещё нет — шкала 0..предел прочности
        vmax = vmax * pk
        # над шкалой — единицы измерения: МПа (глифы 12, 13, 14)
        wcol = ti.Vector([1.0, 1.0, 1.0])
        draw_digit(xb, y_base + stack_h + 6, 12, s, wcol)
        draw_digit(xb + 4 * s, y_base + stack_h + 6, 13, s, wcol)
        draw_digit(xb + 8 * s, y_base + stack_h + 6, 14, s, wcol)
        for k in range(n):
            col = legend_band_color(bi, k)
            y0 = y_base + k * sw_h
            fill_rect(xb, y0, xb + sw_w - 1, y0 + sw_h - 2, col)
            # подпись: верхняя граница интервала (МПа). Точность подбирается так,
            # чтобы значения не сливались и не «прыгали» после округления:
            # <10 МПа — два знака (0.45, 0.89...), 10..100 — один (4.5, 9, 13.5...),
            # >=100 — целые (15, 30...). Раньше всё >=10 округлялось до целых,
            # и шкала сжатия (часто 10..100 МПа) давала рваный ряд 5, 9, 14, 18...
            dec = 2
            if vmax >= 10.0:
                dec = 1
            if vmax >= 100.0:
                dec = 0
            draw_num(xb + sw_w + 6, y0 + (sw_h - 5 * s) // 2,
                     vmax * float(k + 1) / float(n), s, wcol, dec)
        # ноль — под стопкой квадратов
        draw_num(xb + sw_w + 6, y_base - 17, 0.0, s, wcol, 0)


@ti.kernel
def render_bonds(show: ti.i32, scale: ti.i32):
    """Линии напряжений по связям (внутри блоков): растяжение — красная шкала,
    сжатие — синяя. При scale=1 (легенда включена) цвет линии — цвет интервала
    legend_band_color её напряжения, ровно как квадрат на шкале легенды;
    иначе обычные оранжевый/синий."""
    if show == 1:
        z = CAMZ[None]
        ox = CAMX[None]
        oy = CAMY[None]
        for bd in range(bondCount[None]):
            if bondIntact[bd] == 1:
                r = bondR[bd]
                a = bondA[bd]
                b = bondB[bd]
                tphy = r
                cphy = -r * 0.25 * P[None].rcFactor if r < 0.0 else 0.0
                col = ORANGE
                if tphy > 0.12 or cphy > 0.12:
                    if tphy >= cphy:
                        col = ORANGE
                        if scale == 1:
                            col = legend_band_color(0, stress_band(r, stressPeak[0]))
                    else:
                        col = BLUE
                        if scale == 1:
                            col = legend_band_color(1, stress_band(-r * 0.25, stressPeak[1]))
                    draw_line(bx[a] * PX * z + ox, H - 1 - (by[a] * PX * z + oy),
                              bx[b] * PX * z + ox, H - 1 - (by[b] * PX * z + oy), col, 2.0)


@ti.kernel
def render_joints():
    # швы между связанными блоками (линия по общему ребру, поперёк связи)
    z = CAMZ[None]
    ox = CAMX[None]
    oy = CAMY[None]
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
        # нормаль связи В БУФЕРЕ (мир y вниз, буфер Y = H-1-y вверх):
        # связь в буфере (ux, -uy), её перпендикуляр (uy, ux). Раньше перпендикуляр
        # брался в мировых осях (-uy, ux) и рисовался без переворота Y — засечка
        # зеркалилась и на повёрнутой модели стояла поперёк связи с ошибкой 2*theta
        # (ровно как квадраты блоков до фикса fill_rot_rect).
        px = uy * w * PX * z
        py = ux * w * PX * z
        cx0 = mx * PX * z + ox
        cy0 = H - 1 - (my * PX * z + oy)
        draw_line(cx0 - px, cy0 - py, cx0 + px, cy0 + py, JOINT, 1.0)
        # пластически текущая связь — фиолетовым ПО ГРАНИ (не вместо напряжений):
        # усилия (оранжевый/синий) рисуются в render_bonds, здесь только шов-маркер
        if bondFlow[bd] == 1:
            draw_line(cx0 - px, cy0 - py, cx0 + px, cy0 + py, PURPLE, 3.0)


@ti.kernel
def render_cracks():
    z = CAMZ[None]
    ox = CAMX[None]
    oy = CAMY[None]
    for bd in range(bondCount[None]):
        if bondIntact[bd] == 1:
            continue
        a = bondA[bd]
        b = bondB[bd]
        dx = bx[b] - bx[a]
        dy = by[b] - by[a]
        dd = dx * dx + dy * dy
        if dd > 1.7 * LCELL * LCELL:
            continue
        d = ti.sqrt(dd)
        mx = (bx[a] + bx[b]) * 0.5 * PX * z + ox
        my = H - 1 - ((by[a] + by[b]) * 0.5 * PX * z + oy)
        j = bondJ[bd] * PX * z
        hl = ti.min(bsz[a], bsz[b]) * PX * z * 0.55
        # засечка трещины ПОПЕРЁК связи, полюс скольжения j ВДОЛЬ связи;
        # оси буфера (мир y вниз -> Y вверх): вдоль связи (ux, -uy), поперёк (uy, ux).
        # Раньше засечка была жёстко вертикаль/горизонталь по мировым осям
        # (ветвление |dx|>=|dy|) — на повёрнутой модели трещины не лежали поперёк
        # шва, а «провернуты» так же, как засечки связей.
        ux = dx / d
        uy = dy / d
        tx = ux
        ty = -uy
        nx = uy
        ny = ux
        draw_line(mx + 0.4 * j * tx - hl * nx, my + 0.4 * j * ty - hl * ny,
                  mx - 0.4 * j * tx + hl * nx, my - 0.4 * j * ty + hl * ny, CRACK, 2.0)


@ti.kernel
def render_water():
    z = CAMZ[None]
    ox = CAMX[None]
    oy = CAMY[None]
    s = RP_SPH * 2.0 * PX * z * 0.95
    core = CYAN
    for i in range(pCount[None]):
        cx = wx[i] * PX * z + ox - s * 0.5
        cy = H - 1 - (wy[i] * PX * z + oy) - s * 0.5
        fill_rect(int(cx), int(cy), int(cx + s), int(cy + s), core)


@ti.kernel
def render_dust():
    z = CAMZ[None]
    ox = CAMX[None]
    oy = CAMY[None]
    for i in range(dustCount[None]):
        if i >= MAXDUST:
            continue
        a = dustLife[i] * 0.5
        c = ti.Vector([0.745, 0.784, 0.843])
        if dustCol[i] == 1:
            c = ti.Vector([0.471, 0.667, 0.784])
        cc = BG * (1.0 - a) + c * a
        cx = int(dustX[i] * PX * z + ox) - 2
        cy = H - 1 - int(dustY[i] * PX * z + oy) + 2
        fill_rect(cx, cy, cx + 3, cy + 3, cc)


@ti.kernel
def render_sources(t: ti.f32):
    z = CAMZ[None]
    ox = CAMX[None]
    oy = CAMY[None]
    for s in range(srcCount[None]):
        X = srcX[s] * LCELL * PX * z + ox
        Y = H - 1 - (srcY[s] * LCELL * PX * z + oy)
        pulse = (t * 1.1 + srcPh[s]) % 1.0
        r = (0.35 + pulse * 0.95) * CELL * z
        draw_circle_outline(X, Y, r, CYAN * (0.55 * (1.0 - pulse)) + BG * (1.0 - 0.55 * (1.0 - pulse)), 2.0)
        fill_diamond(X, Y, 5.0 * z, CYAN)


@ti.kernel
def render_mouse(mx: ti.f32, my: ti.f32, brush: ti.f32, is_water: ti.i32):
    if mx > -900.0:
        z = CAMZ[None]
        X = mx * CELL * z + CAMX[None]
        Y = H - 1 - (my * CELL * z + CAMY[None])
        if is_water == 1:
            draw_circle_outline(X, Y, 8.0 * z, CYAN * 0.7, 1.5)
        else:
            r = (brush / 2.0 + 0.3) * CELL * z
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
                add_bond_func(a, r_block[c + 1], 1.0, 1 if cutR[c] == 0 else 0)
            if (c // COLS) < ROWS - 1 and r_block[c + COLS] >= 0:
                add_bond_func(a, r_block[c + COLS], 1.0, 1 if cutD[c] == 0 else 0)
        c += 1


# ================================================================== наклонные слои
# НаклонНАЯ решётка: u — вдоль слоя, v — поперёк. Мировые координаты центра блока:
#   x = COLS/2 + u·cosθ − v·sinθ,  y = ROWS/2 + u·sinθ + v·cosθ   (клетки, y вниз)
# Слои и ослабленные плоскости идут ПОД УГЛОМ θ к горизонтали, блоки повёрнуты
# (brot=θ) — OBB-контакт это поддерживает. Гравитация и давление вышележащих
# пород (ovf/sigVI решателя) остаются ВЕРТИКАЛЬНЫМИ, как в реальном массиве:
# в наклонных слоях вес столба даёт вертикальное σ_v, а связи работают на
# сдвиг/отрыв вдоль наклонных швов — распределение сил другое, и оно возникает
# из геометрии честно, без поправки "вручную".
LATN = 160          # сторона решётки (u,v ∈ [−80, 80) — с запасом на диагональ)
LATH = LATN // 2
latB = ti.field(ti.i32, LATN * LATN)


@ti.kernel
def gen_tilted(weak: ti.i32, ct: ti.f32, st: ti.f32, cav: ti.i32):
    """Массив с НАКЛОННЫМИ слоями (θ = atan2(st, ct)), опция: полость выработки.

    weak=1 — две ослабленные плоскости ПО СЛОЯМ (v = WV1/WV2: связи поперёк слоя
    в этих полосах создаются сразу порванными, intact=0 — наклонные швы, вдоль
    которых массив может съезжать). cav=1 — эллиптическая выработка (мировые
    координаты, как в gen_tunnel: горизонтальная выработка в наклонных слоях)."""
    for c in range(N):
        r_filled[c] = 0
        r_block[c] = -1
        cutR[c] = 0
        cutD[c] = 0
    for c in range(LATN * LATN):
        latB[c] = -1
    tx = COLS * 0.5
    ty = ROWS * 0.6
    trx = 7.0
    try_ = 5.0
    cavX[None] = tx
    cavY[None] = ty
    cavRX[None] = trx
    cavRY[None] = try_
    tilt = ti.atan2(st, ct)
    ox0 = COLS * 0.5
    oy0 = ROWS * 0.5
    WV1 = -6        # ослабленная плоскость 1 (номер полосы v, связь v→v+1 рвётся)
    WV2 = 7         # ослабленная плоскость 2
    v = -LATH + 2
    while v < LATH - 2:
        u = -LATH + 2
        while u < LATH - 2:
            x = ox0 + u * ct - v * st
            y = oy0 + u * st + v * ct
            ok = 1
            if x < 0.75 or x > COLS - 0.75 or y < 0.75 or y > ROWS - 0.5:
                ok = 0
            if ok == 1 and cav == 1:
                dx = (x - tx) / trx
                dy = (y - ty) / try_
                if dx * dx + dy * dy < 1.08:
                    ok = 0
            if ok == 1:
                fixed = 1 if (y >= ROWS - 2.0 or y >= ROWS - P[None].fixedRows) else 0
                i = add_block_func(x, y, 1.0, fixed)
                if i >= 0:
                    brot[i] = tilt      # блок повёрнут вдоль слоя
                    latB[(u + LATH) * LATN + (v + LATH)] = i
                    cx2 = int(x)
                    cy2 = int(y)
                    if cx2 >= 0 and cx2 < COLS and cy2 >= 0 and cy2 < ROWS:
                        c2 = cy2 * COLS + cx2
                        if r_block[c2] < 0:
                            r_block[c2] = i
                            r_filled[c2] = 1
            u += 1
        v += 1
    # связи вдоль (u+1) и поперёк (v+1) слоя; в ослабленных полосах — рваные.
    # Индексные переменные vv/uu — НЕ v/u: taichi запрещает переиспользовать
    # имена уже объявленных в ядре переменных как переменные циклов.
    for vv in range(-LATH + 2, LATH - 3):
        for uu in range(-LATH + 2, LATH - 3):
            a = latB[(uu + LATH) * LATN + (vv + LATH)]
            if a >= 0:
                b = latB[(uu + 1 + LATH) * LATN + (vv + LATH)]
                if b >= 0:
                    add_bond_func(a, b, 1.0, 1)
                b = latB[(uu + LATH) * LATN + (vv + 1 + LATH)]
                if b >= 0:
                    intact = 1
                    if weak == 1 and (vv == WV1 or vv == WV2):
                        intact = 0
                    add_bond_func(a, b, 1.0, intact)


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
                # длина покоя — ИСТИННОЕ расстояние между центрами (в клетках,
                # add_bond_func переведёт в метры): max-ось занижала его у пар
                # «большой+малый» блок (смещение центров диагональное), связь
                # создавалась НАТЯНУТОЙ и рвалась лишний раз.
                rest = math.sqrt(dx * dx + dy * dy)
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


def generate(mode, weak, tilt_deg=0.0):
    """Построить модель сцены и передать её в решатель.

    mode 0 — выработка (единичные блоки), mode 1 — склон, mode 2 — мозаика
    произвольных размеров (build_model). weak — ослабленные плоскости.
    tilt_deg — НАКЛОН СЛОЁВ к горизонтали (градусы, + = падение вправо):
    применяется к сцене «выработка» и к «Заполнить массив»; гравитация и
    давление вышележащих пород остаются вертикальными (см. gen_tilted)."""
    global model_angle
    model_angle = 0.0      # новая сцена — ориентация "как сгенерирована"
    if mode == 2:
        build_bigblocks()
    else:
        reset_all()
        refillOff[None] = 0   # новая сцена — досыпка сверху снова включена
        if mode == 0:
            if abs(tilt_deg) > 0.5:
                t = math.radians(tilt_deg)
                gen_tilted(weak, math.cos(t), math.sin(t), 1)
            else:
                gen_tunnel(weak)
        else:
            gen_slope(weak)
    # Престресс горизонтальных связей: σ_h = K0·σ_v заложен в связи сразу после сборки
    # (см. update_prestress). Перегруз loadCur набирается ПЛАВНО в compute_top_load_fused
    # (LOAD_RAMP) — мгновенный полный вес даёт ударный выброс по связям; update_prestress
    # зовётся КАЖДЫЙ тик в step_physics и динамически следует за глубиной/K0 и рампой.
    loadCur[None] = 0.0
    refresh_field()
    update_prestress(DT)


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
    # x, y, r — в клетках; позиции блоков bx/by — в метрах
    i = bcount[None] - 1
    while i >= 0:
        if bfixed[i] == 0:
            dx = bx[i] - x * LCELL
            dy = by[i] - y * LCELL
            if dx * dx + dy * dy < r * r * LCELL * LCELL:
                remove_block_func(i)
        i -= 1


@ti.kernel
def crack_at(x: ti.f32, y: ti.f32, r: ti.f32):
    # x, y, r — в клетках; позиции блоков bx/by — в метрах
    r2 = r * r * LCELL * LCELL
    for bd in range(bondCount[None]):
        if bondIntact[bd] == 1:
            mx = (bx[bondA[bd]] + bx[bondB[bd]]) * 0.5 - x * LCELL
            my = (by[bondA[bd]] + by[bondB[bd]]) * 0.5 - y * LCELL
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
            # rest — в КЛЕТКАХ (add_bond_func умножает на LCELL): метры / LCELL
            if axx > rs - 0.05 * LCELL and axx < rs + 0.05 * LCELL and ayy < ti.min(ri, rj):
                add_bond_func(i, j, ti.sqrt(dx * dx + dy * dy) / LCELL, 1)
            elif ayy > rs - 0.05 * LCELL and ayy < rs + 0.05 * LCELL and axx < ti.min(ri, rj):
                add_bond_func(i, j, ti.sqrt(dx * dx + dy * dy) / LCELL, 1)
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
        dustVY[i] += 0.004 * LCELL
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
# курсор окна (нормализованные координаты) -> координаты КЛЕТОК модели
def cursor_to_cells(pos):
    k = CELL * _CAM[2]
    sx = pos[0] * W
    sy = pos[1] * H
    mx = (sx - _CAM[0]) / k
    my = (H - 1.5 - sy - _CAM[1]) / k
    return mx, my


# зум мышью: масштабируется к точке под курсором, чтобы она осталась на месте
def zoom_at(pos, factor):
    z0 = _CAM[2]
    z1 = min(CAM_ZMAX, max(CAM_ZMIN, z0 * factor))
    if z1 == z0:
        return
    sx = pos[0] * W
    sy = pos[1] * H
    wx = (sx - _CAM[0]) / (PX * z0)
    wy = (H - 1.5 - sy - _CAM[1]) / (PX * z0)
    k = PX * z1
    cam_write(sx - wx * k, H - 1.5 - sy - wy * k, z1)


def reset_camera():
    cam_write(0.0, 0.0, 1.0)


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
    print("selftest: vnutrennee vremya simulyatsii = %.3f s" % simT[None])
    ek, er, ep, ee, et = energy_report()
    cex, cey, cem = center_of_mass_report()
    print("selftest: energiya Dzh: kin=%.3e rot=%.3e pot=%.3e elast=%.3e total=%.3e" % (ek, er, ep, ee, et))
    print("selftest: centr tyazhesti: x=%.3f m, y=%.3f m, massa=%.3f kg" % (cex, cey, cem))
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
    spawn_water(1200, 1.8 * LCELL, COLS * 0.5, ROWS * 0.3, PI * 0.5)
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
    stress_peaks()
    render_blocks(1)
    render_joints()
    render_bonds(1, 1)
    render_cracks()
    render_water()
    render_dust()
    render_sources(0.0)
    render_mouse(-999.0, 0.0, 2.0, 0)
    hud_gather()
    render_stress_legend(hudF[HUD_RP], hudF[HUD_RP] * hudF[HUD_RCFACTOR])
    t_render = time.perf_counter() - t0
    print("selftest: one render pass in %.1f ms" % (t_render * 1000.0))
    print("selftest OK")


def run_gui():
    global model_angle
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
    tilt_deg = 0        # наклон слоёв к горизонтали, градусы (+ = падение вправо)
    show_bonds = 1
    E_GPa = 10
    rho = RHO_R
    scene_names = {0: "vyrabotka", 1: "sklon", 2: "krupnye bloki"}

    # без vsync: окно не ограничивает цикл 60 Гц, тики идут так быстро,
    # как позволяет CPU (см. step_physics: следующий тик — после полного расчёта)
    window = ti.ui.Window("ГЕОМАССИВ/2D · DEM", (W, H), vsync=False)
    canvas = window.get_canvas()
    gui = window.get_gui()

    generate(0, 0)

    last = time.perf_counter()
    fps = 60.0
    hud_t = 0.0
    last_mx = -1.0
    last_my = -1.0
    rain_until = -1.0          # simT (с), до которого идёт ливень; -1 = ливня нет
    flash_msg = "Nazhmite СТАРТ - simulyatsiya nachnetsya"
    flash_wall = 0.0           # реальное время (time.perf_counter) для всплывающих сообщений
    last_broken = 0
    last_hud_broken = 0
    cam_drag = False          # зажата средняя кнопка — таскаем камеру
    cam_prev = (0.0, 0.0)     # курсор прошлого кадра (для дельты драга)
    seen_keys = set()         # диагностика неизвестных событий ввода (в консоль)
    ang_focus = 1             # какое поле угла принимает цифры: 0 = слои, 1 = модель (Tab)
    buf_sloi = ""             # буфер набора угла слоёв, град
    buf_model = ""            # буфер набора угла модели, град
    model_angle = 0.0         # текущий угол модели к горизонту, град (0 = как сгенерирована)
    legend_on = False         # легенда напряжений (кнопка под START)

    while window.running:
        now = time.perf_counter()
        dt_real = min(0.05, now - last)
        last = now
        # снимок HUD одним ядром + одним копированием на кадр (вместо двух
        # десятков read'ов скаляров — каждый read был синхронизацией CPU<->GPU)
        hud = hud_snapshot()
        t_sim = hud[HUD_SIMT]     # внутреннее время симуляции (сек, дробное)

        try:
            # один проход по ВСЕМ событиям (колесо — не типа PRESS, отдельный
            # фильтр его теряет). Отпускание кнопки/клавиши отличаем от нажатия
            # через window.is_pressed(k): у события отпуска is_pressed == False.
            # символы числа (цифры/минус/точка) — опросом клавиатуры: события
            # ti.ui их не приносят (см. poll_type_chars)
            for ch in poll_type_chars():
                if ang_focus == 0:
                    if len(buf_sloi) < 8:
                        buf_sloi += ch
                else:
                    if len(buf_model) < 8:
                        buf_model += ch
            for e in window.get_events():
                k = e.key
                if k == "Wheel" or k == "WHEEL":
                    # колесо — зум к точке под курсором (delta[1] > 0 = к нам)
                    dy = 0.0
                    try:
                        d = e.delta
                        dy = d[1] if isinstance(d, (tuple, list)) else float(d)
                    except Exception:
                        dy = 0.0
                    if dy != 0.0:
                        pos = window.get_cursor_pos()
                        f = 1.15 if dy > 0 else 1.0 / 1.15
                        zoom_at(pos, f)
                    continue
                if k == ti.ui.TAB:
                    # Tab — какое поле угла получает цифры (сами символы приходят
                    # из poll_type_chars: события ti.ui цифры/минус/точку не дают)
                    ang_focus = 1 - ang_focus
                    continue
                if k == ti.ui.RETURN:
                    if ang_focus == 0:
                        tv = parse_deg(buf_sloi)
                        if tv is None or tv < -45.0 or tv > 45.0:
                            flash_msg = "Ugol sloev: nuzhno chislo ot -45 do 45"
                        else:
                            tilt_deg = tv
                            srcCount[None] = 0
                            generate(mode, weak, tilt_deg)
                            model_angle = 0.0
                            flash_msg = ("Naklon sloev %g grad: sloi pod uglom, "
                                         "g i nagruzka sverhu - vertikalno" % tilt_deg)
                            buf_sloi = ""
                        flash_wall = now
                    else:
                        tv = parse_deg(buf_model)
                        if tv is None or tv < -180.0 or tv > 180.0:
                            flash_msg = "Ugol modeli: nuzhno chislo ot -180 do 180"
                        else:
                            d = math.radians(tv - model_angle)
                            if abs(d) > 1e-9:
                                rotate_model(d, COLS * LCELL * 0.5, ROWS * LCELL * 0.5)
                                model_angle = tv
                                flash_msg = "Model povernuta na %g grad (g vertikalno)" % tv
                            buf_model = ""
                        flash_wall = now
                    continue
                if k == ti.ui.ESCAPE:
                    if buf_sloi or buf_model:
                        buf_sloi = ""
                        buf_model = ""
                        flash_msg = "Vvod ugla otmenyon"
                        flash_wall = now
                    else:
                        window.running = False
                    continue
                if k == ti.ui.BACKSPACE:
                    if ang_focus == 0:
                        buf_sloi = buf_sloi[:-1]
                    else:
                        buf_model = buf_model[:-1]
                    continue
                if k not in (ti.ui.SPACE, "c", ti.ui.LMB, ti.ui.RMB, ti.ui.MMB):
                    if k not in seen_keys:
                        seen_keys.add(k)
                        print("input: neizvestnoye sobytiye key=%r delta=%r" % (k, getattr(e, "delta", None)))
                    continue
                if not window.is_pressed(k):
                    continue
                if k == ti.ui.SPACE:
                    paused = not paused
                elif k == "c":
                    reset_camera()
                    flash_msg = "Kamera sbroshena"
                    flash_wall = now
                elif k == ti.ui.LMB:
                    pos = window.get_cursor_pos()
                    mx, my = cursor_to_cells(pos)
                    if pos[0] > 0.71:
                        continue
                    if tool == 2:
                        flash_msg = place_source(mx, my)
                        flash_wall = now
                    else:
                        last_mx = mx
                        last_my = my
                        last_mx, last_my = do_stroke(mx, my, last_mx, last_my, tool, brush)
                elif k == ti.ui.RMB:
                    pos = window.get_cursor_pos()
                    mx, my = cursor_to_cells(pos)
                    if remove_source(mx, my):
                        flash_msg = "Istochnik ubran"
                        flash_wall = now
                elif k == ti.ui.MMB:
                    # зажатие СКМ — начать перетаскивание камеры
                    pos = window.get_cursor_pos()
                    if pos[0] <= 0.71:
                        cam_drag = True
                        cam_prev = pos
        except Exception:
            import traceback
            traceback.print_exc()
            flash_msg = "Oshibka vvoda - smotrite konsol"
            flash_wall = now

        if window.is_pressed(ti.ui.LMB):
            pos = window.get_cursor_pos()
            if pos[0] <= 0.71:
                mx, my = cursor_to_cells(pos)
                if tool != 2 and last_mx >= 0:
                    last_mx, last_my = do_stroke(mx, my, last_mx, last_my, tool, brush)
                if tool == 2 and last_mx < 0:
                    pass

        # перетаскивание камеры СКМ: двигаем за курсором
        if cam_drag and window.is_pressed(ti.ui.MMB):
            pos = window.get_cursor_pos()
            if pos[0] <= 0.71:
                cam_write(_CAM[0] + pos[0] * W - cam_prev[0] * W,
                          _CAM[1] - pos[1] * H - cam_prev[1] * H,
                          _CAM[2])
                cam_prev = pos
        else:
            cam_drag = False

        # ---- GUI
        gui.begin("Upravlenie", 0.72, 0.02, 0.27, 0.96)
        # кнопка старт/стоп: символы ▶/❚❚ не рисуются шрифтом, используем ASCII.
        start_lbl = "> START" if paused else "|| СТОП"
        if gui.button(start_lbl):
            paused = not paused
            flash_msg = "Симуляция запущена" if not paused else "Пауза"
            flash_wall = now
        # под START — переключатель легенды напряжений
        if gui.button("Skryt legendu" if legend_on else "Pokazat legendu"):
            legend_on = not legend_on
        if legend_on:
            pkT = hud[HUD_PEAKT] if hud[HUD_PEAKT] > 1e-4 else 1.0
            pkC = hud[HUD_PEAKC] if hud[HUD_PEAKC] > 1e-4 else 1.0
            gui.text("Legenda: 10 intervalov cveta, 0..pik napryazheniy")
            gui.text("Krasnaya shkala - rastyazhenie: 0..%.1f MPa" % (hud[HUD_RP] * pkT))
            gui.text("Sinyaya shkala - szhatie: 0..%.1f MPa" % (hud[HUD_RP] * hud[HUD_RCFACTOR] * pkC))
        gui.text("Vremya simulyatsii: %.3f s" % t_sim)
        gui.text("Scena: %s" % scene_names[mode])
        gui.text("Naklon sloev: %g grad (g - vertikalno)" % tilt_deg)
        gui.text("Poroda: E=%d GPa, Rt=%d MPa, Rc=%d MPa" % (E_GPa, int(hud[HUD_RP]), int(hud[HUD_RP] * hud[HUD_RCFACTOR])))
        gui.text("Napor: %.2f MPa na porodu" % (RHO_W * G_PHYS * hud[HUD_HEAD] / 1e6))
        gui.text("LKM - instrument * PKM - istochnik")
        gui.text("SKM - peretaskivanie * Koleso - zum")
        if gui.button("Poroda"):
            tool = 0
        if gui.button("Treshchina"):
            tool = 1
        if gui.button("Istochnik vody"):
            tool = 2
        if gui.button("Lastik"):
            tool = 3
        brush = gui.slider_int("Kist", brush, 1, 6)
        E_new = gui.slider_int("Modul E, GPa", E_GPa, 1, 20)
        rp_new = gui.slider_int("Rp (otriv) MPa", int(hud[HUD_RP]), 1, 15)
        mu_new = gui.slider_int("Trenie mu %%", int(hud[HUD_MU] * 100), 20, 90)
        if E_new != E_GPa or rp_new != int(hud[HUD_RP]) or mu_new != int(hud[HUD_MU] * 100):
            E_GPa = E_new
            apply_rock(E_GPa, float(rp_new), mu_new / 100.0, rho)
        depth_new = gui.slider_int("Poroda nad skhemoy, m", int(hud[HUD_DEPTH]), 0, 3000)
        if depth_new != int(hud[HUD_DEPTH]):
            P[None].depth = depth_new
        rho_new = gui.slider_int("Plotnost porody, kg/m3", int(rho), 1000, 3500)
        if rho_new != rho:
            rho = float(rho_new)
            apply_rock(E_GPa, float(hud[HUD_RP]), hud[HUD_MU], rho)
            apply_water(hud[HUD_HEAD])
        k0_new = gui.slider_float("K0 bokovogo davleniya", hud[HUD_K0], 0.0, 2.0)
        if k0_new != hud[HUD_K0]:
            P[None].k0 = k0_new
        fr_new = gui.slider_int("Zakrepl. ryadov", int(hud[HUD_FIXEDROWS]), 1, 4)
        if fr_new != int(hud[HUD_FIXEDROWS]):
            P[None].fixedRows = fr_new
        bc_new = gui.slider_int("Limit razryvov/tik", int(hud[HUD_BRKCAP]), 1, 200)
        if bc_new != int(hud[HUD_BRKCAP]):
            P[None].brkCap = bc_new
        dm_new = gui.slider_int("Tr. povrezhdeniya, tik", int(hud[HUD_DMGMAX]), 1, 60)
        if dm_new != int(hud[HUD_DMGMAX]):
            P[None].dmgMax = dm_new
        pf_new = gui.slider_float("Skor. plast. techeniya, 1/s", hud[HUD_PLASTFLOW], 0.0, 1.0)
        if pf_new != hud[HUD_PLASTFLOW]:
            P[None].plastFlow = pf_new
        walls = gui.checkbox("Bokovye stenki", hud[HUD_WALLS] == 1)
        if walls != (hud[HUD_WALLS] == 1):
            P[None].walls = 1 if walls else 0
        weak_new = gui.checkbox("Oslablennye ploskosti", weak == 1)
        if weak_new != weak:
            weak = 1 if weak_new else 0
            srcCount[None] = 0
            generate(mode, weak, tilt_deg)
            flash_msg = "Massiv %s: svyazey %d" % ("oslablen" if weak else "monolitny", count_intact())
            flash_wall = now
        # угол модели — поворот СОБРАННОЙ модели; угол слоёв — перегенерация
        # с наклонными слоями; в обоих случаях сила тяжести остаётся вертикальной.
        # угол — просто СТРОКИ набора числа (кнопка не нужна): цифры/знак/точка
        # пишутся в активное поле сразу (символы читает poll_type_chars — события
        # ti.ui цифры не приносят), Tab — переключить поле, Enter — применить,
        # BackSpace — удалить, Esc — очистить буферы.
        gui.text("Ugol modeli k horizontu, grad (seychas %g):" % model_angle)
        gui.text("  [ %s ]%s   Tab - pole, Enter - primenit" % (buf_model, "_" if ang_focus == 1 else ""))
        gui.text("Ugol sloev k horizontu, grad (seychas %g):" % tilt_deg)
        gui.text("  [ %s ]%s   peresozdaet scenu, g vertikalno" % (buf_sloi, "_" if ang_focus == 0 else ""))
        gv = gui.slider_int("Gravitatsiya x0.1g", int(hud[HUD_G] / G_PHYS * 10.0), 0, 20)
        if gv != int(hud[HUD_G] / G_PHYS * 10.0):
            apply_gravity(gv / 10.0)
        if gui.button("Vyrabotka"):
            if mode != 0:
                mode = 0
                srcCount[None] = 0
                generate(mode, weak, tilt_deg)
        if gui.button("Sklon"):
            if mode != 1:
                mode = 1
                srcCount[None] = 0
                generate(mode, weak, tilt_deg)
        if gui.button("Krupnye bloki"):
            if mode != 2:
                mode = 2
                srcCount[None] = 0
                refillOff[None] = 0
                generate(mode, weak, tilt_deg)
                flash_msg = "Mozaika: blokov %d, max razmer %d" % (bcount[None], P[None].maxSize)
                flash_wall = now
        if gui.button("Zapolnit massiv"):
            srcCount[None] = 0
            reset_all()
            if abs(tilt_deg) > 0.5:
                t = math.radians(tilt_deg)
                gen_tilted(weak, math.cos(t), math.sin(t), 0)
            else:
                gen_solid()
            flash_msg = "Massiv zapolnen: blokov %d" % bcount[None]
            flash_wall = now
        if gui.button("Peregenerirovat"):
            srcCount[None] = 0
            generate(mode, weak, tilt_deg)
            flash_msg = "Massiv: svyazey %d" % count_intact()
            flash_wall = now
        if gui.button("Dozhd"):
            rain_until = t_sim + 2.5
            flash_msg = "Liven - voda ishchet treshchiny"
            flash_wall = now
        if gui.button("Ochistit porodu"):
            clear_scene()
            model_angle = 0.0
            refillOff[None] = 1   # запретить досыпку обломков сверху
            flash_msg = "Poroda ubrana"
            flash_wall = now
        show_bonds = 1 if gui.checkbox("Pokazyvat svyazi", show_bonds == 1) else 0
        head_new = gui.slider_int("Napor H, m", int(hud[HUD_HEAD]), 1, 40)
        if head_new != int(hud[HUD_HEAD]):
            apply_water(float(head_new))
        if gui.button("Slit vodu"):
            pCount[None] = 0
            filledFlag[None] = 0
            flash_msg = "Voda slita"
            flash_wall = now
        if gui.button("Ubrat istochniki"):
            srcCount[None] = 0
            flash_msg = "Istochniki ubrany"
            flash_wall = now
        if gui.button("Pauza / Pusk (Space)"):
            paused = not paused
            if not paused:
                flash_msg = "Симуляция запущена"
            flash_wall = now
        fp = hud[HUD_FILL]
        gui.text("fps %.0f * blokov %d" % (fps, int(hud[HUD_BCOUNT])))
        gui.text("chastits h2o %d%s * svyazey %d" % (int(hud[HUD_PCOUNT]), " (MAX)" if hud[HUD_PCOUNT] >= PMAX else "", int(hud[HUD_INTACT])))
        gui.text("razryvov %d * zapolnenie %.0f%%" % (int(hud[HUD_BROKEN]), fp))
        gui.text("Energiya: E=%.1e J (K=%.1e, P=%.1e)" % (hud[HUD_ETOT], hud[HUD_EKIN], hud[HUD_EPOT]))
        gui.text("Centr tyazhesti: x=%.2f, y=%.2f m, massa=%.0f kg" % (hud[HUD_COMX], hud[HUD_COMY], hud[HUD_COMASS]))
        if flash_msg and (now - flash_wall < 2.0):
            gui.text(flash_msg)
        gui.end()

        # ---- симуляция
        # Один тик за итерацию окна (без vsync цикл крутится на полной скорости
        # CPU). Каждый тик = фиксированный шаг внутреннего времени TICK; следующий
        # тик начинается только после полного расчёта предыдущего (локальная
        # физика решателя, см. step_physics).
        if not paused:
            if rain_until >= 0.0 and t_sim < rain_until:
                rx = 1.0 + (math.sin(t_sim * 13.7) * 0.5 + 0.5) * (COLS - 2)
                spawn_water(2, 1.8 * LCELL, rx, 1.2, PI * 0.5)
            step_physics(mode)
            t_sim += TICK
            hud[HUD_SIMT] = t_sim
            dust_step()
            dust_compact()
            if int(hud[HUD_BROKEN]) != last_broken:
                if int(hud[HUD_BROKEN]) % 8 == 1:
                    flash_msg = "Smyatie porody (Rc)" if hud[HUD_CRUSH] == 1 else "Razryv svyazi (Rp)"
                    flash_wall = now
                last_broken = int(hud[HUD_BROKEN])
            if hud[HUD_FILLFLAG] == 0 and fp >= 95:
                filledFlag[None] = 1
                flash_msg = "Vyrabotka zapolnena vodoy"
                flash_wall = now
        else:
            if last_broken != int(hud[HUD_BROKEN]):
                last_broken = int(hud[HUD_BROKEN])

        # ---- отрисовка
        render_base()
        render_triangles(t_sim, 1 if (mode == 0 and hud[HUD_DEPTH] > 0) else 0)
        stress_peaks()   # пики напряжений — динамическая шкала легенды
        render_blocks(1 if legend_on else 0)
        render_joints()
        render_bonds(show_bonds, 1 if legend_on else 0)
        render_cracks()
        render_water()
        render_dust()
        render_sources(t_sim)

        pos = window.get_cursor_pos()
        if pos[0] <= 0.71:
            mx, my = cursor_to_cells(pos)
            render_mouse(mx, my, brush, 1 if tool == 2 else 0)
        else:
            render_mouse(-999.0, 0.0, brush, 0)

        if legend_on:
            render_stress_legend(hud[HUD_RP], hud[HUD_RP] * hud[HUD_RCFACTOR])
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
