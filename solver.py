"""ГЕОМАССИВ/2D · DEM — решатель (физический движок)

Чистый физический движок: дискретно-элементный метод (DEM) для породы
+ Position-Based Dynamics (SPH) для флюида. Не знает про сцены, инструменты
и GUI: модель (список блоков/связей/воды/полости) строится СНАРУЖИ
(render.py) и загружается через build_model() либо через примитивы
add_block_func / add_bond_func / spawn_particle.

Блоки — квадраты ПРОИЗВОЛЬНОГО размера в произвольной конфигурации:
у каждого блока свой размер bsz[i], у каждой связи своя длина покоя
brest[bd] и начальное направление bondNX/bondNY. Сетка нужна только как
пространственный хэш (hhead) и для хранения полей фона. Физика, генерация
и инструменты — в render.py, точка входа — mineudec.py.

DEM: клетка = 1 м, блок m = 1 кг (эффективная масса). Силы: Ft = Rp[МПа]·400.
Флюид: Position-Based Dynamics (SPH-плотность), контакт с породой герметичен —
вода идёт только в раскрытые трещины.
"""

import sys
import os
import pathlib
import tempfile
import shutil
import importlib.util
import argparse
import time
import math

import numpy as np


def _is_ascii_path(p):
    try:
        p.encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def _short_ascii_path(p):
    """Возвращает короткий (8.3) ASCII-путь, если исходный содержит не-ASCII."""
    if _is_ascii_path(p):
        return p
    try:
        import ctypes
        buf = ctypes.create_unicode_buffer(1024)
        n = ctypes.windll.kernel32.GetShortPathNameW(p, buf, 1024)
        if n and _is_ascii_path(buf.value):
            return buf.value
    except Exception:
        pass
    return None


def _bootstrap_taichi():
    """GGUI (Vulkan) не может открыть свои shaders по пути с не-ASCII (кириллица в
    имени пользователя). Если taichi установлен в такой каталог — копируем пакет в
    ASCII-временную папку и подключаем её первой в sys.path."""
    spec = importlib.util.find_spec("taichi")
    if spec is None:
        return
    origin = spec.origin or ""
    if _is_ascii_path(origin):
        return
    dst_root = pathlib.Path(tempfile.gettempdir()) / "opencode_taichi_pkg"
    root_str = _short_ascii_path(str(dst_root)) or str(dst_root)
    dst_root = pathlib.Path(root_str)
    dst = dst_root / "taichi"
    src = pathlib.Path(origin).parent
    ver = "1.7.4"
    marker = dst / ".opencode_ascii"
    if not (marker.exists() and marker.read_text() == ver):
        if dst.exists():
            shutil.rmtree(str(dst))
        dst_root.mkdir(parents=True, exist_ok=True)
        print("[bootstrap] копирую taichi в ASCII-путь:", dst_root)
        shutil.copytree(str(src), str(dst))
        marker.write_text(ver)
    if str(dst_root) not in sys.path:
        sys.path.insert(0, str(dst_root))


_bootstrap_taichi()

import taichi as ti  # noqa: E402

ti.init(arch=ti.vulkan)

# ------------------------------------------------------------------ константы
COLS = 72
ROWS = 44
CELL = 16
W = COLS * CELL
H = ROWS * CELL
N = COLS * ROWS
MAXB = 4096
MAXBONDS = 8192
MAXP = 5200
PMAX = 5000
MAXDUST = 240
MAXSRC = 4
MAXNB = 24
BMAPW = (MAXB + 31) // 32   # слов битовой карты связей (пар блоков)
MAXNBR = 4   # статический радиус поиска соседей контакта (в клетках);
             # покрывает блоки до размеров MAXNBR*2 клеток рядом; динамический
             # радиус фильтруется в рантайме (см. phys_contact)

# параметры флюида (константы) — wg динамический, хранится в поле
H_SPH = 0.72
RHO0 = 2.6
KP = 0.32
KN = 0.64
RP_SPH = 0.28

PI = 3.14159265

# ================================================================ физический масштаб
# 1 клетка = 0.5 м, 1 кадр = 1/60 с. Порода — песчаник средней прочности.
# Все параметры дем-модели выводятся из этих величин: apply_rock / apply_water / apply_gravity.
LCELL = 0.5                 # м на клетку
RHO_R = 2500.0              # плотность породы, кг/м3
RHO_W = 1000.0              # плотность воды, кг/м3
G_PHYS = 9.81               # ускорение свободного падения, м/с2
FRAME = 1.0 / 60.0          # с на кадр
VEL_SCALE = FRAME / LCELL   # м/с -> клетки/кадр
G_SIM = G_PHYS / LCELL      # ускорение для породы, клетки/с2
DT_FLUID = 0.25             # шаг флюида в кадрах (3 подшага = 0.75 кадра на кадр)
NFLUID_STEPS = 3            # число подшагов флюида за кадр
# ускорение флюида: 3 подшага по DT_FLUID кадра на кадр -> N*DT=0.75 кадра,
# чтобы дать ровно G_PHYS, нужно fwg*N*DT = G_SIM*FRAME^2:
FWG_BASE = G_SIM * FRAME * FRAME / (NFLUID_STEPS * DT_FLUID)
BLOCK_M = RHO_R * LCELL * LCELL   # масса 2D-блока (толщина 1 м), кг
FT_SIGMA = 1e6 / (RHO_R * LCELL * LCELL)   # 1 МПа прочности -> клетки/с2
WPRESS = RHO_W * G_PHYS / (RHO_R * LCELL * LCELL)  # 1 м напора -> клетки/с2 на блок
LOAD_OVER = G_PHYS / (LCELL * LCELL)     # 1 м глубины -> клетки/с2 на блок
K0_LATERAL = 1.0            # коэффициент бокового давления (σ_h = K0·σ_v); 1.0 — гидростатический распор глубокого массива


def apply_gravity(scale):
    P[None].g = G_SIM * scale
    fwg[None] = FWG_BASE * scale


def apply_rock(E_GPa, rp_MPa, mu, rho=None):
    E = E_GPa * 1e9
    kb = E / (RHO_R * LCELL * LCELL)
    P[None].kb = kb
    P[None].kn = kb
    P[None].ks = kb / (2.0 * (1.0 + 0.25))   # G = E/2(1+nu), nu = 0.25
    P[None].kts = kb * 0.2
    P[None].cn = 0.55 * math.sqrt(kb)   # демпфирование контакта, zeta ~ 0.28, гасит колебания при жёсткой связи
    P[None].rp = rp_MPa
    P[None].mu = mu
    P[None].ftScale = FT_SIGMA
    P[None].rcFactor = 10.0
    P[None].fsFactor = 2.0
    P[None].k0 = K0_LATERAL
    P[None].relBend = 0.15
    P[None].dmgMax = 6.0
    P[None].dmgDecay = 0.85   # быстрее "забывает" перегрузку: дальние связи не рвутся, обвал не расползается
    P[None].brkCap = 24


def apply_water(head_m):
    P[None].kw = 2000.0
    P[None].wCap = WPRESS * head_m
    P[None].head = head_m

# ------------------------------------------------------------------ параметры
@ti.dataclass
class Params:
    g: ti.f32
    kn: ti.f32
    cn: ti.f32
    mu: ti.f32
    kts: ti.f32
    kb: ti.f32
    ks: ti.f32
    rp: ti.f32
    ftScale: ti.f32
    rcFactor: ti.f32
    fsFactor: ti.f32
    kw: ti.f32
    wCap: ti.f32
    fixedRows: ti.i32
    walls: ti.i32
    substeps: ti.i32
    depth: ti.f32
    head: ti.f32
    k0: ti.f32
    relBend: ti.f32
    dmgMax: ti.f32
    dmgDecay: ti.f32
    brkCap: ti.i32
    maxSize: ti.i32   # максимальный размер блока в модели (радиус поиска соседей)


P = Params.field(shape=())

# ------------------------------------------------------------------ поля блоков
bx = ti.field(ti.f32, MAXB)
by = ti.field(ti.f32, MAXB)
bvx = ti.field(ti.f32, MAXB)
bvy = ti.field(ti.f32, MAXB)
bfx = ti.field(ti.f32, MAXB)
bfy = ti.field(ti.f32, MAXB)
wfx = ti.field(ti.f32, MAXB)
wfy = ti.field(ti.f32, MAXB)
ovf = ti.field(ti.f32, MAXB)
bfixed = ti.field(ti.i32, MAXB)
bshade = ti.field(ti.f32, MAXB)
bstress = ti.field(ti.f32, MAXB)
bstressC = ti.field(ti.f32, MAXB)
brot = ti.field(ti.f32, MAXB)
bw = ti.field(ti.f32, MAXB)
btor = ti.field(ti.f32, MAXB)
nbond = ti.field(ti.i32, MAXB)
bsz = ti.field(ti.f32, MAXB)     # размер блока (сторона квадрата), клетки
hx = ti.field(ti.f32, MAXB)      # начальная позиция (для диагностики смещения)
hy = ti.field(ti.f32, MAXB)
bcount = ti.field(ti.i32, shape=())

# сетка верхней нагрузки (только по колонкам)
topYArr = ti.field(ti.f32, COLS)
topCol = ti.field(ti.i32, COLS)

# связи (bonds) — фиксированные массивы, длина покоя и направление у каждой
bondA = ti.field(ti.i32, MAXBONDS)
bondB = ti.field(ti.i32, MAXBONDS)
bondIntact = ti.field(ti.i32, MAXBONDS)
bondDmg = ti.field(ti.f32, MAXBONDS)
bondR = ti.field(ti.f32, MAXBONDS)
bondJ = ti.field(ti.f32, MAXBONDS)
brest = ti.field(ti.f32, MAXBONDS)      # длина покоя связи
bondNX = ti.field(ti.f32, MAXBONDS)     # начальное направление связи (единичный вектор)
bondNY = ti.field(ti.f32, MAXBONDS)
bondE = ti.field(ti.f32, MAXBONDS)   # накопленная упругая энергия связи
bondCount = ti.field(ti.i32, shape=())
# наличие связи между парой блоков — битовая карта (вместо сеточной is_bonded):
# bondMap[a, b>>5] хранит бит b&31. Перестраивается в count_bonds каждый кадр.
bondMap = ti.field(ti.i32, (MAXB, BMAPW))

# контактный хэш блоков
hhead = ti.field(ti.i32, N)
hnext = ti.field(ti.i32, MAXB)

# трение по парам (постоянно между кадрами) — open-addressing хэш вместо MAXB² (16M)
FRIC_SLOTS = 16384   # степень двойки
FRIC_MASK = FRIC_SLOTS - 1
fricKey = ti.field(ti.i32, FRIC_SLOTS)
fricS = ti.field(ti.f32, FRIC_SLOTS)
fricF = ti.field(ti.i32, FRIC_SLOTS)

frameNo = ti.field(ti.i32, shape=())
loadCur = ti.field(ti.f32, shape=())
loadRamp = ti.field(ti.f32, shape=())
broken = ti.field(ti.i32, shape=())
frameBrk = ti.field(ti.i32, shape=())
breakCrush = ti.field(ti.i32, shape=())
dmax = ti.field(ti.f32, shape=())

# ------------------------------------------------------------------ флюид
wx = ti.field(ti.f32, MAXP)
wy = ti.field(ti.f32, MAXP)
wpx = ti.field(ti.f32, MAXP)
wpy = ti.field(ti.f32, MAXP)
wvx = ti.field(ti.f32, MAXP)
wvy = ti.field(ti.f32, MAXP)
corrX = ti.field(ti.f32, MAXP)
corrY = ti.field(ti.f32, MAXP)
fx0 = ti.field(ti.f32, MAXP)
fy0 = ti.field(ti.f32, MAXP)
pCount = ti.field(ti.i32, shape=())
fhead = ti.field(ti.i32, N)
fnxt = ti.field(ti.i32, MAXP)
# буфер перекрывающихся блоков для collide_water (одна частица -> до MAXNB блоков)
cwB = ti.field(ti.i32, (MAXP, MAXNB))
cwDx = ti.field(ti.f32, (MAXP, MAXNB))
cwDy = ti.field(ti.f32, (MAXP, MAXNB))
cwR = ti.field(ti.f32, (MAXP, MAXNB))    # порог контакта каждого блока (bsz/2 + RP_SPH)
fwg = ti.field(ti.f32, shape=())
capWarned = ti.field(ti.i32, shape=())

# источники воды
srcX = ti.field(ti.f32, MAXSRC)
srcY = ti.field(ti.f32, MAXSRC)
srcPh = ti.field(ti.f32, MAXSRC)
srcCount = ti.field(ti.i32, shape=())

# пыль (осколки при разрыве связи)
dustX = ti.field(ti.f32, MAXDUST)
dustY = ti.field(ti.f32, MAXDUST)
dustVX = ti.field(ti.f32, MAXDUST)
dustVY = ti.field(ti.f32, MAXDUST)
dustLife = ti.field(ti.f32, MAXDUST)
dustCol = ti.field(ti.i32, MAXDUST)
dustCount = ti.field(ti.i32, shape=())

# полость выработки (для диагностики заполнения водой)
cavX = ti.field(ti.f32, shape=())
cavY = ti.field(ti.f32, shape=())
cavRX = ti.field(ti.f32, shape=())
cavRY = ti.field(ti.f32, shape=())
filledFlag = ti.field(ti.i32, shape=())

# ------------------------------------------------------------------ вспомогательное
@ti.func
def bkey(a: ti.i32, b: ti.i32) -> ti.i32:
    lo = ti.min(a, b)
    hi = ti.max(a, b)
    return lo * MAXB + hi


@ti.func
def clamp_cell(x: ti.f32) -> ti.i32:
    c = int(x)
    if c < 0:
        c = 0
    elif c >= COLS:
        c = COLS - 1
    return c


@ti.func
def init_block(i: ti.i32, x: ti.f32, y: ti.f32, size: ti.f32, fixed: ti.i32):
    bx[i] = x
    by[i] = y
    bvx[i] = 0.0
    bvy[i] = 0.0
    bfx[i] = 0.0
    bfy[i] = 0.0
    wfx[i] = 0.0
    wfy[i] = 0.0
    ovf[i] = 0.0
    bfixed[i] = fixed
    bshade[i] = ti.random()
    bstress[i] = 0.0
    bstressC[i] = 0.0
    brot[i] = 0.0
    bw[i] = 0.0
    btor[i] = 0.0
    nbond[i] = 0
    bsz[i] = size
    hx[i] = x
    hy[i] = y


@ti.func
def add_block_func(x: ti.f32, y: ti.f32, size: ti.f32, fixed: ti.i32) -> ti.i32:
    """Добавить блок произвольного размера в произвольной точке. Возвращает индекс или -1.

    Выделение слота происходит неатомарно (один поток), поэтому вызов применим
    из последовательных контекстов (gen_tunnel, инструменты). Для массовой
    параллельной загрузки используйте _load_blocks (детерминированный слот i==k)."""
    res = -1
    if bcount[None] < MAXB:
        i = bcount[None]
        init_block(i, x, y, size, fixed)
        bcount[None] += 1
        res = i
    return res


@ti.func
def add_bond_func(a: ti.i32, b: ti.i32, rest: ti.f32, intact: ti.i32):
    """Связать блоки a и b с длиной покоя rest (по умолчанию — текущее расстояние)."""
    if bondCount[None] < MAXBONDS:
        i = ti.atomic_add(bondCount[None], 1)
        bondA[i] = a
        bondB[i] = b
        bondIntact[i] = intact
        bondDmg[i] = 0.0
        bondR[i] = 0.0
        bondJ[i] = (ti.random() - 0.5) * 0.3
        brest[i] = rest
        dx = bx[b] - bx[a]
        dy = by[b] - by[a]
        d = ti.sqrt(dx * dx + dy * dy)
        if d < 1e-9:
            bondNX[i] = 1.0
            bondNY[i] = 0.0
        else:
            bondNX[i] = dx / d
            bondNY[i] = dy / d
        bondE[i] = 0.0


@ti.func
def set_bond_bit(a: ti.i32, b: ti.i32):
    ti.atomic_or(bondMap[a, b >> 5], 1 << (b & 31))
    ti.atomic_or(bondMap[b, a >> 5], 1 << (a & 31))


@ti.func
def clear_bond_bit(a: ti.i32, b: ti.i32):
    ti.atomic_and(bondMap[a, b >> 5], ~(1 << (b & 31)))
    ti.atomic_and(bondMap[b, a >> 5], ~(1 << (a & 31)))


@ti.func
def is_bonded_bit(a: ti.i32, b: ti.i32) -> ti.i32:
    return (bondMap[a, b >> 5] >> (b & 31)) & 1


@ti.func
def remove_block_func(rem: ti.i32):
    """Удалить блок: связи с ним умирают, последний блок переносится на его место."""
    i = 0
    while i < bondCount[None]:
        a = bondA[i]
        b = bondB[i]
        if a == rem or b == rem:
            bondIntact[i] = 0
            clear_bond_bit(a, b)
        i += 1
    last = bcount[None] - 1
    if rem != last:
        bx[rem] = bx[last]
        by[rem] = by[last]
        bvx[rem] = bvx[last]
        bvy[rem] = bvy[last]
        bfx[rem] = bfx[last]
        bfy[rem] = bfy[last]
        wfx[rem] = wfx[last]
        wfy[rem] = wfy[last]
        ovf[rem] = ovf[last]
        bfixed[rem] = bfixed[last]
        bshade[rem] = bshade[last]
        bstress[rem] = bstress[last]
        bstressC[rem] = bstressC[last]
        brot[rem] = brot[last]
        bw[rem] = bw[last]
        btor[rem] = btor[last]
        nbond[rem] = nbond[last]
        bsz[rem] = bsz[last]
        hx[rem] = hx[last]
        hy[rem] = hy[last]
        i = 0
        while i < bondCount[None]:
            if bondIntact[i] == 1:
                a = bondA[i]
                b = bondB[i]
                if a == last:
                    bondA[i] = rem
                elif b == last:
                    bondB[i] = rem
            i += 1
    bcount[None] -= 1


@ti.kernel
def compact_bonds():
    n = 0
    i = 0
    while i < bondCount[None]:
        a = bondA[i]
        b = bondB[i]
        if bondIntact[i] == 1:
            bondA[n] = a
            bondB[n] = b
            bondIntact[n] = 1
            bondR[n] = bondR[i]
            bondDmg[n] = bondDmg[i]
            bondJ[n] = bondJ[i]
            brest[n] = brest[i]
            bondNX[n] = bondNX[i]
            bondNY[n] = bondNY[i]
            bondE[n] = bondE[i]
            n += 1
        i += 1
    bondCount[None] = n


@ti.func
def spawn_particle(x: ti.f32, y: ti.f32, vm: ti.f32, a0: ti.f32):
    if pCount[None] < PMAX:
        i = pCount[None]
        pCount[None] += 1
        wx[i] = x + (ti.random() - 0.5) * 0.35
        wy[i] = y + (ti.random() - 0.5) * 0.35
        wpx[i] = wx[i]
        wpy[i] = wy[i]
        a = a0 + (ti.random() - 0.5) * 0.6
        v = vm * (0.6 + 0.4 * ti.random())
        wvx[i] = ti.cos(a) * v
        wvy[i] = ti.sin(a) * v
    else:
        capWarned[None] = 1


@ti.func
def kill_particle(i: ti.i32):
    last = pCount[None] - 1
    wx[i] = wx[last]
    wy[i] = wy[last]
    wpx[i] = wpx[last]
    wpy[i] = wpy[last]
    wvx[i] = wvx[last]
    wvy[i] = wvy[last]
    pCount[None] -= 1


@ti.func
def break_event(a: ti.i32, b: ti.i32, crush: ti.i32):
    ti.atomic_add(broken[None], 1)
    breakCrush[None] = crush
    d = ti.atomic_add(dustCount[None], 1)
    if d < MAXDUST:
        dustX[d] = (bx[a] + bx[b]) * 0.5
        dustY[d] = (by[a] + by[b]) * 0.5
        dustVX[d] = (ti.random() - 0.5) * 0.1
        dustVY[d] = (ti.random() - 0.5) * 0.1
        dustLife[d] = 1.0
        dustCol[d] = 1 if crush == 1 else 0


@ti.func
def try_fracture(bd: ti.i32, a: ti.i32, b: ti.i32, crush: ti.i32) -> ti.i32:
    """Разрушение с накоплением повреждений (rate-limited).

    Лимит разрывов за кадр (brkCap) не даёт лавине порваться вся сразу:
    сверх лимита связи накапливают 'повреждение' и рвутся только при
    удержании перегрузки несколько кадров подряд. Локальная перегрузка
    успевает перераспределиться, и каскадное обрушение затухает."""
    ok = 0
    if ti.atomic_add(frameBrk[None], 1) < P[None].brkCap:
        bondDmg[bd] = P[None].dmgMax
        ok = 1
    else:
        bondDmg[bd] += 1.0
        if bondDmg[bd] >= P[None].dmgMax:
            ok = 1
    return ok


@ti.kernel
def decay_bond_damage():
    """Сглаживание повреждений между кадрами: мгновенный перегрузка спадает,
    разрушение требует длительного УДЕРЖАНИЯ перегрузки (скорость роста трещины)."""
    for bd in range(bondCount[None]):
        if bondIntact[bd] == 1 and bondDmg[bd] > 0.0:
            bondDmg[bd] *= P[None].dmgDecay


@ti.func
def fracture_bond(bd: ti.i32, a: ti.i32, b: ti.i32, crush: ti.i32,
                  strain: ti.f32, shear: ti.f32, relb: ti.f32):
    """Разрыв связи: вся накопленная упругая энергия bondE[bd]
    одномоментно переходит в кинетическую энергию осколков.
    Направление — по оси связи (нормаль) + касательная (сдвиг)."""
    bondIntact[bd] = 0
    clear_bond_bit(a, b)
    break_event(a, b, crush)

    # --- направление связи ---
    dx = bx[b] - bx[a]
    dy = by[b] - by[a]
    d = ti.sqrt(dx * dx + dy * dy)
    ux = 1.0
    uy = 0.0
    if d > 1e-9:
        ux = dx / d
        uy = dy / d
    else:
        ux = bondNX[bd]
        uy = bondNY[bd]
    px = -uy
    py = ux

    # --- энергетически корректный пинок ---
    E_acc = bondE[bd]          # вся накопленная энергия
    bondE[bd] = 0.0

    if crush == 1:
        pass
    else:
        mred = 0.5
        dv = ti.sqrt(2.0 * E_acc / mred)

        dv_n = dv * 0.7
        dv_s = dv * 0.3

        sgn_n = 1.0 if strain > 0.0 else -1.0
        bvx[a] -= ux * sgn_n * dv_n * 0.5
        bvy[a] -= uy * sgn_n * dv_n * 0.5
        bvx[b] += ux * sgn_n * dv_n * 0.5
        bvy[b] += uy * sgn_n * dv_n * 0.5

        sgn_s = 1.0 if shear > 0.0 else -1.0
        bvx[a] -= px * sgn_s * dv_s * 0.5
        bvy[a] -= py * sgn_s * dv_s * 0.5
        bvx[b] += px * sgn_s * dv_s * 0.5
        bvy[b] += py * sgn_s * dv_s * 0.5

        if ti.abs(relb) > 1e-6:
            Ia = bsz[a] * bsz[a] / 6.0
            Ib = bsz[b] * bsz[b] / 6.0
            Ired = Ia * Ib / (Ia + Ib + 1e-9)
            E_rot = E_acc * 0.15
            dw = ti.sqrt(2.0 * E_rot / Ired)
            sgr = 1.0 if relb > 0.0 else -1.0
            bw[a] -= sgr * dw * 0.5
            bw[b] += sgr * dw * 0.5


@ti.func
def fric_slot(key: ti.i32) -> ti.i32:
    """Найти или создать слот для пары блоков. -1 если хэш переполнен."""
    h = key ^ (key >> 12)
    h = h ^ (h >> 6)
    h = h & FRIC_MASK
    s = h
    res = -1
    guard = 0
    while guard < FRIC_SLOTS:
        if res >= 0:
            break
        k = fricKey[s]
        if k == key:
            res = s
        elif k == -1:
            fricKey[s] = key
            fricS[s] = 0.0
            fricF[s] = 0
            res = s
        else:
            s = (s + 1) & FRIC_MASK
        guard += 1
    return res


@ti.func
def friction_apply(i: ti.i32, j: ti.i32, key: ti.i32, vt: ti.f32, Fn: ti.f32, dt: ti.f32, axis: ti.i32,
                   sx: ti.f32, sy: ti.f32):
    s = fric_slot(key)
    if s >= 0:
        e_s = fricS[s]
        e_f = fricF[s]
        if frameNo[None] - e_f > 3:
            e_s = 0.0
        else:
            e_s *= 0.5
        fricF[s] = frameNo[None]
        e_s += vt * dt
        Ft = -P[None].kts * e_s
        Fm = P[None].mu * Fn
        if Ft > Fm:
            Ft = Fm
        elif Ft < -Fm:
            Ft = -Fm
        fricS[s] = -Ft / P[None].kts
        if axis == 1:
            ti.atomic_add(bfy[i], -Ft)
            ti.atomic_add(bfy[j], Ft)
            if nbond[i] == 0:
                ti.atomic_add(btor[i], -Ft * sx * 0.9)
            if nbond[j] == 0:
                ti.atomic_add(btor[j], -Ft * sx * 0.9)
        else:
            ti.atomic_add(bfx[i], -Ft)
            ti.atomic_add(bfx[j], Ft)
            if nbond[i] == 0:
                ti.atomic_add(btor[i], Ft * sy * 0.9)
            if nbond[j] == 0:
                ti.atomic_add(btor[j], Ft * sy * 0.9)


# ================================================================== DEM физика
@ti.kernel
def phys_reset():
    g = P[None].g
    for i in range(bcount[None]):
        bfx[i] = wfx[i]
        btor[i] = 0.0
        if bfixed[i] == 1:
            bfy[i] = wfy[i] + ovf[i]
        else:
            bfy[i] = wfy[i] + ovf[i] + g


@ti.kernel
def phys_bonds(dt: ti.f32):
    Ftmax = P[None].rp * P[None].ftScale
    Fcmax = Ftmax * P[None].rcFactor
    Fsmax = Ftmax * P[None].fsFactor
    kb = P[None].kb
    cn = P[None].cn
    ks = P[None].ks
    fov = loadCur[None] * loadRamp[None]   # вертикальная перегрузка на верх блока
    for bd in range(bondCount[None]):
        a = bondA[bd]
        b = bondB[bd]
        dx = bx[b] - bx[a]
        dy = by[b] - by[a]
        d = ti.sqrt(dx * dx + dy * dy)
        # боковое горное давление: σ_h = K0·σ_v(z) — распор массива.
        # Прикладываем к почти-горизонтальным связям (и к только что раскрывшимся,
        # пока отслоившийся блок ещё рядом): сжатый массив продолжает выдавливать
        # отслоившийся столб в выработку.
        if ti.abs(bondNX[bd]) > ti.abs(bondNY[bd]) and (bondIntact[bd] == 1 or (d - brest[bd]) < 1.5):
            midy = (by[a] + by[b]) * 0.5
            sv = fov + P[None].g * (ROWS - 0.5 - midy)
            sh = sv * P[None].k0
            sdx = 1.0 if dx > 0.0 else -1.0
            ti.atomic_add(bfx[a], -sh * sdx)
            ti.atomic_add(bfx[b], sh * sdx)
        if bondIntact[bd] == 0:
            continue
        if d < 1e-9:
            continue
        ux = dx / d
        uy = dy / d
        rvx = bvx[b] - bvx[a]
        rvy = bvy[b] - bvy[a]
        vn = rvx * ux + rvy * uy
        strain = d - brest[bd]
        Fn = -kb * strain - cn * vn
        tension = kb * strain
        # сдвиг поперёк начального направления связи
        px = -bondNY[bd]
        py = bondNX[bd]
        off = (dx - brest[bd] * bondNX[bd]) * px + (dy - brest[bd] * bondNY[bd]) * py
        vs = rvx * px + rvy * py
        Fs = -ks * off - cn * vs
        shearF = ks * off
        rel = brot[b] - brot[a]
        rw = bw[b] - bw[a]
        krot = kb * 0.02
        Fx = Fn * ux + Fs * px
        Fy = Fn * uy + Fs * py
        # Накопление упругой энергии за подшаг:
        # E += ½·k·u² · dt  (нормаль + сдвиг + изгиб) ИЗМ
        bondE[bd] = 0.5 * kb * strain * strain + 0.5 * ks * off * off + 0.5 * krot * rel * rel
        ti.atomic_add(bfx[b], Fx)
        ti.atomic_add(bfy[b], Fy)
        ti.atomic_add(bfx[a], -Fx)
        ti.atomic_add(bfy[a], -Fy)
        tor = -krot * rel - cn * 5.0 * rw
        ti.atomic_add(btor[a], tor)
        ti.atomic_add(btor[b], -tor)
        if tension > 0.0:
            r = tension / Ftmax
            bondR[bd] = r
            ti.atomic_max(bstress[a], r)
            ti.atomic_max(bstress[b], r)
            if tension > Ftmax or tension > kb * 0.12:
                if try_fracture(bd, a, b, 0):
                    fracture_bond(bd, a, b, 0, strain, off, rel)
        else:
            bondR[bd] = tension / (Fcmax * 0.25)
            c = -tension / Fcmax
            ti.atomic_max(bstressC[a], c)
            ti.atomic_max(bstressC[b], c)
            if -tension > Fcmax:
                if try_fracture(bd, a, b, 1):
                    fracture_bond(bd, a, b, 1, strain, off, rel)
            elif ti.abs(shearF) > Fsmax:
                if try_fracture(bd, a, b, 0):
                    fracture_bond(bd, a, b, 0, strain, off, rel)
            elif ti.abs(rel) > P[None].relBend:
                if try_fracture(bd, a, b, 0):
                    fracture_bond(bd, a, b, 0, strain, off, rel)


@ti.kernel
def phys_contact(dt: ti.f32):
    kn = P[None].kn
    cn = P[None].cn
    RR = P[None].maxSize   # радиус поиска соседей в клетках (по макс. размеру блока)
    for i in range(bcount[None]):
        cxi = clamp_cell(bx[i])
        cyi = clamp_cell(by[i])
        ri = bsz[i] * 0.5
        for oy in range(-MAXNBR, MAXNBR + 1):
            if oy > RR or oy < -RR:
                continue
            yy = cyi + oy
            if yy >= 0 and yy < ROWS:
                for ox in range(-MAXNBR, MAXNBR + 1):
                    if ox > RR or ox < -RR:
                        continue
                    xx = cxi + ox
                    if xx >= 0 and xx < COLS:
                        j = hhead[yy * COLS + xx]
                        while j != -1:
                            if j > i:
                                if not (bfixed[i] == 1 and bfixed[j] == 1):
                                    key = bkey(i, j)
                                    if is_bonded_bit(i, j) == 0:
                                        dx = bx[j] - bx[i]
                                        dy = by[j] - by[i]
                                        rj = bsz[j] * 0.5
                                        rs = ri + rj
                                        axx = ti.abs(dx)
                                        ayy = ti.abs(dy)
                                        oxx = rs - axx
                                        oyy = rs - ayy
                                        if oxx > 0.0 and oyy > 0.0:
                                            if oxx < oyy:
                                                s = 1.0 if dx > 0.0 else -1.0
                                                vsep = (bvx[j] - bvx[i]) * s
                                                Fn = kn * oxx - cn * vsep
                                                if Fn < 0.0:
                                                    Fn = 0.0
                                                ti.atomic_add(bfx[i], -Fn * s)
                                                ti.atomic_add(bfx[j], Fn * s)
                                                tau = -cn * vsep * s * dy * 0.8
                                                if nbond[i] == 0:
                                                    ti.atomic_add(btor[i], tau)
                                                if nbond[j] == 0:
                                                    ti.atomic_add(btor[j], tau)
                                                friction_apply(i, j, key, bvy[j] - bvy[i], Fn, dt, 1, s, 0.0)
                                            else:
                                                s = 1.0 if dy > 0.0 else -1.0
                                                vsep = (bvy[j] - bvy[i]) * s
                                                Fn = kn * oyy - cn * vsep
                                                if Fn < 0.0:
                                                    Fn = 0.0
                                                ti.atomic_add(bfy[i], -Fn * s)
                                                ti.atomic_add(bfy[j], Fn * s)
                                                tau = cn * vsep * s * dx * 0.8
                                                if nbond[i] == 0:
                                                    ti.atomic_add(btor[i], tau)
                                                if nbond[j] == 0:
                                                    ti.atomic_add(btor[j], tau)
                                                friction_apply(i, j, key, bvx[j] - bvx[i], Fn, dt, 0, 0.0, s)
                            j = hnext[j]


@ti.kernel
def phys_integrate(dt: ti.f32):
    walls = P[None].walls
    for i in range(bcount[None]):
        if bfixed[i] == 1:
            bvx[i] = 0.0
            bvy[i] = 0.0
            bw[i] = 0.0
            continue
        ri = bsz[i] * 0.5
        bvx[i] += bfx[i] * dt
        bvy[i] += bfy[i] * dt
        bvx[i] *= 0.9975
        bvy[i] *= 0.9975
        bw[i] += btor[i] * 1.0 * dt
        if nbond[i] > 0:
            bw[i] *= 0.97
        else:
            bw[i] *= 0.992
        if ti.abs(bw[i]) < 2e-4:
            bw[i] = 0.0
        if bw[i] > 10.0:
            bw[i] = 10.0
        elif bw[i] < -10.0:
            bw[i] = -10.0
        brot[i] += bw[i] * dt
        v2 = bvx[i] * bvx[i] + bvy[i] * bvy[i]
        if v2 > 900.0:
            s = 30.0 / ti.sqrt(v2)
            bvx[i] *= s
            bvy[i] *= s
        bx[i] += bvx[i] * dt
        by[i] += bvy[i] * dt
        if walls == 1:
            if bx[i] < ri:
                bx[i] = ri
                if bvx[i] < 0.0:
                    bvx[i] = 0.0
            elif bx[i] > COLS - ri:
                bx[i] = COLS - ri
                if bvx[i] > 0.0:
                    bvx[i] = 0.0
        if by[i] < ri:
            by[i] = ri
            if bvy[i] < 0.0:
                bvy[i] = 0.0
        if by[i] > ROWS - ri:
            by[i] = ROWS - ri
            if bvy[i] > 0.0:
                bvy[i] = 0.0


@ti.kernel
def count_bonds():
    for i in range(bcount[None]):
        nbond[i] = 0
    for i in range(bcount[None]):
        for w in range(BMAPW):
            bondMap[i, w] = 0
    for bd in range(bondCount[None]):
        if bondIntact[bd] == 1:
            a = bondA[bd]
            b = bondB[bd]
            ti.atomic_add(nbond[a], 1)
            ti.atomic_add(nbond[b], 1)
            ti.atomic_or(bondMap[a, b >> 5], 1 << (b & 31))
            ti.atomic_or(bondMap[b, a >> 5], 1 << (a & 31))


@ti.kernel
def build_hash():
    for c in range(N):
        hhead[c] = -1
    for i in range(bcount[None]):
        cxi = clamp_cell(bx[i])
        cyi = clamp_cell(by[i])
        c = cyi * COLS + cxi
        hnext[i] = hhead[c]
        hhead[c] = i
        bstress[i] *= 0.88
        bstressC[i] *= 0.92


@ti.kernel
def cleanup_fallen():
    i = bcount[None] - 1
    while i >= 0:
        if bfixed[i] == 0 and (by[i] > ROWS + 6 or bx[i] < -4 or bx[i] > COLS + 4):
            remove_block_func(i)
        i -= 1


# ------------------------------------------------------------------ верхняя нагрузка
@ti.kernel
def compute_top_load_fused(mode: ti.i32):
    # 1. обнуление ovf
    for i in range(bcount[None]):
        ovf[i] = 0.0
    # 2. обновление нагрузки
    if mode == 0:
        loadCur[None] += (P[None].depth * LOAD_OVER - loadCur[None]) * 0.04
        if loadRamp[None] < 1.0:
            loadRamp[None] = ti.min(1.0, loadRamp[None] + 1.0 / 90.0)
    else:
        loadCur[None] = 0.0
    # 3. поиск верхних блоков (вложенный цикл, один kernel)
    for c in range(COLS):
        best = -1
        bestY = 1e9
        for i in range(bcount[None]):
            cc = clamp_cell(bx[i])
            if cc == c and by[i] < bestY:
                bestY = by[i]
                best = i
        topCol[c] = best
        topYArr[c] = bestY
    # 4. применение нагрузки
    f = loadCur[None] * loadRamp[None]
    for c in range(COLS):
        i = topCol[c]
        if i >= 0:
            ovf[i] = f


def compute_top_load(mode):
    compute_top_load_fused(mode)


# ================================================================== флюид
@ti.kernel
def fluid_integrate(dt: ti.f32):
    for i in range(pCount[None]):
        wvy[i] += fwg[None] * dt
        sp2 = wvx[i] * wvx[i] + wvy[i] * wvy[i]
        if sp2 > 0.16:
            sc = 0.4 / ti.sqrt(sp2)
            wvx[i] *= sc
            wvy[i] *= sc
        wpx[i] = wx[i]
        wpy[i] = wy[i]
        dx = wvx[i] * dt
        dy = wvy[i] * dt
        if dx > 0.35:
            dx = 0.35
        elif dx < -0.35:
            dx = -0.35
        if dy > 0.35:
            dy = 0.35
        elif dy < -0.35:
            dy = -0.35
        wx[i] += dx
        wy[i] += dy


@ti.kernel
def fluid_hash():
    for c in range(N):
        fhead[c] = -1
    for i in range(pCount[None]):
        cxi = clamp_cell(wx[i])
        cyi = clamp_cell(wy[i])
        b = cyi * COLS + cxi
        fnxt[i] = fhead[b]
        fhead[b] = i


@ti.kernel
def fluid_pbd(dt: ti.f32):
    dt2 = dt * dt
    for i in range(pCount[None]):
        xi = wx[i]
        yi = wy[i]
        bxi = clamp_cell(xi)
        byi = clamp_cell(yi)
        rho = 0.0
        rhoN = 0.0
        for oy in range(-1, 2):
            cy = byi + oy
            if cy < 0 or cy >= ROWS:
                continue
            for ox in range(-1, 2):
                cx = bxi + ox
                if cx < 0 or cx >= COLS:
                    continue
                j = fhead[cy * COLS + cx]
                while j != -1:
                    if j != i:
                        dx = wx[j] - xi
                        dy = wy[j] - yi
                        d2 = dx * dx + dy * dy
                        if d2 < H_SPH * H_SPH and d2 > 1e-9:
                            q = 1.0 - ti.sqrt(d2) / H_SPH
                            rho += q * q
                            rhoN += q * q * q
                    j = fnxt[j]
        Pr = KP * (rho - RHO0)
        PrN = KN * rhoN
        for oy in range(-1, 2):
            cy = byi + oy
            if cy < 0 or cy >= ROWS:
                continue
            for ox in range(-1, 2):
                cx = bxi + ox
                if cx < 0 or cx >= COLS:
                    continue
                j = fhead[cy * COLS + cx]
                while j != -1:
                    if j != i:
                        dx = wx[j] - xi
                        dy = wy[j] - yi
                        d2 = dx * dx + dy * dy
                        if d2 < H_SPH * H_SPH and d2 > 1e-9:
                            d = ti.sqrt(d2)
                            q = 1.0 - d / H_SPH
                            D = dt2 * (Pr * q + PrN * q * q)
                            if D > 0.04:
                                D = 0.04
                            elif D < 0.0:
                                D = 0.0
                            hx = dx / d * D * 0.4
                            hy = dy / d * D * 0.4
                            ti.atomic_add(corrX[j], hx)
                            ti.atomic_add(corrY[j], hy)
                            ti.atomic_add(corrX[i], -hx)
                            ti.atomic_add(corrY[i], -hy)
                    j = fnxt[j]


@ti.kernel
def fluid_apply_corr():
    m = RP_SPH
    for i in range(pCount[None]):
        wx[i] += corrX[i]
        wy[i] += corrY[i]
        corrX[i] = 0.0
        corrY[i] = 0.0
        if wx[i] < m:
            wx[i] = m
        elif wx[i] > COLS - m:
            wx[i] = COLS - m
        if wy[i] < m:
            wy[i] = m
        elif wy[i] > ROWS - m:
            wy[i] = ROWS - m


@ti.kernel
def fluid_vel(dt: ti.f32):
    for i in range(pCount[None]):
        wvx[i] = (wx[i] - wpx[i]) / dt
        wvy[i] = (wy[i] - wpy[i]) / dt
        s2 = wvx[i] * wvx[i] + wvy[i] * wvy[i]
        if s2 > 0.25:
            s = 0.5 / ti.sqrt(s2)
            wvx[i] *= s
            wvy[i] *= s


@ti.kernel
def collide_water(acc: ti.i32):
    if acc == 1:
        for i in range(bcount[None]):
            wfx[i] = 0.0
            wfy[i] = 0.0
    margin = RP_SPH
    RR = int(ti.ceil(P[None].maxSize * 0.5 + RP_SPH))
    for it in range(3):
        for p in range(pCount[None]):
            cxi = clamp_cell(wx[p])
            cyi = clamp_cell(wy[p])
            nL = 0
            nR = 0
            nU = 0
            nD = 0
            maxOX = 0.0
            sOX = 0.0
            bOX = -1
            maxOY = 0.0
            sOY = 0.0
            bOY = -1
            cc = 0
            oy = -RR
            while oy <= RR:
                yy = cyi + oy
                if yy >= 0 and yy < ROWS:
                    ox = -RR
                    while ox <= RR:
                        xx = cxi + ox
                        if xx >= 0 and xx < COLS:
                            b = hhead[yy * COLS + xx]
                            while b != -1:
                                dx = wx[p] - bx[b]
                                dy = wy[p] - by[b]
                                axx = ti.abs(dx)
                                ayy = ti.abs(dy)
                                hbs = bsz[b] * 0.5 + RP_SPH
                                if axx < hbs and ayy < hbs:
                                    if cc < MAXNB:
                                        cwB[p, cc] = b
                                        cwDx[p, cc] = dx
                                        cwDy[p, cc] = dy
                                        cwR[p, cc] = hbs
                                        cc += 1
                                    if dx > 0.0:
                                        nR += 1
                                    elif dx < 0.0:
                                        nL += 1
                                    if dy > 0.0:
                                        nD += 1
                                    elif dy < 0.0:
                                        nU += 1
                                    oxx = hbs - axx
                                    oyy = hbs - ayy
                                    if oxx > maxOX:
                                        maxOX = oxx
                                        sOX = 1.0 if dx > 0.0 else -1.0
                                        bOX = b
                                    if oyy > maxOY:
                                        maxOY = oyy
                                        sOY = 1.0 if dy > 0.0 else -1.0
                                        bOY = b
                                b = hnext[b]
                        ox += 1
                oy += 1
            sqz_x = 1 if (nL > 0 and nR > 0) else 0
            sqz_y = 1 if (nU > 0 and nD > 0) else 0
            if sqz_x == 1 and sqz_y == 1:
                pass
            else:
                sumOX = 0.0
                sumOY = 0.0
                k = 0
                while k < cc:
                    b = cwB[p, k]
                    dx = cwDx[p, k]
                    dy = cwDy[p, k]
                    hbs = cwR[p, k]
                    oxx = hbs - ti.abs(dx)
                    oyy = hbs - ti.abs(dy)
                    if oxx < oyy:
                        s = 1.0 if dx > 0.0 else -1.0
                        sumOX += oxx * s
                        if acc == 1 and it == 0 and bfixed[b] == 0:
                            ti.atomic_sub(wfx[b], s * ti.min(oxx * P[None].kw, P[None].wCap))
                    else:
                        s = 1.0 if dy > 0.0 else -1.0
                        sumOY += oyy * s
                        if acc == 1 and it == 0 and bfixed[b] == 0:
                            ti.atomic_sub(wfy[b], s * ti.min(oyy * P[None].kw, P[None].wCap))
                    k += 1
                wx[p] += sumOX
                wy[p] += sumOY
            if sqz_x == 1 and sqz_y == 0 and maxOY > 0.0:
                wy[p] += maxOY * sOY
                if acc == 1 and it == 0 and bOY != -1 and bfixed[bOY] == 0:
                    ti.atomic_sub(wfy[bOY], sOY * ti.min(maxOY * P[None].kw, P[None].wCap))
            elif sqz_y == 1 and sqz_x == 0 and maxOX > 0.0:
                wx[p] += maxOX * sOX
                if acc == 1 and it == 0 and bOX != -1 and bfixed[bOX] == 0:
                    ti.atomic_sub(wfx[bOX], sOX * ti.min(maxOX * P[None].kw, P[None].wCap))
            if wx[p] < margin:
                wx[p] = margin
            elif wx[p] > COLS - margin:
                wx[p] = COLS - margin
            if wy[p] < margin:
                wy[p] = margin
            elif wy[p] > ROWS - margin:
                wy[p] = ROWS - margin
    p = pCount[None] - 1
    while p >= 0:
        if wx[p] < -1 or wx[p] > COLS + 1 or wy[p] < -1 or wy[p] > ROWS + 1:
            kill_particle(p)
        p -= 1


@ti.kernel
def fluid_frame_start():
    for i in range(pCount[None]):
        fx0[i] = wx[i]
        fy0[i] = wy[i]


@ti.kernel
def fluid_clamp_disp():
    LIM = 1.5
    for i in range(pCount[None]):
        dx = wx[i] - fx0[i]
        if dx > LIM:
            wx[i] = fx0[i] + LIM
        elif dx < -LIM:
            wx[i] = fx0[i] - LIM
        dy = wy[i] - fy0[i]
        if dy > LIM:
            wy[i] = fy0[i] + LIM
        elif dy < -LIM:
            wy[i] = fy0[i] - LIM


@ti.kernel
def spawn_water(n: ti.i32, vm: ti.f32, x0: ti.f32, y0: ti.f32, a0: ti.f32):
    k = 0
    while k < n:
        if pCount[None] >= PMAX:
            capWarned[None] = 1
            break
        spawn_particle(x0, y0, vm, a0)
        k += 1


@ti.kernel
def fill_pct() -> ti.f32:
    n = 0.0
    for i in range(pCount[None]):
        dx = (wx[i] - cavX[None]) / cavRX[None]
        dy = (wy[i] - cavY[None]) / cavRY[None]
        if dx * dx + dy * dy < 1.0:
            n += 1.0
    cap = PI * cavRX[None] * cavRY[None] / (PI * RP_SPH * RP_SPH)
    p = n / cap * 100.0
    if p > 100.0:
        p = 100.0
    return p


# ================================================================== диагностика
@ti.kernel
def count_intact() -> ti.i32:
    n = 0
    for i in range(bondCount[None]):
        if bondIntact[i] == 1:
            n += 1
    return n


@ti.kernel
def diag_stats() -> ti.f32:
    mx = 0.0
    for i in range(bcount[None]):
        dx = bx[i] - hx[i]
        dy = by[i] - hy[i]
        d = ti.sqrt(dx * dx + dy * dy)
        if d > mx:
            mx = d
    return mx


@ti.kernel
def diag_strain() -> ti.f32:
    dmax[None] = 0.0
    for bd in range(bondCount[None]):
        if bondIntact[bd] == 1:
            a = bondA[bd]
            b = bondB[bd]
            dx = bx[b] - bx[a]
            dy = by[b] - by[a]
            st = ti.abs(ti.sqrt(dx * dx + dy * dy) - brest[bd])
            ti.atomic_max(dmax[None], st)
    return dmax[None]


# ================================================================== загрузка модели
@ti.kernel
def reset_all():
    """Полный сброс состояния к пустой модели (конфигурация не трогается)."""
    bcount[None] = 0
    bondCount[None] = 0
    pCount[None] = 0
    dustCount[None] = 0
    broken[None] = 0
    loadCur[None] = 0.0
    loadRamp[None] = 0.0
    capWarned[None] = 0
    filledFlag[None] = 0
    srcCount[None] = 0
    frameNo[None] = 0
    frameBrk[None] = 0
    breakCrush[None] = 0
    dmax[None] = 0.0
    P[None].maxSize = 1
    for c in range(N):
        hhead[c] = -1
        fhead[c] = -1
    for c in range(COLS):
        topCol[c] = -1
        topYArr[c] = 1e9
    for s in range(FRIC_SLOTS):
        fricKey[s] = -1
        fricS[s] = 0.0
        fricF[s] = 0
    for i in range(MAXB):
        bfixed[i] = 0
        nbond[i] = 0
    for bd in range(MAXBONDS):
        bondIntact[bd] = 0
        bondE[bd] = 0.0
    for i in range(MAXB):
        for w in range(BMAPW):
            bondMap[i, w] = 0


@ti.kernel
def _load_blocks(blocks: ti.types.ndarray()):
    for k in range(blocks.shape[0]):
        if k < MAXB:
            init_block(k, blocks[k, 0], blocks[k, 1], blocks[k, 2], ti.cast(blocks[k, 3], ti.i32))


@ti.kernel
def _load_bonds(bonds: ti.types.ndarray()):
    for k in range(bonds.shape[0]):
        add_bond_func(ti.cast(bonds[k, 0], ti.i32), ti.cast(bonds[k, 1], ti.i32), bonds[k, 2],
                      ti.cast(bonds[k, 3], ti.i32))


@ti.kernel
def _load_water(water: ti.types.ndarray()):
    for k in range(water.shape[0]):
        spawn_particle(water[k, 0], water[k, 1], water[k, 2], water[k, 3])


def build_model(blocks, bonds, water, cavity):
    """Загрузить модель, построенную снаружи (render.py).

    blocks — [(x, y, size, fixed), ...]   позиция центра, размер (сторона), флаг закрепления
    bonds  — [(a, b, rest, intact), ...]  индексы блоков, длина покоя, целостность
    water  — [(x, y, v, ang), ...]        частицы воды
    cavity — (cx, cy, rx, ry) или None    полость (для диагностики заполнения)
    """
    reset_all()
    if cavity:
        cavX[None] = cavity[0]
        cavY[None] = cavity[1]
        cavRX[None] = cavity[2]
        cavRY[None] = cavity[3]
    maxs = 1.0
    if blocks:
        for b in blocks:
            if b[2] > maxs:
                maxs = b[2]
    P[None].maxSize = max(1, int(math.ceil(maxs)))
    if blocks:
        nb = min(len(blocks), MAXB)
        bcount[None] = nb
        _load_blocks(np.asarray(blocks, dtype=np.float32))
    if bonds:
        _load_bonds(np.asarray(bonds, dtype=np.float32))
    if water:
        _load_water(np.asarray(water, dtype=np.float32))


# ================================================================== источники воды
def place_source(x, y):
    if srcCount[None] >= 4:
        return "Maksimum 4 istochnika"
    s = srcCount[None]
    srcX[s] = x
    srcY[s] = y
    srcPh[s] = time.time() % 1.0
    srcCount[None] += 1
    if ((x - cavX[None]) / cavRX[None]) ** 2 + ((y - cavY[None]) / cavRY[None]) ** 2 < 1:
        return "Istochnik: H=%d m * v vyrabotke - zapolnenie" % P[None].head
    return "Istochnik: H=%d m" % P[None].head


def remove_source(x, y):
    bi = -1
    bd = 9.0
    for i in range(srcCount[None]):
        d = math.hypot(srcX[i] - x, srcY[i] - y)
        if d < bd:
            bd = d
            bi = i
    if bi >= 0:
        last = srcCount[None] - 1
        srcX[bi] = srcX[last]
        srcY[bi] = srcY[last]
        srcPh[bi] = srcPh[last]
        srcCount[None] -= 1
        return True
    return False


def emit_sources():
    if srcCount[None] == 0:
        return
    v0 = math.sqrt(2.0 * G_PHYS * P[None].head) * VEL_SCALE
    for i in range(srcCount[None]):
        spawn_water(3, v0, srcX[i], srcY[i], PI * 0.5)


# ================================================================== главный цикл
DT = 1.0 / (60.0 * 192)
SUBSTEPS = 192


def fluid_step(dt):
    fluid_integrate(dt)
    fluid_hash()
    fluid_pbd(dt)
    fluid_apply_corr()
    fluid_vel(dt)


def step_physics(mode, dt):
    frameNo[None] += 1
    frameBrk[None] = 0
    decay_bond_damage()
    emit_sources()
    if capWarned[None] == 1 and pCount[None] < PMAX * 0.85:
        capWarned[None] = 0
    build_hash()
    fluid_frame_start()
    fluid_step(DT_FLUID)
    collide_water(1)
    fluid_step(DT_FLUID)
    collide_water(0)
    fluid_step(DT_FLUID)
    collide_water(0)
    fluid_clamp_disp()
    compute_top_load(mode)
    count_bonds()
    for _ in range(SUBSTEPS):
        phys_reset()
        phys_bonds(dt)
        phys_contact(dt)
        phys_integrate(dt)
    cleanup_fallen()
    if frameNo[None] % 30 == 0:
        compact_bonds()
