"""ГЕОМАССИВ/2D · DEM — решатель (порт mineudec.html на Python/Taichi)

DEM: клетка = 1 м, блок m = 1 кг. Силы: Ft = Rp[МПа]·400 (соответствие ρ=2500).
Флюид: Position-Based Dynamics (SPH-плотность), контакт с породой герметичен —
вода идёт только в раскрытые трещины. Физика, генерация и инструменты.
Отрисовка и GUI — в render.py, точка входа — mineudec.py.
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
STIFF_SOFT = 8.0            # искусственное смягчение жёсткости (явная схема при dt=1/2880 c)


def apply_gravity(scale):
    P[None].g = G_SIM * scale
    fwg[None] = FWG_BASE * scale


def apply_rock(E_GPa, rp_MPa, mu):
    E = E_GPa * 1e9
    kb = E / (RHO_R * LCELL * LCELL) / STIFF_SOFT
    P[None].kb = kb
    P[None].kn = kb
    P[None].ks = kb / (2.0 * (1.0 + 0.25))   # G = E/2(1+nu), nu = 0.25
    P[None].kts = kb * 0.2
    P[None].cn = 0.15 * math.sqrt(kb)   # демпфирование контакта, zeta ~ 0.08
    P[None].rp = rp_MPa
    P[None].mu = mu
    P[None].ftScale = FT_SIGMA
    P[None].rcFactor = 10.0
    P[None].fsFactor = 2.0
    P[None].k0 = K0_LATERAL
    P[None].relBend = 0.15
    P[None].dmgMax = 6.0
    P[None].dmgDecay = 0.94
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
ovfX = ti.field(ti.f32, MAXB)
bfixed = ti.field(ti.i32, MAXB)
bhome = ti.field(ti.i32, MAXB)
bshade = ti.field(ti.f32, MAXB)
bstress = ti.field(ti.f32, MAXB)
bstressC = ti.field(ti.f32, MAXB)
brot = ti.field(ti.f32, MAXB)
bw = ti.field(ti.f32, MAXB)
btor = ti.field(ti.f32, MAXB)
nbond = ti.field(ti.i32, MAXB)
bcount = ti.field(ti.i32, shape=())

# сетка ячеек
homeFilled = ti.field(ti.i32, N)
homeToBlock = ti.field(ti.i32, N)
topYArr = ti.field(ti.f32, COLS)
topCol = ti.field(ti.i32, COLS)

# связи (bonds) — фиксированные массивы
bondA = ti.field(ti.i32, MAXBONDS)
bondB = ti.field(ti.i32, MAXBONDS)
bondAxis = ti.field(ti.i32, MAXBONDS)
bondIntact = ti.field(ti.i32, MAXBONDS)
bondDmg = ti.field(ti.f32, MAXBONDS)
bondR = ti.field(ti.f32, MAXBONDS)
bondJ = ti.field(ti.f32, MAXBONDS)
bondCount = ti.field(ti.i32, shape=())
# наличие связи между соседними домами: hBond[c] — индекс блока за правой гранью
# ячейки c (связь axis=0), vBond[c] — индекс блока за нижней гранью (axis=1)
hBond = ti.field(ti.i32, N)
vBond = ti.field(ti.i32, N)

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

# генерация / полость
cutR = ti.field(ti.i32, N)
cutD = ti.field(ti.i32, N)
noisePts = ti.field(ti.f32, 128)
n1Arr = ti.field(ti.f32, COLS)
n2Arr = ti.field(ti.f32, COLS)
hgtArr = ti.field(ti.f32, COLS)
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
def add_block_func(cell: ti.i32, fixed: ti.i32) -> ti.i32:
    i = bcount[None]
    bx[i] = (cell % COLS) + 0.5
    by[i] = (cell // COLS) + 0.5
    bvx[i] = 0.0
    bvy[i] = 0.0
    bfx[i] = 0.0
    bfy[i] = 0.0
    wfx[i] = 0.0
    wfy[i] = 0.0
    ovf[i] = 0.0
    bfixed[i] = fixed
    bhome[i] = cell
    bshade[i] = ti.random()
    bstress[i] = 0.0
    bstressC[i] = 0.0
    brot[i] = 0.0
    bw[i] = 0.0
    btor[i] = 0.0
    homeFilled[cell] = 1
    homeToBlock[cell] = i
    bcount[None] += 1
    return i


@ti.func
def set_bond_edge(a: ti.i32, b: ti.i32, axis: ti.i32, intact: ti.i32):
    ha = bhome[a]
    hb = bhome[b]
    if axis == 0:
        if ti.abs(ha - hb) == 1:
            base = ti.min(ha, hb)
            ob = a if ha > hb else b
            hBond[base] = ob if intact == 1 else -1
    else:
        if ti.abs(ha - hb) == COLS:
            base = ti.min(ha, hb)
            ob = a if ha > hb else b
            vBond[base] = ob if intact == 1 else -1


@ti.func
def clear_bond_edge(a: ti.i32, b: ti.i32):
    ha = bhome[a]
    hb = bhome[b]
    if ha // COLS == hb // COLS:
        hBond[ti.min(ha, hb)] = -1
    else:
        vBond[ti.min(ha, hb)] = -1


@ti.func
def add_bond_func(a: ti.i32, b: ti.i32, axis: ti.i32, intact: ti.i32):
    i = ti.atomic_add(bondCount[None], 1)
    bondA[i] = a
    bondB[i] = b
    bondAxis[i] = axis
    bondIntact[i] = intact
    bondDmg[i] = 0.0
    bondR[i] = 0.0
    bondJ[i] = (ti.random() - 0.5) * 0.3
    set_bond_edge(a, b, axis, intact)


@ti.func
def is_bonded(i: ti.i32, j: ti.i32) -> ti.i32:
    hi = bhome[i]
    hj = bhome[j]
    bonded = 0
    if hj == hi + 1 and (hi % COLS) < COLS - 1:
        bonded = 1 if hBond[hi] == j else 0
    else:
        if hj == hi - 1 and (hi % COLS) > 0:
            bonded = 1 if hBond[hj] == i else 0
        else:
            if hj == hi + COLS:
                bonded = 1 if vBond[hi] == j else 0
            else:
                if hj == hi - COLS:
                    bonded = 1 if vBond[hj] == i else 0
    return bonded


@ti.func
def remove_block_func(rem: ti.i32):
    homeFilled[bhome[rem]] = 0
    homeToBlock[bhome[rem]] = -1
    last = bcount[None] - 1
    # помечаем связи с rem как мёртвые (без уплотнения) + очищаем грани
    i = 0
    while i < bondCount[None]:
        a = bondA[i]
        b = bondB[i]
        if a == rem or b == rem:
            bondIntact[i] = 0
            clear_bond_edge(a, b)
        i += 1
    if rem != last:
        homeToBlock[bhome[last]] = rem
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
        bhome[rem] = bhome[last]
        bshade[rem] = bshade[last]
        bstress[rem] = bstress[last]
        bstressC[rem] = bstressC[last]
        brot[rem] = brot[last]
        bw[rem] = bw[last]
        btor[rem] = btor[last]
        nbond[rem] = nbond[last]
        # пересчёт граней: last -> rem, rem-связи уже мертвы
        i = 0
        while i < bondCount[None]:
            if bondIntact[i] == 1:
                a = bondA[i]
                b = bondB[i]
                if a == last:
                    bondA[i] = rem
                    set_bond_edge(rem, b, bondAxis[i], 1)
                elif b == last:
                    bondB[i] = rem
                    set_bond_edge(a, rem, bondAxis[i], 1)
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
            bondAxis[n] = bondAxis[i]
            bondIntact[n] = 1
            bondR[n] = bondR[i]
            bondDmg[n] = bondDmg[i]
            bondJ[n] = bondJ[i]
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
def fracture_bond(bd: ti.i32, a: ti.i32, b: ti.i32, crush: ti.i32):
    """Полный обрыв связи: отметить развязанным, выбить скорость и пыль."""
    axis = bondAxis[bd]
    bondIntact[bd] = 0
    set_bond_edge(a, b, axis, 0)
    break_event(a, b, crush)
    if axis == 0:
        bvx[a] -= 0.15
        bvx[b] += 0.15
    else:
        bvy[a] -= 0.15
        bvy[b] += 0.15


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
        bfx[i] = wfx[i] + ovfX[i]
        btor[i] = 0.0
        if bfixed[i] == 1:
            bfy[i] = wfy[i] + ovf[i]
        else:
            bfy[i] = wfy[i] + ovf[i] + g


@ti.kernel
def phys_bonds():
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
        # боковое горное давление: σ_h = K0·σ_v(z) — распор массива.
        # Прикладываем и к целым, и к только что раскрывшимся горизонтальным
        # связям, пока отслоившийся блок ещё рядом (спад/трещина не уехала):
        # сжатый массив продолжает выдавливать отслоившийся столб в выработку,
        # а не "приклеивает" его к стенке после разрыва связи.
        if bondAxis[bd] == 0 and (bondIntact[bd] == 1 or ti.abs(bx[b] - bx[a]) < 1.5):
            r = clamp_cell(by[a])
            sv = fov + P[None].g * (ROWS - 1 - r)
            sh = sv * P[None].k0
            ti.atomic_add(bfx[a], -sh)
            ti.atomic_add(bfx[b], sh)
        if bondIntact[bd] == 0:
            continue
        rvx = bvx[b] - bvx[a]
        rvy = bvy[b] - bvy[a]
        rel = brot[b] - brot[a]
        rw = bw[b] - bw[a]
        krot = kb * 0.02
        tor = -krot * rel - cn * 3.0 * rw
        ti.atomic_add(btor[a], tor)
        ti.atomic_add(btor[b], -tor)
        strain = 0.0
        tension = 0.0
        shearF = 0.0
        if bondAxis[bd] == 0:
            strain = bx[b] - bx[a] - 1.0
            Fn = -kb * strain - cn * rvx
            tension = kb * strain
            off = by[b] - by[a]
            Fs = -ks * off - cn * rvy
            shearF = ks * off
            ti.atomic_add(bfx[b], Fn)
            ti.atomic_add(bfx[a], -Fn)
            ti.atomic_add(bfy[b], Fs)
            ti.atomic_add(bfy[a], -Fs)
            tau = 0.0 * -Fs * 0.5
            ti.atomic_add(btor[a], tau)
            ti.atomic_add(btor[b], tau)
        else:
            strain = by[b] - by[a] - 1.0
            Fn = -kb * strain - cn * rvy
            tension = kb * strain
            off = bx[b] - bx[a]
            Fs = -ks * off - cn * rvx
            shearF = ks * off
            ti.atomic_add(bfy[b], Fn)
            ti.atomic_add(bfy[a], -Fn)
            ti.atomic_add(bfx[b], Fs)
            ti.atomic_add(bfx[a], -Fs)
            tau = 0.0 * Fs * 0.5
            ti.atomic_add(btor[a], tau)
            ti.atomic_add(btor[b], tau)
        if tension > 0.0:
            r = tension / Ftmax
            bondR[bd] = r
            ti.atomic_max(bstress[a], r)
            ti.atomic_max(bstress[b], r)
            if tension > Ftmax or tension > kb * 0.12:
                if try_fracture(bd, a, b, 0):
                    fracture_bond(bd, a, b, 0)
        else:
            bondR[bd] = tension / (Fcmax * 0.25)
            c = -tension / Fcmax
            ti.atomic_max(bstressC[a], c)
            ti.atomic_max(bstressC[b], c)
            if -tension > Fcmax:
                if try_fracture(bd, a, b, 1):
                    fracture_bond(bd, a, b, 1)
            elif ti.abs(shearF) > Fsmax:
                if try_fracture(bd, a, b, 0):
                    fracture_bond(bd, a, b, 0)
            elif ti.abs(rel) > P[None].relBend:
                if try_fracture(bd, a, b, 0):
                    fracture_bond(bd, a, b, 0)


@ti.kernel
def phys_contact(dt: ti.f32):
    kn = P[None].kn
    cn = P[None].cn
    for i in range(bcount[None]):
        cxi = clamp_cell(bx[i])
        cyi = clamp_cell(by[i])
        for oy in range(-1, 2):
            yy = cyi + oy
            if yy < 0 or yy >= ROWS:
                continue
            for ox in range(-1, 2):
                xx = cxi + ox
                if xx < 0 or xx >= COLS:
                    continue
                j = hhead[yy * COLS + xx]
                while j != -1:
                    if j > i:
                        if not (bfixed[i] == 1 and bfixed[j] == 1):
                            key = bkey(i, j)
                            if is_bonded(i, j) == 0:
                                dx = bx[j] - bx[i]
                                dy = by[j] - by[i]
                                axx = ti.abs(dx)
                                ayy = ti.abs(dy)
                                oxx = 1.0 - axx
                                oyy = 1.0 - ayy
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
            if bx[i] < 0.5:
                bx[i] = 0.5
                if bvx[i] < 0.0:
                    bvx[i] = 0.0
            elif bx[i] > COLS - 0.5:
                bx[i] = COLS - 0.5
                if bvx[i] > 0.0:
                    bvx[i] = 0.0
        if by[i] < 0.5:
            by[i] = 0.5
            if bvy[i] < 0.0:
                bvy[i] = 0.0
        if by[i] > ROWS - 0.5:
            by[i] = ROWS - 0.5
            if bvy[i] > 0.0:
                bvy[i] = 0.0


@ti.kernel
def count_bonds():
    for i in range(bcount[None]):
        nbond[i] = 0
    for bd in range(bondCount[None]):
        if bondIntact[bd] == 1:
            ti.atomic_add(nbond[bondA[bd]], 1)
            ti.atomic_add(nbond[bondB[bd]], 1)


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
def decay_stress():
    for i in range(bcount[None]):
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
    # 5. внешний боковой распор по краям убран: сжатие краёв удерживало бы
    #    выработку в воздухе; боковое давление действует только через связи (K0)


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
    hs = 0.5 + RP_SPH
    margin = RP_SPH
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
            for oy in range(-1, 2):
                yy = cyi + oy
                if yy < 0 or yy >= ROWS:
                    continue
                for ox in range(-1, 2):
                    xx = cxi + ox
                    if xx < 0 or xx >= COLS:
                        continue
                    b = hhead[yy * COLS + xx]
                    while b != -1:
                        dx = wx[p] - bx[b]
                        dy = wy[p] - by[b]
                        axx = ti.abs(dx)
                        ayy = ti.abs(dy)
                        if axx < hs and ayy < hs:
                            if cc < MAXNB:
                                cwB[p, cc] = b
                                cwDx[p, cc] = dx
                                cwDy[p, cc] = dy
                                cc += 1
                            if dx > 0.0:
                                nR += 1
                            elif dx < 0.0:
                                nL += 1
                            if dy > 0.0:
                                nD += 1
                            elif dy < 0.0:
                                nU += 1
                            oxx = hs - axx
                            oyy = hs - ayy
                            if oxx > maxOX:
                                maxOX = oxx
                                sOX = 1.0 if dx > 0.0 else -1.0
                                bOX = b
                            if oyy > maxOY:
                                maxOY = oyy
                                sOY = 1.0 if dy > 0.0 else -1.0
                                bOY = b
                        b = hnext[b]
            sqz_x = 1 if (nL > 0 and nR > 0) else 0
            sqz_y = 1 if (nU > 0 and nD > 0) else 0
            sumOX = 0.0
            sumOY = 0.0
            if sqz_x == 1 and sqz_y == 0:
                pass
            elif sqz_y == 1 and sqz_x == 0:
                pass
            else:
                k = 0
                while k < cc:
                    b = cwB[p, k]
                    dx = cwDx[p, k]
                    dy = cwDy[p, k]
                    oxx = hs - ti.abs(dx)
                    oyy = hs - ti.abs(dy)
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


# ================================================================== генерация
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
def generate(mode: ti.i32, weak: ti.i32):
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
    for c in range(N):
        homeFilled[c] = 0
        homeToBlock[c] = -1
        cutR[c] = 0
        cutD[c] = 0
    for c in range(N):
        hBond[c] = -1
        vBond[c] = -1
    for s in range(FRIC_SLOTS):
        fricKey[s] = -1
        fricS[s] = 0.0
        fricF[s] = 0

    if mode == 0:
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
                if y >= ROWS - 2:
                    add_block_func(cell, 1)
                else:
                    dx = (x + 0.5 - tx) / trx
                    dy = (y + 0.5 - ty) / try_
                    if dx * dx + dy * dy >= 1.0:
                        fixed = 1 if y >= ROWS - P[None].fixedRows else 0
                        add_block_func(cell, fixed)
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
    else:
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
                if y >= ROWS - 2:
                    add_block_func(cell, 1)
                else:
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
                    add_block_func(cell, fixed)
                x += 1
            y += 1
        if weak == 1:
            nj = 4 + int(ti.random() * 3)
            jn = 0
            while jn < nj:
                jx = int(4.0 + ti.random() * (COLS - 8))
                jy = 0
                while jy < ROWS - 1 and homeFilled[jy * COLS + jx] == 0:
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
                    if nx < 1 or nx >= COLS - 1 or ny < 1 or ny >= ROWS - 2 or homeFilled[ny * COLS + nx] == 0:
                        break
                    if nx != cx2 or ny != cy2:
                        cut_between(cx2, cy2, nx, ny)
                        cx2 = nx
                        cy2 = ny
                    s += 1
                jn += 1
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

    # связи между соседними блоками
    c = 0
    while c < N:
        a = homeToBlock[c]
        if a >= 0:
            if (c % COLS) < COLS - 1 and homeToBlock[c + 1] >= 0:
                intact = 1 if cutR[c] == 0 else 0
                add_bond_func(a, homeToBlock[c + 1], 0, intact)
            if (c // COLS) < ROWS - 1 and homeToBlock[c + COLS] >= 0:
                intact = 1 if cutD[c] == 0 else 0
                add_bond_func(a, homeToBlock[c + COLS], 1, intact)
        c += 1


@ti.kernel
def generate_solid():
    """Полностью заполненный массив породой (монолит без полостей)."""
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
    for c in range(N):
        homeFilled[c] = 0
        homeToBlock[c] = -1
        cutR[c] = 0
        cutD[c] = 0
    for c in range(N):
        hBond[c] = -1
        vBond[c] = -1
    for s in range(FRIC_SLOTS):
        fricKey[s] = -1
        fricS[s] = 0.0
        fricF[s] = 0
    c = 0
    while c < N:
        fixed = 1 if c // COLS >= ROWS - P[None].fixedRows else 0
        add_block_func(c, fixed)
        c += 1
    c = 0
    while c < N:
        a = homeToBlock[c]
        if a >= 0:
            if (c % COLS) < COLS - 1 and homeToBlock[c + 1] >= 0:
                add_bond_func(a, homeToBlock[c + 1], 0, 1)
            if (c // COLS) < ROWS - 1 and homeToBlock[c + COLS] >= 0:
                add_bond_func(a, homeToBlock[c + COLS], 1, 1)
        c += 1


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
        dx = bx[i] - ((bhome[i] % COLS) + 0.5)
        dy = by[i] - ((bhome[i] // COLS) + 0.5)
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
            st = 0.0
            if bondAxis[bd] == 0:
                st = ti.abs(bx[b] - bx[a] - 1.0)
            else:
                st = ti.abs(by[b] - by[a] - 1.0)
            ti.atomic_max(dmax[None], st)
    return dmax[None]


@ti.kernel
def clear_scene():
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
                bhome[n] = bhome[i]
                bshade[n] = bshade[i]
                bstress[n] = bstress[i]
                bstressC[n] = bstressC[i]
                brot[n] = brot[i]
                bw[n] = bw[i]
                btor[n] = btor[i]
                nbond[n] = nbond[i]
            homeToBlock[bhome[n]] = n
            n += 1
        else:
            homeFilled[bhome[i]] = 0
            homeToBlock[bhome[i]] = -1
        i += 1
    bcount[None] = n
    for c in range(N):
        hBond[c] = -1
        vBond[c] = -1
    for s in range(FRIC_SLOTS):
        fricKey[s] = -1
        fricS[s] = 0.0
        fricF[s] = 0
    bondCount[None] = 0
    c = 0
    while c < N:
        if homeFilled[c] == 1:
            a = homeToBlock[c]
            if (c % COLS) < COLS - 1 and homeFilled[c + 1] == 1:
                add_bond_func(a, homeToBlock[c + 1], 0, 1)
            if (c // COLS) < ROWS - 1 and homeFilled[c + COLS] == 1:
                add_bond_func(a, homeToBlock[c + COLS], 1, 1)
        c += 1
    pCount[None] = 0
    dustCount[None] = 0
    broken[None] = 0


@ti.kernel
def apply_fixed_rows():
    for i in range(bcount[None]):
        hy = bhome[i] // COLS
        bfixed[i] = 1 if hy >= ROWS - P[None].fixedRows else 0
        if bfixed[i] == 1:
            bvx[i] = 0.0
            bvy[i] = 0.0


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
                    if homeFilled[cell] == 0 and bcount[None] < MAXB:
                        fixed = 1 if ay >= ROWS - P[None].fixedRows else 0
                        i = add_block_func(cell, fixed)
                        if ax > 0 and homeToBlock[cell - 1] >= 0:
                            add_bond_func(homeToBlock[cell - 1], i, 0, 1)
                        if ax < COLS - 1 and homeToBlock[cell + 1] >= 0:
                            add_bond_func(i, homeToBlock[cell + 1], 0, 1)
                        if ay > 0 and homeToBlock[cell - COLS] >= 0:
                            add_bond_func(homeToBlock[cell - COLS], i, 1, 1)
                        if ay < ROWS - 1 and homeToBlock[cell + COLS] >= 0:
                            add_bond_func(i, homeToBlock[cell + COLS], 1, 1)
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
                bondIntact[bd] = 0
                set_bond_edge(bondA[bd], bondB[bd], bondAxis[bd], 0)
                ti.atomic_add(broken[None], 1)
                breakCrush[None] = 0
                if bondAxis[bd] == 0:
                    bvx[bondA[bd]] -= 0.15
                    bvx[bondB[bd]] += 0.15
                else:
                    bvy[bondA[bd]] -= 0.15
                    bvy[bondB[bd]] += 0.15


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


# ================================================================== главный цикл
DT = 1.0 / (60.0 * 48)
SUBSTEPS = 48

def place_source(x, y):
    if srcCount[None] >= 4:
        return "Maksimum 4 istochnika"
    s = srcCount[None]
    srcX[s] = x
    srcY[s] = y
    srcPh[s] = ti.random() if False else (time.time() % 1.0)
    srcCount[None] += 1
    inside = (x - cavX[None]) / cavRX[None]
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
        phys_bonds()
        phys_contact(dt)
        phys_integrate(dt)
    cleanup_fallen()
    if frameNo[None] % 30 == 0:
        compact_bonds()
    dust_step()
    dust_compact()


def do_stroke(mx, my, last_mx, last_my, tool, brush):
    dx = mx - last_mx
    dy = my - last_my
    dist = math.hypot(dx, dy)
    steps = max(1, int(math.ceil(dist / 0.35)))
    r_rock = brush / 2.0 + 0.4
    r_crack = brush / 2.0 + 0.25
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
