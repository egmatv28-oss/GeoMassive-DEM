"""ГЕОМАССИВ/2D · DEM — состояние модели и конфигурация (physstate)

Единая точка хранения СОСТОЯНИЯ решателя: константы, физический масштаб (СИ),
параметры породы/воды (Params) и ВСЕ taichi-поля. Сюда же вынесены низкоуровневые
примитивы (создание/удаление блоков и связей, частицы воды, разрыв связей,
трение), т.к. они нужны и расчёту взаимодействий (interactions.py), и оркестратору
тиков (solver.py), и генератору/отрисовке (render.py).

Порядок импорта: physstate -> interactions -> solver -> render.
Здесь НЕТ парных расчётов и оркестрации тиков — только состояние и примитивы.
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

# offline_cache=False: тайчи не пишет/не лочит кеш компиляции ядер. Иначе при
# нескольких запусках программы параллельно возникает гонка за ticache.lock
# и предупреждение "[W] Lock ... failed". Кеш экономит только повторные запуски,
# а решатель один — теряем немного, зато без инструктур.
ti.init(arch=ti.vulkan, offline_cache=False)

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
MAXNBR = 6   # статический радиус поиска соседей контакта (в клетках);
             # покрывает блоки до размеров MAXNBR*2 клеток рядом; динамический
             # радиус фильтруется в рантайме (см. phys_contact). 6 — с запасом на
             # УГЛЫ повёрнутого квадрата (brot): проекция r·(|cos|+|sin|) длиннее
             # радиуса вписанного круга (r·√2 на 45°), поэтому углы зацепляются
             # за соседей дальше, чем круглый "радиус".

# ================================================================ физический масштаб (СИ)
# ВНУТРЕННЯЯ СИСТЕМА ЕДИНИЦ — СИ: позиции и размеры блоков/воды в МЕТРАХ,
# скорости в м/с, силы в НЬЮТОНАХ, масса в КИЛОГРАММАХ, время в СЕКУНДАХ.
# Клетки (COLS/ROWS) нужны только для пространственного хэша, генерации модели
# и рендера. LCELL — РАЗМЕР КЛЕТКИ В МЕТРАХ: это параметр дискретизации, его
# можно менять свободно (0.25, 0.5, 1.0 ...) — физика не изменится, изменится
# только зерно сетки. Порода — песчаник средней прочности.
LCELL = 0.5                 # м на клетку (параметр дискретизации, свободно настраивается)
RHO_R = 2500.0              # плотность породы, кг/м3
RHO_W = 1000.0              # плотность воды, кг/м3
G_PHYS = 9.81               # ускорение свободного падения, м/с2
TICK = 1.0 / 60.0           # с: время модели за один тик (единица расчёта времени)
# производные величины (СИ, из базовых констант):
BMASS_UNIT = RHO_R * LCELL * LCELL   # кг: масса блока размером 1 клетку (толщина 1 м)
FT_SIGMA = 1e6              # Па: 1 МПа прочности -> Н на метр ширины контакта
LOAD_OVER = RHO_R * G_PHYS * LCELL   # Н: вес столба породы глубиной 1 м (на столб шириной 1 клетку)
WPRESS = RHO_W * G_PHYS * LCELL      # Н на 1 м напора: сила давления на грань блока 1 клетку
WSPR = 2000.0 * RHO_R * LCELL        # Н/м: жёсткость упругого контакта вода-блок
VMAX = 30.0 * LCELL         # м/с: предельная скорость блока (кламп)
WVMAX = 24.0 * LCELL        # м/с: предельная скорость водяной частицы
CONTACT_CLOSE_TOL = 0.08 * LCELL     # м: зазор смыкания трещины (передача бокового сжатия)
DT_FLUID = 0.25 * TICK      # с: шаг флюида (3 подшага = 0.75 тика на тик)
NFLUID_STEPS = 3            # число подшагов флюида за тик
# ускорение флюида: за тик флюид интегрируется 0.75 тика, поэтому чтобы дать
# ровно G_PHYS, нужно fwg*N*DT = G_PHYS*TICK:
FWG_BASE = G_PHYS * TICK / (NFLUID_STEPS * DT_FLUID)   # м/с2

# параметры флюида (СИ; wg динамический, хранится в поле)
H_SPH = 0.72 * LCELL        # м: радиус SPH-ядра
RHO0 = 2.6
KP = 0.32
KN = 0.64
RP_SPH = 0.28 * LCELL       # м: радиус частицы воды

PI = 3.14159265

K0_LATERAL = 1.0            # коэффициент бокового давления (σ_h = K0·σ_v); 1.0 — гидростатический распор глубокого массива
# коэффициент бокового давления ОБРУШЕННОЙ (рыхлой) породы: меньше, чем у массива
# (активное давление Ka ~ tan²(45°−φ/2); для φ≈30° это ~0.33). Обломки давят друг
# на друга и на массив слабее, но всё равно дают боковой распор.
K0_RUBBLE = 0.33
# БАЗОВОЕ боковое сжатие, действующее ВСЕГДА (доля от собств. веса g): даже когда
# вертикальная перегрузка σ_v ≈ 0 (верхняя кромка, открытая трещина, рельеф), блоки
# всё равно должны давить друг на друга вбок и передавать горизонтальную силу
# (аналог распора кучи/арки без веса сверху). Добавляется к σ_h = K0·σ_v.
LATERAL_FLOOR = 0.25   # доля g: условное "собственное боковое давление"

# УЧЁТ ГРАНЕЙ КУБА (вместо идеального круга). Контакты блок-блок считаются ТОЧНО
# для повёрнутых квадратов (SAT + клиппинг граней, phys_contact): нормальная сила
# прикладывается в РЕАЛЬНЫХ точках контакта, жёсткость пропорциональна ШИРИНЕ
# контакта (площадь в 2D), а эксцентриситет точек относительно ЦТ даёт честный
# опрокидывающий/возвращающий момент. Блок на краю опоры реально валится.
# Решатель перекатывания phys_rock остался только для ПОЛА (пол — не блок, его
# реакция приложена к центру и момента не даёт):
#   ROCK_LIFT — момент тяжести вокруг нижнего ребра: τ = −k·m·g·r·(cosφ−sin|φ|)·tanh(10φ).
#   Максимум при малом φ (куб возвращается на грань), НОЛЬ на ребре (45°) — неустойчивое
#   равновесие, куб падает на следующую грань. Это честное смещение ЦТ реального куба.
ROCK_LIFT = 0.8             # доля m·g·r: момент перекатывания вокруг нижнего ребра (пол)
ROLL_MU = 0.35              # коэффициент сопротивления ПЕРЕКАТЫВАНИЮ: момент =
                            # μ_r·Fn·r·tanh(bw·2), Fn — реальная нормаль контакта.
                            # Точки контакта теперь настоящие (SAT), и трение проходит
                            # ЧЕРЕЗ точку опоры — ложного раскручивания куба нет, поэтому
                            # μ_r можно держать умеренным: он гасит вечное вращение, но НЕ
                            # удерживает блок от опрокидывания, когда ЦТ вышел за опору
                            # (в этом случае comIn==0 и μ_r дополнительно ослабляется).


# apply_gravity / apply_rock / apply_water (настройка параметров породы и воды)
# живут в solver.py — это API движка, не состояние.

# ------------------------------------------------------------------ параметры
@ti.dataclass
class Params:
    g: ti.f32
    E: ti.f32    # модуль Юнга, Н/м2 (жёсткости связей/контактов считаются из него и геометрии)
    rho: ti.f32  # кг/м3: текущая плотность породы (массы блоков пересчитываются)
    cn: ti.f32   # ζ СВЯЗИ: относительное демпфирование (доля критического) упругой связи
    mu: ti.f32
    rp: ti.f32
    ftScale: ti.f32
    rcFactor: ti.f32
    fsFactor: ti.f32
    kw: ti.f32   # Н/м: жёсткость упругого контакта вода-блок
    wCap: ti.f32 # Н: максимальная сила давления воды на блок (напор)
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
    cnc: ti.f32   # ζ КОНТАКТА блоков (отдельно от связей) — гасит подпрыгивание обломков
    cncR: ti.f32  # ζ КОНТАКТА ДВУХ СВОБОДНЫХ (несвязанных) блоков — перекритическое:
                  # рухляк поглощает удары, обломки не "пружинят" как резина
    maxSize: ti.i32   # максимальный размер блока в модели (радиус поиска соседей, клетки)
    plastYield: ti.f32   # доля σ_раст (tension), с которой начинается пластическое течение (предел текучести)
    plastLimit: ti.f32   # предельная пластическая деформация (доля длины покоя) — после неё обрыв связи
    plastFlow: ti.f32    # скорость пластического течения: рост остаточной длины за секунду на единицу перегрузки
    creepRecover: ti.f32 # 1/с: скорость залечивания криповой деформации при снятии нагрузки
    mcBond: ti.f32   # коэффициент трения Мора-Кулона ШВА: сдвиговая прочность связи
                     # растёт с обжатием по шву (σ_n<0) и падает при растяжении (σ_n>0):
                     # Fs_cap = Fs0 + mcBond·σ_n. Честный Mohr-Coulomb на связях.


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
dbgFt = ti.field(ti.f32, ())
dbgFnf = ti.field(ti.f32, ())
dbgEs = ti.field(ti.f32, ())
dbgVt = ti.field(ti.f32, ())
dbgI = ti.field(ti.i32, ())
dbgJ = ti.field(ti.i32, ())
dbgAxis = ti.field(ti.i32, ())
dbgVx0 = ti.field(ti.f32, ())
dbgVx1 = ti.field(ti.f32, ())
dbgPairs = ti.field(ti.i32, ())
dbgContacts = ti.field(ti.i32, ())
dbgFixed = ti.field(ti.i32, ())
# 1 = досыпка обломков сверху (refill_top_overburden) отключена ("Ochistit porodu")
refillOff = ti.field(ti.i32, ())
brot = ti.field(ti.f32, MAXB)
bw = ti.field(ti.f32, MAXB)
btor = ti.field(ti.f32, MAXB)
nbond = ti.field(ti.i32, MAXB)
hasL = ti.field(ti.i32, MAXB)   # 1 = есть целая БОКОВАЯ связь слева (ось ~горизонтальная, сосед слева)
hasR = ti.field(ti.i32, MAXB)   # 1 = есть целая БОКОВАЯ связь справа (сосед справа)
# геометрические флаги граней/трещин/опоры (размер-осведомлённые, вместо occ_at в силовом
# контуре). Вычисляются в phys_contact из ФАКТИЧЕСКИХ расстояний между центрами соседей:
#   faceL[i]=0 — слева есть сосед (в пределах lat+GAP, тот же "этаж") → грань НЕ свободна
#   faceR[i]=0 — справа есть сосед (по умолчанию faceL=faceR=1, т.е. грань свободна)
#   crackL[i]=1 — слева есть сосед, но БЕЗ связи → внутренняя трещина (сжать шов)
#   crackR[i]=1 — справа есть сосед без связи
#   supp[i]=1   — снизу есть блок (реальная опора для сна/демпфирования)
faceL = ti.field(ti.i32, MAXB)
faceR = ti.field(ti.i32, MAXB)
crackL = ti.field(ti.i32, MAXB)
crackR = ti.field(ti.i32, MAXB)
supp = ti.field(ti.i32, MAXB)
# интервал ОПОРЫ по x (метры): границы области реальных нижних контактов блока
# (собирается в phys_contact из точек контакта с upward-нормалью). supLo>supHi — опоры нет.
supLo = ti.field(ti.f32, MAXB)
supHi = ti.field(ti.f32, MAXB)
# 1 = проекция ЦЕНТРА ТЯЖЕСТИ попадает в интервал опоры (блок устойчив: может "спать",
# получает полный тормоз перекатывания). 0 = ЦТ вне опоры — блок ДОЛЖЕН завалиться:
# сон запрещён, восстанавливающие моменты не действуют, тормоз перекатывания ослаблен.
comIn = ti.field(ti.i32, MAXB)
inContact = ti.field(ti.i32, MAXB)   # 1 = блок участвует ХОТЯ БЫ В ОДНОМ контакте в этот
                                     # подшаг (любое перекрытие). Собирается в phys_contact;
                                     # phys_rock гасит вращение при ЛЮБОМ контакте (не только
                                     # при опоре снизу) — зажатый с боков куб не крутится.
cnck = ti.field(ti.f32, MAXB)        # максимальная нормальная сила контакта блока за подшаг
                                     # (Н). С ней масштабируется сопротивление перекатыванию
                                     # (как в реальности: сильнее прижат — сильнее тормоз).
bsz = ti.field(ti.f32, MAXB)     # размер блока (сторона квадрата), МЕТРЫ
bmass = ti.field(ti.f32, MAXB)   # масса блока, кг = RHO_R * bsz^2 (толщина 1 м)
hx = ti.field(ti.f32, MAXB)      # начальная позиция (для диагностики смещения), м
hy = ti.field(ti.f32, MAXB)
bcount = ti.field(ti.i32, shape=())

# сетка верхней нагрузки (только по колонкам)
topYArr = ti.field(ti.f32, COLS)
topCol = ti.field(ti.i32, COLS)
# индекс досыпанного за текущий тик обломка для каждой колонки (-1 = нет),
# чтобы связывать между собой только НОВЫЕ обломки (не со старым массивом)
newRubble = ti.field(ti.i32, COLS)

# топология массива: занятость клетки блоком + вертикальная перегрузка (сверху вниз)
gOcc = ti.field(ti.i32, N)
sigVI = ti.field(ti.f32, N)
# горизонтальное удержание: holdL — давление, удерживающее блок слева (σ_h от левого
# соседа), holdR — справа. Равновесие: holdL == holdR (сумма проекций = 0).
holdL = ti.field(ti.f32, N)
holdR = ti.field(ti.f32, N)

# связи (bonds) — фиксированные массивы, длина покоя и направление у каждой
bondA = ti.field(ti.i32, MAXBONDS)
bondB = ti.field(ti.i32, MAXBONDS)
bondIntact = ti.field(ti.i32, MAXBONDS)
bondDmg = ti.field(ti.f32, MAXBONDS)
bondR = ti.field(ti.f32, MAXBONDS)
bondJ = ti.field(ti.f32, MAXBONDS)
brest = ti.field(ti.f32, MAXBONDS)      # длина покоя связи
bondPlast = ti.field(ti.f32, MAXBONDS)  # накопленная ПЛАСТИЧЕСКАЯ вытяжка связи (клетки)
bondSlip = ti.field(ti.f32, MAXBONDS)   # накопленное ПЛАСТИЧЕСКОЕ проскальзывание (клетки) —
                                           # сдвиговая текучесть: связь реально "ползёт" по шву,
                                           # блок отодвигается, а не висит на упругом сдвиге (Баг 7)
bondFlow = ti.field(ti.i32, MAXBONDS)   # 1 = связь сейчас "течёт" пластически (для отрисовки фиолетовым)
bondCreep = ti.field(ti.f32, MAXBONDS)  # криповое "повреждение" (клетки): рвёт только при УДЕРЖИВАЕМОЙ нагрузке,
                                        # залечивается при разгрузке (кратковременный перегруз не рвёт)
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
fricF = ti.field(ti.f32, FRIC_SLOTS)   # момент последнего контакта (simT, с)

simT = ti.field(ti.f32, shape=())       # внутреннее время симуляции, с (не зависит от кадров)
compactT = ti.field(ti.f32, shape=())   # simT последней пересборки связей, с
loadCur = ti.field(ti.f32, shape=())
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
    c = int(x / LCELL)
    if c < 0:
        c = 0
    elif c >= COLS:
        c = COLS - 1
    return c


@ti.func
def block_prj(i: ti.i32) -> ti.f32:
    """Половина проекции КВАДРАТА (AABB) на ось с учётом поворота brot: r·(|cos|+|sin|).

    Рост проекции — не резкий, а ГЛАДКИЙ (smoothstep по отклонению φ от ближайшей
    грани 0..45°): при малом наклоне проекция почти равна вписанному радиусу r
    (контакт как раньше — куча не встряхивается жёсткой пружиной от каждого
    подшага вращения), при сильном наклоне стремится к r·√2 — углы куба
    вылезают за круг и по-настоящему цепляются за соседей, пол и стены.
    Так учёт ГРАНЕЙ работает основным у опрокидывающихся блоков, не расшатывая
    стабильный массив."""
    r = bsz[i] * 0.5
    half = PI * 0.5
    q = brot[i] / half
    q = q - ti.floor(q)
    if q > 0.5:
        q -= 1.0
    t = ti.abs(q) * 2.0          # 0 = на грани, 1 = на углу (45°)
    u = t * t * (3.0 - 2.0 * t)  # smoothstep: гладкое включение углов
    return r * (1.0 + 0.4142 * u)


@ti.func
def row_cell(y: ti.f32) -> ti.i32:
    c = int(y / LCELL)
    if c < 0:
        c = 0
    elif c >= ROWS:
        c = ROWS - 1
    return c


@ti.func
def occ_at(x: ti.f32, y: ti.f32) -> ti.i32:
    """Занята ли клетка под точкой (x, y) (метры) блоком (1 = есть блок, 0 = пусто)."""
    return gOcc[row_cell(y) * COLS + clamp_cell(x)]


@ti.kernel
def refresh_field():
    """Пересчитать топологию и поля напряжений.

    1) Занятость клеток блоком — по текущим позициям (без геометрии полости).
    2) Вертикальная перегрузка в каждой занятой клетке: нагрузка от поверхности
       + вес всего столба блоков НАД ней (сверху вниз).
    3) Горизонтальное удержание двумя встречными проходами (сверху вниз):
       блок держится слева/справа с давлением σ_h = K0·σ_v, ЕСЛИ сосед есть;
       у свободной грани (выработка) с этой стороны удержания нет.
    """
    for c in range(N):
        gOcc[c] = 0
        sigVI[c] = 0.0
        holdL[c] = 0.0
        holdR[c] = 0.0
    for i in range(bcount[None]):
        cxi = clamp_cell(bx[i])
        cyi = row_cell(by[i])
        gOcc[cyi * COLS + cxi] = 1
    for x in range(COLS):
        # loadCur — уже Н (вес породы над колонкой шириной 1 клетку)
        carry = loadCur[None]
        y = 0
        while y < ROWS:
            c = y * COLS + x
            if gOcc[c] == 1:
                sigVI[c] = carry
                # вес клетки: плотность однородна, поэтому любая клетка (и любой
                # блок, покрывающий клетку) даёт одинаковый вклад BMASS_UNIT*g, Н
                carry += BMASS_UNIT * P[None].g
            y += 1
    for y in range(ROWS):
        # слева-направо: удержание справа (σ_h от правого соседа)
        for x in range(COLS):
            c = y * COLS + x
            if gOcc[c] == 1 and x + 1 < COLS and gOcc[y * COLS + x + 1] == 1:
                holdR[c] = sigVI[c] * P[None].k0
        # справа-налево: удержание слева (σ_h от левого соседа)
        for j in range(COLS):
            x = COLS - 1 - j
            c = y * COLS + x
            if gOcc[c] == 1 and x - 1 >= 0 and gOcc[y * COLS + x - 1] == 1:
                holdL[c] = sigVI[c] * P[None].k0


@ti.func
def init_block(i: ti.i32, x: ti.f32, y: ti.f32, size: ti.f32, fixed: ti.i32):
    bx[i] = x * LCELL
    by[i] = y * LCELL
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
    bsz[i] = size * LCELL
    bmass[i] = P[None].rho * bsz[i] * bsz[i]
    hx[i] = x * LCELL
    hy[i] = y * LCELL


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
    """Связать блоки a и b с длиной покоя rest (в КЛЕТКАХ, переводится в метры)."""
    if bondCount[None] < MAXBONDS:
        i = ti.atomic_add(bondCount[None], 1)
        bondA[i] = a
        bondB[i] = b
        bondIntact[i] = intact
        bondDmg[i] = 0.0
        bondR[i] = 0.0
        bondJ[i] = (ti.random() - 0.5) * 0.3
        brest[i] = rest * LCELL
        bondPlast[i] = 0.0
        bondSlip[i] = 0.0
        bondFlow[i] = 0
        bondCreep[i] = 0.0
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
    # очистить память трения о жёстких границах для удаляемого и переносимого
    # индекса — блок, занявший слот, не должен получить устаревший остаточный сдвиг
    s = fric_find(-(rem + 1))
    if s >= 0:
        fricKey[s] = -1
        fricS[s] = 0.0
        fricF[s] = 0
    s = fric_find(-(rem + 1) - FRIC_SLOTS)
    if s >= 0:
        fricKey[s] = -1
        fricS[s] = 0.0
        fricF[s] = 0
    s = fric_find(-(last + 1))
    if s >= 0:
        fricKey[s] = -1
        fricS[s] = 0.0
        fricF[s] = 0
    s = fric_find(-(last + 1) - FRIC_SLOTS)
    if s >= 0:
        fricKey[s] = -1
        fricS[s] = 0.0
        fricF[s] = 0
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
        bmass[rem] = bmass[last]
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
            bondPlast[n] = bondPlast[i]
            bondSlip[n] = bondSlip[i]
            bondFlow[n] = bondFlow[i]
            bondCreep[n] = bondCreep[i]
            bondNX[n] = bondNX[i]
            bondNY[n] = bondNY[i]
            bondE[n] = bondE[i]
            n += 1
        i += 1
    bondCount[None] = n


@ti.func
def spawn_particle(x: ti.f32, y: ti.f32, vm: ti.f32, a0: ti.f32):
    """Создать частицу воды: x, y — в КЛЕТКАХ (переводятся в метры), vm — м/с."""
    if pCount[None] < PMAX:
        i = pCount[None]
        pCount[None] += 1
        wx[i] = x * LCELL + (ti.random() - 0.5) * 0.35 * LCELL
        wy[i] = y * LCELL + (ti.random() - 0.5) * 0.35 * LCELL
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
        dustVX[d] = (ti.random() - 0.5) * 0.1 * LCELL
        dustVY[d] = (ti.random() - 0.5) * 0.1 * LCELL
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
        bondFlow[bd] = 0


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
        # Приведённая масса из реальных масс блоков (кг):
        ma = bmass[a]
        mb = bmass[b]
        mred = ma * mb / (ma + mb + 1e-9)
        # В хрупком разрушении почти вся энергия уходит в поверхность/тепло:
        # в кинетическую уходит малая доля. Пинок держим СКРОМНЫМ (eta = 0.08):
        # при большом пинке берега на подшаг раскрываются, контакт Fn обнуляется
# и пропадает трение, скрепляющее берега — трещина сразу уходит вверх
        # (именно диагностированный пробой). Пинок чуть увеличен (eta=0.14).
        eta = 0.14
        E_kin = E_acc * eta
        dv = ti.sqrt(2.0 * E_kin / mred)

        dv_n = dv * 0.7
        dv_s = dv * 0.3

        sgn_n = 1.0 if strain > 0.0 else -1.0
        # Сохранение импульса: относительная скорость расхождения dv задаётся
        # приведённой массой (E_kin = ½·mred·dv²), но распределяется по блокам
        # обратно пропорционально массам (Δv_a = −mb/(ma+mb)·dv, Δv_b = +ma/(ma+mb)·dv).
        # Симметричный 0.5/0.5 корректен только при ma==mb; при перепаде масс он
        # рождал импульс и в (ma+mb)²/(4·ma·mb) раз больше кинетической энергии,
        # чем заложено бюджетом η·E_acc. Теперь Δp=0 и ΔE=½·mred·dv²=E_kin.
        w_a = mb / (ma + mb + 1e-9)     # доля ускорения для блока a
        w_b = ma / (ma + mb + 1e-9)     # доля ускорения для блока b
        bvx[a] -= ux * sgn_n * dv_n * w_a
        bvy[a] -= uy * sgn_n * dv_n * w_a
        bvx[b] += ux * sgn_n * dv_n * w_b
        bvy[b] += uy * sgn_n * dv_n * w_b

        sgn_s = 1.0 if shear > 0.0 else -1.0
        bvx[a] -= px * sgn_s * dv_s * w_a
        bvy[a] -= py * sgn_s * dv_s * w_a
        bvx[b] += px * sgn_s * dv_s * w_b
        bvy[b] += py * sgn_s * dv_s * w_b

        if ti.abs(relb) > 1e-6:
            Ia = bmass[a] * bsz[a] * bsz[a] / 6.0
            Ib = bmass[b] * bsz[b] * bsz[b] / 6.0
            Ired = Ia * Ib / (Ia + Ib + 1e-9)
            E_rot = E_acc * 0.15
            dw = ti.sqrt(2.0 * E_rot / Ired)
            sgr = 1.0 if relb > 0.0 else -1.0
            # то же распределение по моментам инерции: Δω_a = −Ib/(Ia+Ib)·dw,
            # Δω_b = +Ia/(Ia+Ib)·dw — сохранение углового импульса пары
            bw[a] -= sgr * dw * (Ib / (Ia + Ib + 1e-9))
            bw[b] += sgr * dw * (Ia / (Ia + Ib + 1e-9))


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
def fric_find(key: ti.i32) -> ti.i32:
    """Найти слот пары БЕЗ его создания (для очистки). -1 если нет."""
    h = key ^ (key >> 12)
    h = h ^ (h >> 6)
    h = h & FRIC_MASK
    s = h
    res = -1
    guard = 0
    while guard < FRIC_SLOTS:
        k = fricKey[s]
        if k == key:
            res = s
            break
        elif k == -1:
            break
        s = (s + 1) & FRIC_MASK
        guard += 1
    return res


@ti.func
def friction_against(i: ti.i32, vt: ti.f32, Fn: ti.f32, dt: ti.f32, key: ti.i32, kts: ti.f32, axis: ti.i32):
    """Кулоновское трение свободного блока о ЖЁСТКУЮ границу домена (пол/стены).
    Опора неподвижна, поэтому vt — просто скорость блока; пружинная модель та же,
    что в friction_apply. axis 0 — трение вдоль x (пол), 1 — вдоль y (стороны)."""
    s = fric_slot(key)
    if s >= 0:
        e_s = fricS[s]
        e_f = fricF[s]
        if simT[None] - e_f > FRIC_FORGET_T:
            e_s = 0.0
        fricF[s] = simT[None]
        e_s += vt * dt
        Ft = -kts * e_s
        Fm = P[None].mu * Fn
        if Ft > Fm:
            Ft = Fm
        elif Ft < -Fm:
            Ft = -Fm
        fricS[s] = -Ft / kts
        if axis == 0:
            ti.atomic_add(bfx[i], Ft)
        else:
            ti.atomic_add(bfy[i], Ft)


@ti.func
def friction_vec(i: ti.i32, j: ti.i32, key: ti.i32, vt: ti.f32, Fn: ti.f32, dt: ti.f32,
                  tx: ti.f32, ty: ti.f32, px: ti.f32, py: ti.f32, kts: ti.f32):
    """Кулоновское трение пары блоков В ТОЧКЕ КОНТАКТА (произвольное направление).

    (tx, ty) — единичная касательная, (px, py) — точка контакта, vt — относительная
    касательная скорость точки j относительно i, Fn — нормальная сила точки (верхний
    предел трения μ·Fn), kts — жёсткость пружины памяти сдвига (Н/м).
    Сила прикладывается ровно в точке контакта, поэтому эксцентриситет относительно
    ЦТ даёт ФИЗИЧЕСКИЙ момент трения (r × F) — без осевых приближений.
    Память сдвига (fricS) — своя на каждую точку контакта (key различает точки)."""
    s = fric_slot(key)
    if s >= 0:
        e_s = fricS[s]
        # Пружина трения ЧИСТО накапливает сдвиг (без затухания за подшаг) и сбрасывается
        # только при потере контакта дольше FRIC_FORGET_T: статическое трение держит
        # обломки на месте, а не превращается в вязкое ползучее сопротивление.
        if simT[None] - fricF[s] > FRIC_FORGET_T:
            e_s = 0.0
        fricF[s] = simT[None]
        e_s += vt * dt
        Ft = -kts * e_s
        Fm = P[None].mu * Fn
        if Ft > Fm:
            Ft = Fm
        elif Ft < -Fm:
            Ft = -Fm
        fricS[s] = -Ft / kts
        ftx = Ft * tx
        fty = Ft * ty
        ti.atomic_add(bfx[i], -ftx)
        ti.atomic_add(bfy[i], -fty)
        ti.atomic_add(bfx[j], ftx)
        ti.atomic_add(bfy[j], fty)
        rix = px - bx[i]
        riy = py - by[i]
        rjx = px - bx[j]
        rjy = py - by[j]
        if nbond[i] == 0:
            # τ = r × F, F_i = −Ft·t
            ti.atomic_add(btor[i], -Ft * (rix * ty - riy * tx))
        if nbond[j] == 0:
            ti.atomic_add(btor[j], Ft * (rjx * ty - rjy * tx))

SUBSTEPS = 192
DT = TICK / SUBSTEPS              # с: подшаг интегрирования DEM внутри тика
# ПРЕДЕЛ УСТОЙЧИВОСТИ явной схемы (симплектический Эйлер): частота пары
# ω=√(k/m_red) должна давать ω·dt ≤ ~0.5, тогда худшая мода решётки (цепочка
# пружин + вращение, до ~4×ω пары) остаётся внутри границы устойчивости ω·dt<2.
# Жёсткость любой пружины (связь, контакт, трение о пол) сверху ограничивается:
#   k ≤ STAB_C · m_red / dt²
# Без этого жёсткие связи (E=30 ГПа на блоках 0.5 м: k=3e10 Н/м, ω·dt≈2.06)
# САМОВОЗБУЖДАЛИСЬ: силы в связях росли ×1.6 за подшаг и массив разрывал сам
# себя на первом же тике. Кап не меняет ПРОЧНОСТЬ (Ftmax/Fcmax от k не зависят),
# лишь делает пружину чуть мягче (деформации и так микроскопические).
STAB_C = 0.25
TOPO_EVERY = 48   # пересборка топологии (хэш, nbond, поля напряжений) каждые N подшагов:
                  # слишком частая (12) усиливает вертикальные трещины сверху при большой нагрузке,
                  # 48 — баланс между свежестью топологии и стабильностью
FRIC_FORGET_T = 3.0 * TICK        # с: как долго "помнить" накопленный сдвиг трения
COMPACT_EVERY_T = 0.5             # с: период пересборки/уплотнения связей
# Демпфирование скорости и вращения блоков (задумано НА ТИК, применяется на подшаг —
# в phys_integrate множители пересчитываются как base^(dt/TICK), см. там):
DAMP_FREE_TICK = 0.9995    # свободный полёт обломка: почти без гашения (реальное падение)
DAMP_SUPP_TICK = 0.9975    # связанный/опирающийся блок: оседание, устойчивость массива
DAMP_ROT_FREE_TICK = 0.992 # вращение свободного обломка (кувыркание)
DAMP_ROT_BOND_TICK = 0.97  # вращение блока в составе массива
SLEEP_V2 = 1.5e-3 * LCELL * LCELL     # (м/с)²: порог скорости "уснувшего" блока
SLEEP_A2 = 4.0 * LCELL * LCELL        # (м/с²)²: порог ускорения — блок спит только
                                  # если почти неподвижен И силы уравновешены (лежит на опоре), иначе
                                  # падающий/подпрыгнувший блок не замораживается в воздухе

# ------------------------------------------------------------------ энергия и центр тяжести
# Интегралы энергии всей системы и центра тяжести (масс) подвижных блоков.
# Считаются в solver.compute_energy() / solver.center_of_mass() после каждого тика,
# чтобы «закон сохранения энергии» и «центр тяжести» были доступны наружу и
# проверялись равномерно для ВСЕХ блоков каждый тик.
benergy = ti.field(ti.f32, MAXB)          # полная энергия блока i, Дж (кинетика + вращение + потенциал + ½ упругой от связей)
energyKin = ti.field(ti.f32, shape=())    # суммарная поступательная кинетическая энергия, Дж
energyRot = ti.field(ti.f32, shape=())    # суммарная вращательная кинетическая энергия, Дж
energyPot = ti.field(ti.f32, shape=())    # суммарная гравитационная потенциальная энергия, Дж
energyEl = ti.field(ti.f32, shape=())     # суммарная упругая энергия связей, Дж
energyTotal = ti.field(ti.f32, shape=())  # полная энергия системы, Дж
comX = ti.field(ti.f32, shape=())         # центр тяжести (масс) системы, м (X)
comY = ti.field(ti.f32, shape=())         # центр тяжести (масс) системы, м (Y)
comMass = ti.field(ti.f32, shape=())      # суммарная масса подвижных блоков, кг
