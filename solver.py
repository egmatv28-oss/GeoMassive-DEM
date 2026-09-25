"""ГЕОМАССИВ/2D · DEM — решатель: тик времени и оркестрация физики (solver)

Главный модуль физического движка. Его задача — чтобы физика работала РАВНОМЕРНО
и для всех блоков одновременно (плоскопараллельно), с соблюдением закона
сохранения энергии.

Каждый ТИК (step_physics) строит единый распорядок, одинаковый для всех блоков:
  1) проходка состояния — топология, напряжённое поле, контактный хэш, связи;
   2) передача на расчёт — силы взаимодействий считаются в interactions.py
      ОДНИМ слитым ядром phys_forces (reset + bonds + contact) за подшаг;
  3) приём и обработка  — phys_integrate читает полученные силы (bfx/bfy/btor),
     интегрирует скорости/позиции, демпфирует, держит блоки в домене;
  4) сводка интегралов  — compute_energy() (полная энергия КАЖДОГО блока и всей
     системы, закон сохранения энергии) и center_of_mass() (центр тяжести)
     после каждого тика.

Решатель не знает про сцены, инструменты и GUI: модель строится СНАРУЖИ
(render.py) и загружается через build_model() либо примитивы
add_block_func / add_bond_func / spawn_particle. Состояние — в physstate.py.
Точка входа — mineudec.py.
"""

import math
import time

import numpy as np

import taichi as ti

from physstate import *
from interactions import *

def apply_gravity(scale):
    P[None].g = G_PHYS * scale
    fwg[None] = FWG_BASE * scale


def apply_rock(E_GPa, rp_MPa, mu, rho=None):
    E = E_GPa * 1e9                      # модуль Юнга, Н/м2
    P[None].E = E
    if rho is not None:
        P[None].rho = rho
        update_mass_rho()
    elif P[None].rho == 0.0:
        P[None].rho = RHO_R
    # ζ — относительное демпфирование (доля критического) пары блоков:
    # оно же используется и для связей, и для контактов (см. phys_bonds/phys_contact).
    P[None].cn = 0.28    # ζ связи: гасит колебания при жёсткой связи (в критич. единицах)
    P[None].cnc = 0.6    # ζ КОНТАКТА блоков: заметно гасит подпрыгивание обломков.
                         # Больше брать НЕЛЬЗЯ: эмпирически ζ≈0.8 и выше дают взрывной
                         # каскад при росте нагрузки, а полное критическое 2·√(k/m)
                         # рвёт связи уже при усадке.
    P[None].cncR = 1.1   # ζ контакта ДВУХ СВОБОДНЫХ блоков (оба nbond==0): перекритическое
                         # демпфирование — падающий обломок не отскакивает от рухляка.
                         # На связанный массив не влияет (там ζ=cnc).
    P[None].rp = rp_MPa
    P[None].mu = mu
    P[None].ftScale = FT_SIGMA
    P[None].rcFactor = 10.0
    P[None].fsFactor = 2.0
    # Честный Мора-Кулон на связи: сдвиговая прочность шва зависит от нормального
    # напряжения по шву (tension>0 — растяжение, tension<0 — обжатие).
    # τ_s = Fs0 + mcBond·σ_n, где σ_n = tension. Обжатие (σ_n<0) добавляет
    # прочности на сдвиг, растяжение (σ_n>0) — отнимает. mcBond — тангенс угла
    # внутреннего трения по шву (реалистичный ~tan 35°≈0.7).
    P[None].mcBond = 0.7
    P[None].k0 = K0_LATERAL
    P[None].relBend = 0.15
    P[None].dmgMax = 6.0
    P[None].dmgDecay = 0.85   # быстрее "забывает" перегрузку: дальние связи не рвутся, обвал не расползается
    P[None].brkCap = 24
    # Пластическое течение (крип) связей — как в реальной породе: выше предела
    # текучести связь медленно "течёт" (остаточная длина растёт), продолжая
    # передавать усилие и оставаясь герметичной; обрыв наступает после накопления
    # предельной пластической деформации (plastLimit).
    P[None].plastYield = 0.5   # крип (пластическая стадия) начинается с ~50% прочности —
                            # раньше (0.65), чтобы перегруз расползался постепенно, а не рвался
    P[None].plastLimit = 0.32  # обрыв при ~32% накопленной криповой деформации (было 0.24) —
                                # больше запаса до разрыва, связь дольше течёт и перераспределяет
    P[None].plastFlow = 1.6     # скорость течения (ещё быстрее), самоперераспределение успевает
    P[None].creepRecover = 0.6  # медленнее залечивание: накопленный крип дольше держится,
                                # без резких скачков, и трещина растёт равномерно


def apply_water(head_m):
    P[None].kw = WSPR
    P[None].wCap = WPRESS * head_m
    P[None].head = head_m
@ti.kernel
def phys_integrate(dt: ti.f32, dfree: ti.f32, dsupp: ti.f32, drotFree: ti.f32, drotBond: ti.f32):
    # моменты качания на полу и тормоз перекатывания (forces_rock) считаются
    # первыми, прямо перед интегрированием: им нужны supp/comIn/cnck, которые
    # только что собрал контакт того же подшага. Слитое ядро = минус один лаунч.
    forces_rock()
    walls = P[None].walls
    for i in range(bcount[None]):
        if bfixed[i] == 1:
            bvx[i] = 0.0
            bvy[i] = 0.0
            bw[i] = 0.0
            continue
        rx = block_prj(i)   # проекция повёрнутого квадрата: угол достигает пола/стен раньше круга
        # --- трение свободных блоков о жёсткие границы домена (пол/стены) ---
        # У обломка, лежащего прямо на дне, нет блока-соседа снизу, поэтому пара
        # (friction_apply) к нему не применяется; без этого пол — идеально гладкий,
        # и свободные обломки беспрепятственно раскатываются по поверхности.
        # Нормаль = собственный вес + вес столба над ним (sigVI — Н на клетку).
        kfric = P[None].E
        # предел устойчивости явной схемы для пружины трения о жёсткую границу (см. STAB_C)
        kstabF = STAB_C * bmass[i] / (dt * dt)
        if kfric > kstabF:
            kfric = kstabF
        nF = bmass[i] * P[None].g + sigVI[row_cell(by[i]) * COLS + clamp_cell(bx[i])]
        if nF < bmass[i] * P[None].g:
            nF = bmass[i] * P[None].g
        if by[i] <= rx + 0.02 * LCELL or by[i] >= (ROWS * LCELL - rx - 0.02 * LCELL):
            friction_against(i, bvx[i], nF, dt, -(i + 1), kfric, 0)
        if walls == 1:
            if bx[i] <= rx + 0.02 * LCELL or bx[i] >= (COLS * LCELL - rx - 0.02 * LCELL):
                friction_against(i, bvy[i], nF, dt, -(i + 1) - FRIC_SLOTS, kfric, 1)
        bvx[i] += bfx[i] / bmass[i] * dt
        bvy[i] += bfy[i] / bmass[i] * dt
        # Связанные и опирающиеся блоки гасятся сильно (устойчивость массива,
        # оседание обломков). Блок, летящий в пустоте — связей нет и снизу нет опоры
        # (supp==0, геометрия из phys_contact) — почти не гасится: обрушение выходит
        # динамичным, а не "вязким". Как только блок садится на опору, демпфирование
        # возвращается.
        # dfree/dsupp/drot* — множители демпфирования ЗА ПОДШАГ: базовые константы
        # (DAMP_*_TICK) задуманы на тик, но здесь применяются SUBSTEPS раз за тик,
        # поэтому шагом раньше пересчитаны как base^(dt/TICK). Иначе 192-кратное
        # применение давало бы v *= 0.9995^192 ≈ 0.908 за тик и искусственную
        # терминальную скорость падения ~3.4 клетки/с вместо реального свободного
        # падения (обломки "ползли" вниз много секунд).
        if nbond[i] == 0 and supp[i] == 0:
            bvx[i] *= dfree
            bvy[i] *= dfree
        else:
            bvx[i] *= dsupp
            bvy[i] *= dsupp
        bw[i] += btor[i] / (bmass[i] * bsz[i] * bsz[i] / 6.0) * dt
        if nbond[i] > 0:
            bw[i] *= drotBond
        else:
            bw[i] *= drotFree
        if ti.abs(bw[i]) < 2e-5:
            bw[i] = 0.0
        if bw[i] > 10.0:
            bw[i] = 10.0
        elif bw[i] < -10.0:
            bw[i] = -10.0
        brot[i] += bw[i] * dt
        v2 = bvx[i] * bvx[i] + bvy[i] * bvy[i]
        if v2 > VMAX * VMAX:
            s = VMAX / ti.sqrt(v2)
            bvx[i] *= s
            bvy[i] *= s
        elif v2 < SLEEP_V2 and comIn[i] == 1:
            # блок почти стоит, силы уравновешены И он УСТОЙЧИВ: проекция ЦТ попадает
            # в интервал реальной опоры (comIn из phys_contact — точки контакта снизу,
            # size-aware) или блок лежит на полу.
            # Блок, висящий НА КРАЮ опоры (comIn==0), не засыпает: момент от
            # эксцентриситета веса продолжает его опрокидывать — кривые "пружинящие"
            # столбы из блоков больше не замерзают в равновесии, которого нет.
            ax = bfx[i] / bmass[i]
            ay = bfy[i] / bmass[i]
            # момент перекоса (блок на краю опоры: линейные силы в равновесии, но
            # момент от Fn≠CM крутит его) не даём "уснуть" — иначе валежник навсегда
            # застревает на краю блока и не опрокидывается/не съезжает.
            angA = btor[i] / (bmass[i] * bsz[i] * bsz[i] / 6.0)
            if ax * ax + ay * ay < SLEEP_A2 and angA * angA < SLEEP_A2:
                bvx[i] = 0.0
                bvy[i] = 0.0
                bw[i] = 0.0
        bx[i] += bvx[i] * dt
        by[i] += bvy[i] * dt
        # Вязко-поглощающие границы (dashpot) вместо жёсткой стенки с обнулением
        # скорости: скорость в стенку гасится экспоненциально (v *= 1 - cnc·dt),
        # энергия не отражается обратно в модель, осцилляций на границе нет.
        # Позиция всё равно клампится в домен, так что блок не вылетает.
        damp = 2.0 * P[None].cnc * dt * ti.sqrt(P[None].E / bmass[i])
        if damp > 1.0:
            damp = 1.0
        if walls == 1:
            if bx[i] < rx:
                bx[i] = rx
                if bvx[i] < 0.0:
                    bvx[i] *= 1.0 - damp
            elif bx[i] > COLS * LCELL - rx:
                bx[i] = COLS * LCELL - rx
                if bvx[i] > 0.0:
                    bvx[i] *= 1.0 - damp
        if by[i] < rx:
            by[i] = rx
            if bvy[i] < 0.0:
                bvy[i] *= 1.0 - damp
        if by[i] > ROWS * LCELL - rx:
            by[i] = ROWS * LCELL - rx
            if bvy[i] > 0.0:
                bvy[i] *= 1.0 - damp
@ti.kernel
def count_bonds():
    for i in range(bcount[None]):
        nbond[i] = 0
        hasL[i] = 0
        hasR[i] = 0
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
            # боковая связь: ось связи ближе к горизонтали -> у левого блока
            # правая грань связана (hasR), у правого — левая грань (hasL)
            if ti.abs(bx[b] - bx[a]) >= ti.abs(by[b] - by[a]):
                if bx[b] >= bx[a]:
                    ti.atomic_max(hasR[a], 1)
                    ti.atomic_max(hasL[b], 1)
                else:
                    ti.atomic_max(hasL[a], 1)
                    ti.atomic_max(hasR[b], 1)


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
        if bfixed[i] == 0 and (by[i] > (ROWS + 6) * LCELL or bx[i] < -4 * LCELL or bx[i] > (COLS + 4) * LCELL):
            remove_block_func(i)
        i -= 1


# ------------------------------------------------------------------ верхняя нагрузка
@ti.kernel
def compute_top_load_fused(mode: ti.i32):
    # 1. обнуление ovf
    for i in range(bcount[None]):
        ovf[i] = 0.0
    # 2. нагрузка: в режиме шахты (mode==0) давление на верхний ряд прикладывается
    #    ВСЕГДА — сразу полным весом породы над схемой (depth), без плавного
    #    нарастания. В остальных режимах (склон/мозаика) нагрузки сверху нет.
    if mode == 0:
        loadCur[None] = P[None].depth * LOAD_OVER
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
    # 4. приложение нагрузки: полный вес давит на верхний блок каждой колонки.
    #    loadCur — уже Н (вес породы над колонкой шириной 1 клетку).
    f = loadCur[None]
    for c in range(COLS):
        i = topCol[c]
        if i >= 0:
            ovf[i] = f


@ti.kernel
def refill_top_overburden(mode: ti.i32):
    """Досыпка обломков сверху (модель "кучи породы над выработкой").

    Каждый тик проверяем верх колонок: если верхняя клетка колонки пуста
    (кровля просела/обрушилась), кладём туда новый ОДИНОЧНЫЙ обломок. Он НЕ
    связывается со старыми блоками (пришёл сверху), но если в этом же тике
    досыпаны обломки в соседние колонки — они связываются между собой (если
    рядом). Так обрушение сверху просто приносит новые блоки вниз, как кучу
    породы в шахте."""
    if mode == 0 and refillOff[None] == 0:
        c = 0
        while c < COLS:
            newRubble[c] = -1
            c += 1
        # 1. находим верхний блок каждой колонки и досыпаем при пустой верхней клетке
        c = 0
        while c < COLS:
            best = -1
            bestY = 1e9
            i = 0
            while i < bcount[None]:
                cc = clamp_cell(bx[i])
                if cc == c and by[i] < bestY:
                    bestY = by[i]
                    best = i
                i += 1
            # верхняя клетка (y=0) свободна, если в колонке вообще нет блока,
            # либо самый верхний блок опустился ниже неё (координаты в м)
            if best < 0 or bestY >= LCELL:
                j = add_block_func(c + 0.5, 0.5, 1.0, 0)
                if j >= 0:
                    newRubble[c] = j
            c += 1
        # 2. связываем только новые обломки соседних колонок (старые не трогаем)
        c = 0
        while c < COLS:
            a = newRubble[c]
            if a >= 0 and c + 1 < COLS:
                b = newRubble[c + 1]
                if b >= 0:
                    add_bond_func(a, b, 1.0, 1)
            c += 1


def compute_top_load(mode):
    compute_top_load_fused(mode)
    refill_top_overburden(mode)


# ================================================================== флюид
@ti.kernel
def fluid_integrate(dt: ti.f32):
    for i in range(pCount[None]):
        wvy[i] += fwg[None] * dt
        sp2 = wvx[i] * wvx[i] + wvy[i] * wvy[i]
        if sp2 > WVMAX * WVMAX:
            sc = WVMAX / ti.sqrt(sp2)
            wvx[i] *= sc
            wvy[i] *= sc
        wpx[i] = wx[i]
        wpy[i] = wy[i]
        dx = wvx[i] * dt
        dy = wvy[i] * dt
        if dx > 0.35 * LCELL:
            dx = 0.35 * LCELL
        elif dx < -0.35 * LCELL:
            dx = -0.35 * LCELL
        if dy > 0.35 * LCELL:
            dy = 0.35 * LCELL
        elif dy < -0.35 * LCELL:
            dy = -0.35 * LCELL
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
                            if D > 0.04 * LCELL * LCELL:
                                D = 0.04 * LCELL * LCELL
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
        elif wx[i] > COLS * LCELL - m:
            wx[i] = COLS * LCELL - m
        if wy[i] < m:
            wy[i] = m
        elif wy[i] > ROWS * LCELL - m:
            wy[i] = ROWS * LCELL - m


@ti.kernel
def fluid_vel(dt: ti.f32):
    for i in range(pCount[None]):
        wvx[i] = (wx[i] - wpx[i]) / dt
        wvy[i] = (wy[i] - wpy[i]) / dt
        s2 = wvx[i] * wvx[i] + wvy[i] * wvy[i]
        if s2 > WVMAX * WVMAX:
            s = WVMAX / ti.sqrt(s2)
            wvx[i] *= s
            wvy[i] *= s


@ti.kernel
def collide_water(acc: ti.i32):
    if acc == 1:
        for i in range(bcount[None]):
            wfx[i] = 0.0
            wfy[i] = 0.0
    margin = RP_SPH
    RR = int(ti.ceil(P[None].maxSize * 0.5 + RP_SPH / LCELL))
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
            elif wx[p] > COLS * LCELL - margin:
                wx[p] = COLS * LCELL - margin
            if wy[p] < margin:
                wy[p] = margin
            elif wy[p] > ROWS * LCELL - margin:
                wy[p] = ROWS * LCELL - margin
    p = pCount[None] - 1
    while p >= 0:
        if wx[p] < -LCELL or wx[p] > COLS * LCELL + LCELL or wy[p] < -LCELL or wy[p] > ROWS * LCELL + LCELL:
            kill_particle(p)
        p -= 1


@ti.kernel
def fluid_frame_start():
    for i in range(pCount[None]):
        fx0[i] = wx[i]
        fy0[i] = wy[i]


@ti.kernel
def fluid_clamp_disp():
    LIM = 1.5 * LCELL
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
        dx = (wx[i] / LCELL - cavX[None]) / cavRX[None]
        dy = (wy[i] / LCELL - cavY[None]) / cavRY[None]
        if dx * dx + dy * dy < 1.0:
            n += 1.0
    cap = PI * cavRX[None] * cavRY[None] * LCELL * LCELL / (PI * RP_SPH * RP_SPH)
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
    capWarned[None] = 0
    filledFlag[None] = 0
    srcCount[None] = 0
    simT[None] = 0.0
    compactT[None] = 0.0
    frameBrk[None] = 0
    breakCrush[None] = 0
    dmax[None] = 0.0
    P[None].maxSize = 1
    for c in range(N):
        hhead[c] = -1
        fhead[c] = -1
        gOcc[c] = 0
        sigVI[c] = 0.0
        holdL[c] = 0.0
        holdR[c] = 0.0
    for c in range(COLS):
        topCol[c] = -1
        topYArr[c] = 1e9
        newRubble[c] = -1
    for s in range(FRIC_SLOTS):
        fricKey[s] = -1
        fricS[s] = 0.0
        fricF[s] = 0
    for i in range(MAXB):
        bfixed[i] = 0
        nbond[i] = 0
        hasL[i] = 0
        hasR[i] = 0
        supp[i] = 0
        comIn[i] = 0
        supLo[i] = 1e9
        supHi[i] = -1e9
    for bd in range(MAXBONDS):
        bondIntact[bd] = 0
        bondE[bd] = 0.0
        bondPlast[bd] = 0.0
        bondSlip[bd] = 0.0
        bondFlow[bd] = 0
        bondCreep[bd] = 0.0
    for i in range(MAXB):
        for w in range(BMAPW):
            bondMap[i, w] = 0


@ti.kernel
def update_mass_rho():
    for i in range(bcount[None]):
        bmass[i] = P[None].rho * bsz[i] * bsz[i]


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
    v0 = math.sqrt(2.0 * G_PHYS * P[None].head)    # м/с: скорость вылета воды по высоте напора
    for i in range(srcCount[None]):
        spawn_water(3, v0, srcX[i], srcY[i], PI * 0.5)


# ================================================================== главный цикл
# ЛОКАЛЬНАЯ ФИЗИКА РЕШАТЕЛЯ. Один ТИК = один вызов step_physics(): модель
# продвигается на TICK секунд внутреннего времени, и следующий тик наступает
# только после полного расчёта предыдущего (никакой привязки к кадрам или
# реальному времени — скорость тиков ограничена только CPU). Подшаги интегрирования
# внутри тика — локальная деталь решателя; наружу виден только тик.
def fluid_step(dt):
    fluid_integrate(dt)
    fluid_hash()
    fluid_pbd(dt)
    fluid_apply_corr()
    fluid_vel(dt)


def step_physics(mode, dt=None):
    """Продвинуть локальную физику на ОДИН ТИК внутреннего времени (TICK с).

    dt — подшаг интегрирования внутри тика (по умолчанию DT = TICK/SUBSTEPS).
    Тик не зависит от кадров/реального времени: следующий тик стартует только
    после полного расчёта этого и идёт так быстро, как позволяет CPU."""
    if dt is None:
        dt = DT
    simT[None] += TICK
    frameBrk[None] = 0
    decay_bond_damage()
    emit_sources()
    pc = pCount[None]
    if capWarned[None] == 1 and pc < PMAX * 0.85:
        capWarned[None] = 0
    build_hash()
    # когда воды нет, флюидные подшаги пропускаются ЦЕЛИКОМ: ~25 пустых лаунчей
    # ядер за тик — заметная доля CPU-накладных расходов (GPU при этом простаивает)
    if pc > 0:
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
    refresh_field()
    # множители демпфирования на ПОДШАГ: базовые константы заданы на тик, подшагов
    # SUBSTEPS за тик, поэтому берём base^(dt/TICK) — итог за тик ровно DAMP_*_TICK.
    dfree = DAMP_FREE_TICK ** (dt / TICK)
    dsupp = DAMP_SUPP_TICK ** (dt / TICK)
    drotFree = DAMP_ROT_FREE_TICK ** (dt / TICK)
    drotBond = DAMP_ROT_BOND_TICK ** (dt / TICK)
    for ss in range(SUBSTEPS):
        if ss % TOPO_EVERY == 0:
            # топология (занятость клеток, вертикальное напряжение, контактный хэш,
            # счётчики связей) пересобирается по ТЕКУЩИМ позициям, а не раз в тик;
            # frameBrk сбрасывается вместе с ней, чтобы лимит разрывов не создавал
            # импульсную волну разрушения в первые подшаги тика.
            build_hash()
            count_bonds()
            refresh_field()
            frameBrk[None] = 0
        # ВСЕ силы подшага — одним слитым ядром (reset+bonds+contact), затем
        # интегрирование (с моментами качания внутри). Два лаунча вместо пяти.
        phys_forces(dt)
        phys_integrate(dt, dfree, dsupp, drotFree, drotBond)
    cleanup_fallen()
    if simT[None] - compactT[None] >= COMPACT_EVERY_T:
        compact_bonds()
        compactT[None] = simT[None]
    # --- равномерная сводка по всем блокам после каждого тика ---
    # Полная энергия каждого блока и интегралы системы + центр тяжести.
    # Считаются для ВСЕХ блоков одинаково, каждый тик — контроль закона
    # сохранения энергии и положения центра тяжести.
    compute_energy()
    center_of_mass()

# ================================================================== энергия и центр тяжести
@ti.kernel
def compute_energy():
    """Полная энергия КАЖДОГО блока + общие интегралы системы (Дж).

    benergy[i] = ½·m·v² (кинетика) + ½·I·ω² (вращение) + m·g·y (потенциал)
                 + ½·Σ упругих энергий своих связей.
    Закреплённые блоки (опора) исключены из интегралов: у них бесконечная
    реакция опоры, их учёт не даёт физически содержательного баланса. Работа
    выполняется ОДИНАКОВО для всех блоков каждый тик — физика равномерна.

    Диссипация (демпфирование, кулоновское трение, пластика) уменьшает полную
    энергию — это честный отток энергии в тепло. Баланс нужен, чтобы доказать,
    что энергия НЕ генерируется из ниоткуда (нет дрейфа/взрывов)."""
    energyKin[None] = 0.0
    energyRot[None] = 0.0
    energyPot[None] = 0.0
    energyEl[None] = 0.0
    g = P[None].g
    for i in range(bcount[None]):
        benergy[i] = 0.0
        if bfixed[i] == 1:
            continue
        m = bmass[i]
        v2 = bvx[i] * bvx[i] + bvy[i] * bvy[i]
        ke = 0.5 * m * v2
        I = m * bsz[i] * bsz[i] / 6.0
        re = 0.5 * I * bw[i] * bw[i]
        pe = m * g * by[i]
        benergy[i] = ke + pe + re
        energyKin[None] += ke
        energyRot[None] += re
        energyPot[None] += pe
    for bd in range(bondCount[None]):
        if bondIntact[bd] == 1:
            e = bondE[bd]
            energyEl[None] += e
            ti.atomic_add(benergy[bondA[bd]], e * 0.5)
            ti.atomic_add(benergy[bondB[bd]], e * 0.5)
    energyTotal[None] = energyKin[None] + energyRot[None] + energyPot[None] + energyEl[None]


@ti.kernel
def center_of_mass():
    """Центр тяжести системы (при однородном g совпадает с центром масс).

    Учитываются только подвижные блоки (bfixed==0); закреплённая «стена» — опора
    и граница модели, а не часть обрушаемого массива."""
    comX[None] = 0.0
    comY[None] = 0.0
    comMass[None] = 0.0
    for i in range(bcount[None]):
        if bfixed[i] == 1:
            continue
        m = bmass[i]
        comMass[None] += m
        comX[None] += m * bx[i]
        comY[None] += m * by[i]
    if comMass[None] > 0.0:
        comX[None] /= comMass[None]
        comY[None] /= comMass[None]


def energy_report():
    """Сводка энергии системы после последнего тика: (kin, rot, pot, elast, total), Дж."""
    return (energyKin[None], energyRot[None], energyPot[None], energyEl[None], energyTotal[None])


def center_of_mass_report():
    """Центр тяжести (масс) после последнего тика: (x, y, масса), м/кг."""
    return (comX[None], comY[None], comMass[None])
