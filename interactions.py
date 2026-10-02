"""ГЕОМАССИВ/2D · DEM — расчёт взаимодействий между блоками (interactions)

Парные взаимодействия дискретно-элементной модели (DEM). Каждая функция ПРИНИМАЕТ
текущее состояние (поля из physstate) и ВОЗВРАЩАЕТ результат в силовые поля
(bfx/bfy/btor); оркестратор тиков (solver.py) затем читает эти силы и интегрирует.

Ядра расчёта — @ti.func (forces_*), чтобы решатель мог вызывать их ПО ОТДЕЛЬНОСТИ
(phys_reset / phys_bonds / phys_contact / phys_rock) или ОДНИМ слитым ядром
(phys_forces — все силы; phys_substep в solver.py — весь подшаг, включая
интегрирование) — меньше лаунчей на подшаг, меньше накладных расходов CPU и
выше загрузка GPU. Порядок силовых проходов: zero -> contact -> reset -> bonds:
  forces_zero    — обнуление аккумуляторов сил (bfx/bfy/btor);
  forces_contact — точные контакты квадратов (SAT) + трение + конфайнмент трещин
                   (заодно собирает флаги граней faceL/faceR/crack* — до чтения
                   в forces_reset);
  forces_reset   — базовые силы: гравитация, вода, перегрузка, боковой распор;
  forces_bonds   — силы упругих связей (нормаль/сдвиг/изгиб, пластика, энергия);
  forces_rock    — качание на полу и тормоз перекатывания (вызывается из
                   phys_integrate — момент нужен непосредственно перед интегрированием).
Каждый проход выполняется параллельно по ВСЕМ блокам сразу — физика равномерна,
ни один блок не обходится проходкой.
"""

import taichi as ti

from physstate import *

# ================================================================== DEM физика
@ti.func
def forces_zero():
    """Обнуление аккумуляторов сил подшага — строго ДО всех проходов сил.

    Все проходы сил (контакт/связи/распор/качание) только НАКАПЛИВАЮТ сюда
    (atomic_add +=), поэтому слитое ядро подшага считает их в любом порядке,
    не опасаясь затирания (раньше forces_reset делал ПРИСВАИВАНИЕ bfx=wfx и
    стёр бы всё, что набрал контакт)."""
    for i in range(bcount[None]):
        bfx[i] = 0.0
        bfy[i] = 0.0
        btor[i] = 0.0


@ti.func
def forces_reset():
    g = P[None].g
    for i in range(bcount[None]):
        bfx[i] += wfx[i]
        if bfixed[i] == 1:
            bfy[i] += wfy[i] + ovf[i]
        else:
            bfy[i] += wfy[i] + ovf[i] + bmass[i] * g
# боковое горное давление (распор) — ТОЛЬКО для ОБРУШЕННОЙ породы (обломки).
    # Для СВЯЗАННОГО массива σ_h = K0·σ_v заложен в горизонтальные связи (update_prestress
    # в physstate, обновляется КАЖДЫЙ тик — усилие связей растёт с глубиной/K0 в рантайме)
    # — силовой распор ему НЕ НУЖЕН и вреден: он
    # складывается с престрессом (двойной счёт) и на свободной грани выработки
    # рисует ложное растяжение вместо радиального ≈ 0 (см. диагностику Кирша).
    # У связанного массива горизонталь держится престрессом связей + клампом стенок;
    # у контура выработки радиальное (в полость) само падает в 0 — связи, смотревшие
    # в полость, удалены вместе с блоками.
    # Обломкам (nbond == 0) распор оставлен: у рыхлой кучи/арки нет связей, и её
    # боковое давление (K0_RUBBLE) передаётся только контактами — см. phys_contact.
    # Флаги faceL/faceR/crackL/crackR собирает forces_contact того же подшага ИЗ
    # ФАКТИЧЕСКИХ расстояний между центрами (size-aware).
    for i in range(bcount[None]):
        if bfixed[i] == 1:
            continue
        if nbond[i] > 0:
            continue   # связанный массив: σ_h заложен престрессом связей
        sv = sigVI[row_cell(by[i]) * COLS + clamp_cell(bx[i])]
        coef = K0_RUBBLE
        # пол бокового сжатия действует даже при sv=0: блоки передают горизонтальную
        # силу друг другу без вертикальной нагрузки (распор кучи/арки)
        # СИЛА на грань (Н), а не «эквивалентное ускорение»:
        # - перекрытие: F = σ_h·h = K0·σ_v·h = coef·sv·n, где σ_v = sv/LCELL
        #   (sv — вес столба шириной в клетку, Н), h = n·LCELL — высота грани,
        #   n = bsz/LCELL. Сила пропорциональна ПЛОЩАДИ грани (n в 2D), а не
        #   массе блока (n²): старый вариант «sv/BMASS_UNIT·bmass» завышал
        #   распор у крупных блоков (n>1) в n раз.
        # - собственный вес: F = g·LATERAL_FLOOR·0.5·m ∝ массе (∝ n² — верно:
        #   интеграл ρ·g·y по высоте свободной грани даёт h²).
        th = (coef * sv * (bsz[i] / LCELL) + P[None].g * LATERAL_FLOOR * bmass[i]) * 0.5
        # faceL/faceR: 1 = грань СВОБОДНА (нет соседа в пределах лат+GAP), 0 = есть сосед.
        # crackL/crackR: 1 = внутренняя трещина (сосед есть, но без связи), массив нетронут
        # (nbond>0; обломки не получают трещинного push — их участь только контакт).
        freeL = faceL[i]
        freeR = faceR[i]
        push = 0.0
        # блок у БОКОВОЙ ГРАНИЦЫ домена (стенки включены): за границей — порода,
        # её боковое давление σ_h = K0·σ_v давит на грань ПОЛНОЙ силой σ_h·h (= 2·th)
        # и сжимает массив ВНУТРЬ. Прежний распор НАРУЖУ упирался в жёсткую стенку
        # (кламп позиции) и не сжимал массив: в нём σ_h ≈ 0 при любом K0 — поле было
        # одноосным, K0 на напряжения не влиял (проверено замером по связям).
        rxh = block_prj(i)   # та же проекция, что у клампа стенки в forces_integrate
        atL = P[None].walls == 1 and bx[i] <= rxh + 0.02 * LCELL
        atR = P[None].walls == 1 and bx[i] >= COLS * LCELL - rxh - 0.02 * LCELL
        if atL == 1 or atR == 1:
            if atL == 1:
                push += th * 2.0
            if atR == 1:
                push -= th * 2.0
        elif freeL == 1 or freeR == 1:
            # ЕСТЬ свободная грань (полость/кромка): весь боковой распор выдавливает
            # блок В НЕЁ. Трещина на противоположной стороне не должна гасить это
            # распирание: стена сходит в выработку, а не стоит зажатой с трещиной.
            if freeL == 1:
                push -= th
            if freeR == 1:
                push += th
        else:
            # свободных граней нет — блок внутри массива. Распор передаётся только
            # через ВНУТРЕННИЕ ТРЕЩИНЫ: обе стороны сжимают трещину (закрытый шов
            # передаёт σ_h = K0·σ_v, как целый массив). В неразорванном массиве
            # лево/право развёрнуты и гасятся (push = 0).
            if crackL[i] == 1:
                push -= th
            if crackR[i] == 1:
                push += th
        # должна возникать только через контакты и связи; добавление bfy здесь создаёт
        # положительную обратную связь (push -> придавливание вниз -> рост sigVI -> ...)
        # и каскадное обрушение. Поэтому bfy НЕ трогаем.
        if ti.abs(push) > 1e-9:
            ti.atomic_add(bfx[i], push)   # push уже в Ньютонах (см. th выше)


@ti.func
def forces_bonds(dt: ti.f32):
    E = P[None].E
    zeta = P[None].cn
    for bd in range(bondCount[None]):
        a = bondA[bd]
        b = bondB[bd]
        # Связь ГРАНИЦЫ модели (с закреплённым блоком — дно/опора): это не порода,
        # крип/разрывы тут не копим, силы и моменты считаем как обычно.
        bfixBond = bfixed[a] + bfixed[b]
        dx = bx[b] - bx[a]
        dy = by[b] - by[a]
        d = ti.sqrt(dx * dx + dy * dy)
        # боковое давление на свободных гранях (σ_h = K0·σ_v) считается в phys_reset
        # для каждого блока отдельно — см. там.
        if bondIntact[bd] == 0:
            continue
        if d < 1e-9:
            continue
        ux = dx / d
        uy = dy / d
        rvx = bvx[b] - bvx[a]
        rvy = bvy[b] - bvy[a]
        vn = rvx * ux + rvy * uy
        # --- геометрия контакта (СИ) ---
        w = bsz[a]
        if bsz[b] < w:
            w = bsz[b]          # ширина шва = меньший блок, м
        ma = bmass[a]
        mb = bmass[b]
        mred = ma * mb / (ma + mb + 1e-9)
        k_ax = E * w / brest[bd]    # жёсткость связи, Н/м
        # предел устойчивости явной схемы (см. STAB_C): k ≤ C·m_red/dt²
        kstab = STAB_C * mred / (dt * dt)
        if k_ax > kstab:
            k_ax = kstab
        Ftmax = P[None].rp * P[None].ftScale * w    # Н
        Fcmax = Ftmax * P[None].rcFactor
        Fsmax = Ftmax * P[None].fsFactor
        strain = d - brest[bd]
        tension = k_ax * strain          # упругое осевое усилие (до пластики)
        yieldT = Ftmax * P[None].plastYield
        # Предел по СЖАТИЮ растёт с обжатием (Друкер-Прагер): σ_conf = σ3 в середине
        # связи (sigVI обновляется каждый подшаг). Без этого yieldC константа и
        # гидростатика ρgh течёт/рвётся как одноосное сжатие на любой глубине.
        # При K0=1: 0.5·Rc + 0.7·σ_v > σ_v — дальнеe поле упругое, разрушение
        # локализуется у контура выработки (там тангенциальное ~2σ).
        mx = (bx[a] + bx[b]) * 0.5
        my = (by[a] + by[b]) * 0.5
        sv = sigVI[row_cell(my) * COLS + clamp_cell(mx)]
        sigma_v = sv / LCELL
        sigma_h = P[None].k0 * sigma_v
        sigma_conf = sigma_h if sigma_h < sigma_v else sigma_v   # σ3, Па
        yieldC = Fcmax * P[None].plastYield + P[None].mcBond * sigma_conf * w
        # --- пуассоново раскалывание: сжатие σ_c растягивает связи ПЕРПЕНДИКУЛЯРНО
        # ему (в реальном массиве это даёт неоднородность; в идеальной решётке
        # поперечное растяжение не накапливается само). Трещины идут ВДОЛЬ сжатия —
        # осевое раскалывание/слоение, как в бетоне и вокруг выработки, а не
        # "блины" поперёк нагрузки. σ_c — локальное главное сжатие из
        # bond_stress_field (направление само поворачивается у контура выработки);
        # собственное обжатие связи гасит разрыв (net tension = tension + split).
        cx = 0.25 * (bSxx[a] + bSxx[b])
        cy = 0.25 * (bSyy[a] + bSyy[b])
        split = P[None].nuSplit * ti.abs(cy) * ux * ux if ti.abs(cy) >= ti.abs(cx) \
            else P[None].nuSplit * ti.abs(cx) * uy * uy
        tens_eff = tension + split   # только критерий растяжения; силы не трогаем
        anyOver = 0
        # --- пластическое течение (крип) решается ДО расчёта усилий ---
        # Любой вид перегруза (растяжение, сжатие, сдвиг, изгиб) НЕ рвёт связь
        # мгновенно: она течёт, оставаясь герметичной. В пластике связь НЕ
        # урезает усилие до плато: она честно передаёт реальную нагрузку, но
        # растягивается (растёт brest — блоки отдаляются) или сплющивается
        # (падает brest — блоки сближаются), и уже само это изменение длины
        # покоя в следующий подшаг снимает избыток, разгружая связь и
        # перераспределяя усилие по сети на соседей (боковые грани — сдвиг,
        # прямые оси — отрыв/обжатие). Так вертикальные полосы-плато исчезают:
        # пластикой живёт только место, где прочность ДЕЙСТВИТЕЛЬНО превышена.
        # Криповое повреждение bondCreep растёт по степенному закону от
        # перегруза и ЗАЛЕЧИВАЕТСЯ при разгрузке. Обрыв — только если высокую
        # нагрузку УДЕРЖИВАЛИ (накопление за время) либо превышение ЗНАЧИТЕЛЬНО.
        if bfixBond == 0 and tens_eff > yieldT:
            anyOver = 1
            overs = (tens_eff - yieldT) / Ftmax    # 0..~на Ftmax
            # медленное пластическое течение (реальная вытяжка связи, саморазгрузка):
            grow = P[None].plastFlow * overs * overs * dt * brest[bd] * 0.1
            if grow > strain * 0.9:
                grow = strain * 0.9    # не даём длине покоя перескочить текущую деформацию
            brest[bd] += grow           # растяжение: блоки отдаляются, связь растёт
            bondPlast[bd] += grow
            # БЫСТРОЕ накопление ПОВРЕЖДЕНИЯ (отдельно от вытяжки): удержанный
            # перегруз должен рвать за ~2 секунды, а не за 400. Иначе связь
            # бесконечно "течёт" (сама себя разгружает ростом brest), не рвётся,
            # и фронт текучести ползёт вверх по кромке вместо обрушения в свод.
            # Накопление в МЕТРАХ (× brest): порог plastLimit·brest — масштабно
            # и по размеру блока инвариантен (время до разрыва не зависит от LCELL).
            bondCreep[bd] += overs * dt * 0.5 * brest[bd]
        elif bfixBond == 0 and tension < 0.0:
            cstr = -tension
            if cstr > yieldC:
                anyOver = 1
                overs = (cstr - yieldC) / Fcmax
                grow = P[None].plastFlow * overs * overs * dt * brest[bd] * 0.1
                if grow > -strain * 0.9:
                    grow = -strain * 0.9
                brest[bd] -= grow       # сжатие: блоки сближаются (остаточное уплотнение)
                bondPlast[bd] += grow
                bondCreep[bd] += overs * dt * 0.5 * brest[bd]
        # --- пересчёт упругих величин ПОСЛЕ пластики (Баг 1/3): сила и энергия
        # должны считаться по АКТУАЛЬНОМУ остаточной деформации (d-brest), иначе
        # каждый подшаг прикладывается старая/большая сила с резким последующим
        # падением -> осцилляция и ложные полосы пластики
        strain = d - brest[bd]
        tension = k_ax * strain
        # Честный Мора-Кулон на связи: сдвиговая прочность шва зависит от нормального
        # напряжения по шву (tension>0 — растяжение, tension<0 — обжатие).
        # τ_s = Fs0 + mcBond·σ_n: обжатие добавляет прочности на сдвиг, растяжение — отнимает.
        # floor: даже сильно растянутый шов сохраняет остаточное сцепление (не делит на 0).
        FsC = Fsmax - P[None].mcBond * tension
        if FsC < Ftmax * 0.2:
            FsC = Ftmax * 0.2
        # сдвиг поперёк НАЧАЛЬНОГО направления связи (bondNX/NY) — это правильная
        # опорная система склеенного шва: сдвиг меряется относительно исходного
        # контакта, а не текущей оси (иначе off становится тождественно нулю и
        # связь теряет сдвиговую жёсткость). Ложный сдвиг при пластической вытяжке
        # НЕ копится, потому что strain уже пересчитан по актуальному brest.
        px = -bondNY[bd]
        py = bondNX[bd]
        ux0 = bondNX[bd]
        uy0 = bondNY[bd]
        # упругий сдвиг поперёк НАЧАЛЬНОГО направления связи: off_eff = (off - bondSlip)
        # — пластическое проскальзывание УМЕНЬШАЕТ упругий сдвиг (как длина покоя для
        # нормали), поэтому блок фактически СЪЕЗЖАЕТ по шву, а не висит на упругой
        # деформации. Направление slip совпадает со знаком текущего off.
        # ТОЧКА ШВА — середина связи (общая грань блоков), а не ЦТ блока: точки
        # приложения сдвинуты от ЦТ каждого блока на hArm = d/2, поэтому вращение
        # блоков даёт вклад в ПОПЕРЕЧНЫЙ сдвиг шва: off -= hArm·(θ_a+θ_b),
        # vs -= hArm·(ω_a+ω_b). Без этого поля смещений u(y) и v(y) решётки
        # РАЗВЯЗАНЫ: вертикальная нагрузка несётся только сдвигом горизонтальных
        # связей, осевые силы верх/низ балки не развиваются — изгибающего
        # напряжения нет, балка ведёт себя как «мембрана».
        hArm = 0.5 * w * (ti.abs(ux0) + ti.abs(uy0))
        ca = ti.cos(brot[a])
        sa = ti.sin(brot[a])
        cb = ti.cos(brot[b])
        sb = ti.sin(brot[b])
        rax = hArm * (ux0 * ca - uy0 * sa)
        ray = hArm * (ux0 * sa + uy0 * ca)
        rbx = -hArm * (ux0 * cb - uy0 * sb)
        rby = -hArm * (ux0 * sb + uy0 * cb)
        off = ((bx[b] + rbx) - (bx[a] + rax)) * px + ((by[b] + rby) - (by[a] + ray)) * py
        offEff = off - bondSlip[bd]
        vs = ((bvx[b] - bw[b] * rby) - (bvx[a] - bw[a] * ray)) * px \
           + ((bvy[b] + bw[b] * rbx) - (bvy[a] + bw[a] * rax)) * py
        k_sh = k_ax / (2.0 * (1.0 + 0.25))    # G = E / 2(1+ν)
        Fs = -k_sh * offEff - 2.0 * zeta * ti.sqrt(k_sh * mred) * vs
        shearF = k_sh * offEff
        rel = brot[b] - brot[a]
        rw = bw[b] - bw[a]
        krot = k_ax * w * w / 12.0             # Н·м/рад
        Ia = bmass[a] * bsz[a] * bsz[a] / 6.0
        Ib = bmass[b] * bsz[b] * bsz[b] / 6.0
        Ired = Ia * Ib / (Ia + Ib + 1e-9)
        # демпфер поворота связи (ζ≈0.8 для пары). ОГРАНИЧЕНИЕ ПО СТАБИЛЬНОСТИ:
        # в решётке блок участвует в nbond связях, и в шахматной моде вращения
        # (соседние блоки крутятся в противофазе) каждый демпфер даёт на блок
        # 2·c·ω, поэтому суммарное торможение доходит до 8·c. Без капа оно
        # превышает предел явной схемы 2·I/dt и решётка входит в период-2 дребезг
        # на клампе скорости (±10 рад/с). Кап: 2·crot·nbond·dt < 2·I  => crot <
        # I/(nbond·dt), берём с запасом 0.5 (тот же стиль, что STAB_C для k).
        crot = 1.6 * ti.sqrt(krot * Ired)
        na_ = nbond[a] if nbond[a] > 1 else 1
        nb_ = nbond[b] if nbond[b] > 1 else 1
        cCap = 0.5 * ti.min(Ia / na_, Ib / nb_) / dt
        if crot > cCap:
            crot = cCap
        cd = 2.0 * zeta * ti.sqrt(k_ax * mred)   # Н·с/м
        Fn = -tension - cd * vn
        Fx = Fn * ux + Fs * px
        Fy = Fn * uy + Fs * py
        # Мгновенная упругая энергия связи (нормаль + сдвиг + изгиб), Дж:
        bondE[bd] = 0.5 * k_ax * strain * strain + 0.5 * k_sh * offEff * offEff + 0.5 * krot * rel * rel
        ti.atomic_add(bfx[b], Fx)
        ti.atomic_add(bfy[b], Fy)
        ti.atomic_add(bfx[a], -Fx)
        ti.atomic_add(bfy[a], -Fy)
        # момент от приложения силы в ТОЧКЕ ШВА (r = ±hArm·u от ЦТ каждого блока):
        # τ = r×F, для пары τ_a = τ_b = -hArm·(u×F). Нормальная составляющая (вдоль u)
        # момента не даёт — только сдвиговая. Эквивалентно градиенту потенциала по
        # углам (см. off -= hArm·(θ_a+θ_b) выше). Без него изгибающий момент шва
        # несётся ложным рычагом силы, приложенной в ЦТ, и решётка не гнётся.
        tlv_a = -(rax * Fy - ray * Fx)
        tlv_b = (rbx * Fy - rby * Fx)
        ti.atomic_add(btor[a], tlv_a)
        ti.atomic_add(btor[b], tlv_b)
        # Момент связи: tor - момент, приложенный к блоку A (b - противоположный).
        # ЗНАК: восстанавливающий. Потенциал изгиба U = ½·krot·rel² (rel = brot[b]-brot[a])
        # даёт τ_a = -∂U/∂brot[a] = +krot·rel, τ_b = -krot·rel; демпфер тянет блоки
        # к равенству угловых скоростей: +c·(bw[b]-bw[a]) на A. Раньше стояло
        # «tor = -krot·rel», что при применении btor[a]+=tor / btor[b]-=tor давало
        # АНТИ-восстанавливающий момент: rel разгонялся, связи рвались по
        # перегибу (|rel| > 3·relBend), а решётка раскручивалась в шахматном
        # порядке (±bw до клампа). Баг был скрыт, пока контакт не давал блокам
        # момент (brot связанных блоков оставался 0 и krot вообще не работал).
        tor = krot * rel + crot * rw
        ti.atomic_add(btor[a], tor)
        ti.atomic_add(btor[b], -tor)
        # --- сдвиг и изгиб: тоже копят крип (не рвут мгновенно) ---
        yieldS = FsC * P[None].plastYield
        if bfixBond == 0 and ti.abs(shearF) > yieldS:
            anyOver = 1
            so = (ti.abs(shearF) - yieldS) / FsC
            bondCreep[bd] += so * dt * 0.5 * brest[bd]
            # пластическое проскальзывание по шву: bondSlip растёт, снижая offEff —
            # блок реально съезжает в направлении сдвига, разгружая связь (и себя)
            sgn = 1.0 if shearF > 0.0 else -1.0
            slipG = P[None].plastFlow * so * so * dt * brest[bd] * 0.1
            if slipG > ti.abs(off) * 0.9:
                slipG = ti.abs(off) * 0.9
            bondSlip[bd] += sgn * slipG
        if bfixBond == 0 and ti.abs(rel) > P[None].relBend * P[None].plastYield:
            anyOver = 1
            bo = (ti.abs(rel) - P[None].relBend * P[None].plastYield) / P[None].relBend
            bondCreep[bd] += bo * dt * 2.0 * brest[bd]
        # --- предел суммарной пластической вытяжки И проскальзывания: исчерпав
        # пластичность (и запас деформации), связь больше не течёт — только рвётся.
        # Без этого кромка сцепления текла бы бесконечно, давая всё тот же ползущий
        # фронт. 0.15 = 15% от длины покоя: блок может "отойти"/"съехать" на 15%
        # (раньше было 6% — запас слишком мал и связи рвались слишком рано,
        # не дав блоку реально возобновиться).
        if bfixBond == 0 and (ti.abs(bondPlast[bd]) > 0.15 * brest[bd] or ti.abs(bondSlip[bd]) > 0.15 * brest[bd]):
            anyOver = 1
            bondCreep[bd] = P[None].plastLimit * brest[bd]   # исчерпана пластичность -> рвём
        # залечивание: если ни один критерий не превышен — крип возвращается,
        # кратковременный выброс не рвёт связь
        if anyOver == 0:
            bondCreep[bd] *= ti.exp(-P[None].creepRecover * dt)
            if bondCreep[bd] < 1e-6:
                bondCreep[bd] = 0.0
        # маркеры текущей связи (фиолетовый — идёт крип)
        if anyOver == 1:
            bondFlow[bd] = 1
            bondDmg[bd] = P[None].dmgMax
        # маркеры напряжений
        if tension > 0.0:
            r = tension / Ftmax
            bondR[bd] = r
            ti.atomic_max(bstress[a], r)
            ti.atomic_max(bstress[b], r)
        else:
            bondR[bd] = tension / (Fcmax * 0.25)
            c = -tension / Fcmax
            ti.atomic_max(bstressC[a], c)
            ti.atomic_max(bstressC[b], c)
        # --- обрыв ---
        # (а) аварийный ХРУПКИЙ разрыв при значительном перегрузе: это останавливает
        # фронт локальным разрывом и передаёт нагрузку в свод. Окно 0.9–1.25·Ftmax —
        # чистое пластическое течение; выше — хрупкий обрыв.
        # (б) иначе — только при УДЕРЖАННОМ криповом повреждении (плавное оседание).
        overE = 0
        crushE = 0
        if tens_eff > Ftmax * 2.5 or strain > 0.24 * LCELL:
            overE = 1
        elif -tension > Fcmax * 2.5:
            overE = 1
            crushE = 1
        elif ti.abs(shearF) > FsC * 3.0 or ti.abs(rel) > P[None].relBend * 3.0:
            overE = 1
        if overE == 1 and bfixBond == 0:
            if try_fracture(bd, a, b, crushE):
                fracture_bond(bd, a, b, crushE, strain, off, rel)
        elif bfixBond == 0 and anyOver == 1 and bondCreep[bd] >= P[None].plastLimit * brest[bd]:
            crush = 0
            if tension < -1e-6:
                crush = 1
            if try_fracture(bd, a, b, crush):
                fracture_bond(bd, a, b, crush, strain, off, rel)


# ================================================================== геометрия контакта квадратов
# Точный контакт двух ПОВЁРНУТЫХ квадратов (OBB): SAT по 4 осям + клиппинг граней.
# Даёт нормаль, глубину и РЕАЛЬНЫЕ точки контакта (1 или 2) с глубиной каждой.
# Никаких "раздутых проекций": углы зацепляются только там, где квадраты
# действительно перекрываются — фантомных пружин под повёрнутым блоком нет.

_mat3f = ti.types.matrix(3, 3, ti.f32)


@ti.func
def sat_squares(cix: ti.f32, ciy: ti.f32, ri: ti.f32, cti: ti.f32, sti: ti.f32,
                cjx: ti.f32, cjy: ti.f32, rj: ti.f32, ctj: ti.f32, stj: ti.f32) -> _mat3f:
    """ Separating Axis Theorem для двух квадратов.
    Возврат M: [0,0..1] — нормаль (единичная, от i к j), [0,2] — глубина проникновения,
    [1,0] — 1 если перекрытие есть, [1,1] — 1 если опорная грань у блока i (иначе у j)."""
    dx = cjx - cix
    dy = cjy - ciy
    vix = -sti
    viy = cti
    vjx = -stj
    vjy = ctj
    best = 1e9
    nx = 1.0
    ny = 0.0
    refI = 1.0
    hit = 1.0
    a = 0
    while a < 4 and hit > 0.5:
        ax = 0.0
        ay = 0.0
        if a == 0:
            ax = cti
            ay = sti
        elif a == 1:
            ax = vix
            ay = viy
        elif a == 2:
            ax = ctj
            ay = stj
        else:
            ax = vjx
            ay = vjy
        # полупроекция квадрата на ось: r·(|u·a| + |v·a|)
        ra = ri * (ti.abs(cti * ax + sti * ay) + ti.abs(vix * ax + viy * ay))
        rb = rj * (ti.abs(ctj * ax + stj * ay) + ti.abs(vjx * ax + vjy * ay))
        dc = dx * ax + dy * ay
        ov = ra + rb - ti.abs(dc)
        if ov <= 0.0:
            hit = 0.0    # разделяющая ось найдена — контакта нет
        elif ov < best:
            best = ov
            sgn = 1.0
            if dc < 0.0:
                sgn = -1.0
            nx = ax * sgn    # нормаль — от i к j
            ny = ay * sgn
            refI = 1.0
            if a >= 2:
                refI = 0.0
        a += 1
    return ti.Matrix([[nx, ny, best], [hit, refI, 0.0], [0.0, 0.0, 0.0]])


@ti.func
def clip_contact_points(cix: ti.f32, ciy: ti.f32, ri: ti.f32, cti: ti.f32, sti: ti.f32,
                        cjx: ti.f32, cjy: ti.f32, rj: ti.f32, ctj: ti.f32, stj: ti.f32,
                        nx: ti.f32, ny: ti.f32, refI: ti.f32) -> _mat3f:
    """Клиппинг падающей грани об опорную (Sutherland-Hodgman, 2 плоскости).
    Возврат M: строка0 = (p0x, p0y, δ0), строка1 = (p1x, p1y, δ1),
    строка2 = (число точек, ширина контакта W, 0). δ — глубина каждой точки."""
    # опорная грань: у блока refI её внешняя нормаль = n (от i к j), иначе −n
    nrx = nx
    nry = ny
    crx = cix
    cry = ciy
    rr = ri
    qix = cjx
    qiy = cjy
    rq = rj
    if refI < 0.5:
        nrx = -nx
        nry = -ny
        crx = cjx
        cry = cjy
        rr = rj
        qix = cix
        qiy = ciy
        rq = ri
    mx = -nry
    my = nrx
    # опорная грань: центр fc = cr + rr·nref, концы fc ± rr·m
    fcx = crx + rr * nrx
    fcy = cry + rr * nry
    # падающая грань второго квадрата (внешняя нормаль ≈ −nref): ic ± rq·m
    icx = qix - rq * nrx
    icy = qiy - rq * nry
    b0x = icx - rq * mx
    b0y = icy - rq * my
    b1x = icx + rq * mx
    b1y = icy + rq * my
    # клиппинг отрезка b0→b1 по полосе m·p ∈ [lo, hi]
    lo = (fcx * mx + fcy * my) - rr
    hi = lo + 2.0 * rr
    s0 = b0x * mx + b0y * my
    s1 = b1x * mx + b1y * my
    t0 = 0.0
    t1 = 1.0
    ds = s1 - s0
    ok = 1.0
    if ti.abs(ds) < 1e-12:
        if s0 < lo or s0 > hi:
            ok = 0.0
    else:
        ta = (lo - s0) / ds
        tb = (hi - s0) / ds
        if ta > tb:
            tmp = ta
            ta = tb
            tb = tmp
        if ta > t0:
            t0 = ta
        if tb < t1:
            t1 = tb
        if t0 > t1:
            ok = 0.0
    count = 0.0
    wid = 0.0
    p0x = 0.0
    p0y = 0.0
    d0 = 0.0
    p1x = 0.0
    p1y = 0.0
    d1 = 0.0
    if ok > 0.5:
        dRef = fcx * nrx + fcy * nry    # плоскость опорной грани
        qax = b0x + (b1x - b0x) * t0
        qay = b0y + (b1y - b0y) * t0
        qbx = b0x + (b1x - b0x) * t1
        qby = b0y + (b1y - b0y) * t1
        da = dRef - (qax * nrx + qay * nry)
        db = dRef - (qbx * nrx + qby * nry)
        if da > 1e-9 and db > 1e-9:
            count = 2.0
            p0x = qax
            p0y = qay
            d0 = da
            p1x = qbx
            p1y = qby
            d1 = db
            wid = ti.sqrt((qbx - qax) * (qbx - qax) + (qby - qay) * (qby - qay))
        elif da > 1e-9:
            count = 1.0
            p0x = qax
            p0y = qay
            d0 = da
        elif db > 1e-9:
            count = 1.0
            p0x = qbx
            p0y = qby
            d0 = db
    return ti.Matrix([[p0x, p0y, d0], [p1x, p1y, d1], [count, wid, 0.0]])


@ti.func
def contact_fn(i: ti.i32, j: ti.i32, px: ti.f32, py: ti.f32, dp: ti.f32,
               nx: ti.f32, ny: ti.f32, kpt: ti.f32, cd: ti.f32) -> ti.f32:
    """Нормальная сила одной точки контакта: пружина (kpt·δ) + демпфер по скорости
    СБЛИЖЕНИЯ ТОЧЕК с учётом вращения обоих блоков (ω×r). Отскок гасится: Fn ≥ 0."""
    rix = px - bx[i]
    riy = py - by[i]
    rjx = px - bx[j]
    rjy = py - by[j]
    # скорость точки на теле: v + ω×r, где ω×r = ω·(−ry, rx)
    vrx = (bvx[j] - bw[j] * rjy) - (bvx[i] - bw[i] * riy)
    vry = (bvy[j] + bw[j] * rjx) - (bvy[i] + bw[i] * rix)
    vsep = vrx * nx + vry * ny      # < 0 — сближение
    Fn = kpt * dp - cd * vsep
    if Fn < 0.0:
        Fn = 0.0
    return Fn


@ti.func
def apply_contact_point(i: ti.i32, j: ti.i32, fkey: ti.i32,
                        px: ti.f32, py: ti.f32,
                        nx: ti.f32, ny: ti.f32,
                        Fn: ti.f32, FnF: ti.f32, kpt: ti.f32, dt: ti.f32):
    """Приложить силу одной точки контакта: нормаль Fn (распор + момент вокруг ЦТ),
    опора/копилка интервала поддержки, кулоновское трение FnF в той же точке.

    Момент от нормали — р × F: у блока, вставшего КРАЕМ на опору, точка контакта
    уходит от вертикали под ЦТ, и вес реально опрокидывает его (никакой пружины
    "вверх из центра"). FnF — нормаль для трения (может быть выше Fn за счёт
    конфайнмента закрытой трещины)."""
    rix = px - bx[i]
    riy = py - by[i]
    rjx = px - bx[j]
    rjy = py - by[j]
    if Fn > 0.0:
        fxx = Fn * nx
        fyy = Fn * ny
        ti.atomic_add(bfx[i], -fxx)
        ti.atomic_add(bfy[i], -fyy)
        ti.atomic_add(bfx[j], fxx)
        ti.atomic_add(bfy[j], fyy)
        ti.atomic_max(cnck[i], Fn)
        ti.atomic_max(cnck[j], Fn)
        # τ = r × F прикладывается ВСЕГДА, независимо от связей блока:
        # сила в точке контакта -> сила в ЦТ + момент вокруг ЦТ. Пропуск момента
        # для связанных блоков (nbond>0) замораживал поворот ЛЮБОГО составного
        # тела: единственный внешний источник вращения - эксцентриситет контакта
        # (гравитация идёт в ЦТ, моменты связей krot отвечают лишь на ОТНОСИТЕЛЬНЫЙ
        # поворот блоков и при нулевых brot дают ноль). Балка-составное тело не
        # перевешивала и не опрокидывалась, а в её связях не развивались осевые
        # напряжения изгиба (brot оставался ровно 0 -> без натяжения верх/низ).
        ti.atomic_add(btor[i], Fn * (riy * nx - rix * ny))
        ti.atomic_add(btor[j], Fn * (rjx * ny - rjy * nx))
        # интервал опоры: контакт толкает блок ВВЕРХ (сила против +y)
        if ny > 0.01:
            ti.atomic_min(supLo[i], px)
            ti.atomic_max(supHi[i], px)
        elif ny < -0.01:
            ti.atomic_min(supLo[j], px)
            ti.atomic_max(supHi[j], px)
    tx = -ny
    ty = nx
    # относительная касательная скорость в точке (с учётом вращения)
    vrx = (bvx[j] - bw[j] * rjy) - (bvx[i] - bw[i] * riy)
    vry = (bvy[j] + bw[j] * rjx) - (bvy[i] + bw[i] * rix)
    vt = vrx * tx + vry * ty
    friction_vec(i, j, fkey, vt, FnF, dt, tx, ty, px, py, kpt)


@ti.func
def forces_contact(dt: ti.f32):
    k_c = P[None].E * 0.1   # жёсткость контакта полной ширины (Н/м), k = E·A/L
    # динамический радиус поиска в клетках: описанная окружность квадрата r·√2
    # + запас на PROX (пре-контакт для phys_rock)
    RR = int(ti.ceil(P[None].maxSize * 1.5))
    # флаги граней собираются ЗДЕСЬ (размер-осведомлённо, каждый подшаг) — не в
    # клеточном поле gOcc. По умолчанию грань свободна (1), трещины/опоры нет (0);
    # нюанс-init в начале каждого витка параллельного цикла обязателен: ниже флаги
    # только ОБНУЛЯЮТСЯ для занятых сторон.
    for i in range(bcount[None]):
        faceL[i] = 1
        faceR[i] = 1
        crackL[i] = 0
        crackR[i] = 0
        supp[i] = 0
        comIn[i] = 0
        inContact[i] = 0
        cnck[i] = 0.0
        supLo[i] = 1e9
        supHi[i] = -1e9
    for i in range(bcount[None]):
        cxi = clamp_cell(bx[i])
        cyi = clamp_cell(by[i])
        ri = bsz[i] * 0.5          # ПОЛОВИНА СТОРОНЫ квадрата (без раздутия проекций)
        cir = ri * 1.41421356      # описанный радиус: предел досягаемости углов
        cti = ti.cos(brot[i])
        sti = ti.sin(brot[i])
        rx = block_prj(i)          # только для size-aware флагов граней ниже
        # радиус поиска — динамический (±RR клеток): без статических 13×13 итераций
        # на блок (MAXNBR), которые почти все отсеивались фильтром — минус ~85%
        # холостых проходов хэша на подшаг
        for oy in range(-RR, RR + 1):
            yy = cyi + oy
            if yy >= 0 and yy < ROWS:
                for ox in range(-RR, RR + 1):
                    xx = cxi + ox
                    if xx >= 0 and xx < COLS:
                        j = hhead[yy * COLS + xx]
                        while j != -1:
                            if j > i:
                                dx = bx[j] - bx[i]
                                dy = by[j] - by[i]
                                rj = bsz[j] * 0.5
                                rxj = block_prj(j)
                                rs = rx + rxj
                                lat = rs
                                GAP = 0.25 * lat
                                # ---- геометрические флаги (size-aware), для ВСЕХ пар ----
                                # тот же "этаж" (без вертикального смещения больше лат*0.95):
                                if ti.abs(dy) < lat * 0.95:
                                    # j правее i в пределах lat+GAP
                                    if dx > 0.0 and dx < lat + GAP:
                                        faceR[i] = 0
                                        faceL[j] = 0
                                        if is_bonded_bit(i, j) == 0:
                                            # внутренняя трещина: сосед есть, связи нет
                                            crackR[i] = 1
                                            crackL[j] = 1
                                    elif dx < 0.0 and -dx < lat + GAP:
                                        faceL[i] = 0
                                        faceR[j] = 0
                                        if is_bonded_bit(i, j) == 0:
                                            crackL[i] = 1
                                            crackR[j] = 1
                                if not (bfixed[i] == 1 and bfixed[j] == 1):
                                    key = bkey(i, j)
                                    if is_bonded_bit(i, j) == 0:
                                        axx = ti.abs(dx)
                                        ayy = ti.abs(dy)
                                        rsum = cir + rj * 1.41421356
                                        # близость (не только перекрытие): блок, зависший
                                        # над кучей, уже попадает под сопротивление перекатыванию
                                        PROX = 0.7 * LCELL
                                        if axx < rsum + PROX and ayy < rsum + PROX:
                                            inContact[i] = 1
                                            inContact[j] = 1
                                        if axx < rsum and ayy < rsum:
                                            ctj = ti.cos(brot[j])
                                            stj = ti.sin(brot[j])
                                            sat = sat_squares(bx[i], by[i], ri, cti, sti,
                                                              bx[j], by[j], rj, ctj, stj)
                                            if sat[1, 0] > 0.5:
                                                nx = sat[0, 0]
                                                ny = sat[0, 1]
                                                pts = clip_contact_points(
                                                    bx[i], by[i], ri, cti, sti,
                                                    bx[j], by[j], rj, ctj, stj,
                                                    nx, ny, sat[1, 1])
                                                npts = int(pts[2, 0])
                                                if npts > 0:
                                                    inContact[i] = 1
                                                    inContact[j] = 1
                                                    # --- ПЛОЩАДЬ КОНТАКТА (ширина в 2D) ---
                                                    # жёсткость пропорциональна ширине пятна:
                                                    # грань-к-грани — жёстко, угол-к-грани —
                                                    # мягко (0.2). Сила по-прежнему приложена
                                                    # В ТОЧКАХ пятна — момент от ЦТ честный.
                                                    W = pts[2, 1]
                                                    sAvg = 0.5 * (bsz[i] + bsz[j])
                                                    wFrac = W / sAvg
                                                    if npts == 1 or wFrac < 0.2:
                                                        wFrac = 0.2
                                                    if wFrac > 1.0:
                                                        wFrac = 1.0
                                                    fp = npts * 1.0
                                                    kpt = k_c * wFrac / fp
                                                    mred = bmass[i] * bmass[j] / (bmass[i] + bmass[j] + 1e-9)
                                                    # предел устойчивости (см. STAB_C)
                                                    kstab = STAB_C * mred / (dt * dt)
                                                    if kpt > kstab:
                                                        kpt = kstab
                                                    # ζ контакта: перекритическое для пары
                                                    # свободных блоков (рухляк не отскакивает),
                                                    # обычное — если хоть один на связи
                                                    zc = P[None].cnc
                                                    if nbond[i] == 0 and nbond[j] == 0:
                                                        zc = P[None].cncR
                                                    cd = 2.0 * zc * ti.sqrt(kpt * mred)
                                                    # страховка от глубокого проникновения
                                                    # (быстрый удар): δ ≤ 30% меньшего блока
                                                    dMax = 0.3 * ti.min(bsz[i], bsz[j])
                                                    d0 = ti.min(pts[0, 2], dMax)
                                                    d1 = 0.0
                                                    p0x = pts[0, 0]
                                                    p0y = pts[0, 1]
                                                    p1x = 0.0
                                                    p1y = 0.0
                                                    Fn0 = contact_fn(i, j, p0x, p0y, d0, nx, ny, kpt, cd)
                                                    Fn1 = 0.0
                                                    if npts == 2:
                                                        p1x = pts[1, 0]
                                                        p1y = pts[1, 1]
                                                        d1 = ti.min(pts[1, 2], dMax)
                                                        Fn1 = contact_fn(i, j, p1x, p1y, d1, nx, ny, kpt, cd)
                                                    # --- конфайнмент ПО ЗАКРЫТИЮ (аналогия сжатого стержня) ---
                                                    # Вертикальная трещина = вертикальная грань:
                                                    # как только берега сомкнулись, сквозь закрытую
                                                    # трещину передаётся горизонтальное сжатие
                                                    # σ_h = K0·σ_v — как у целого массива. Это
                                                    # поднимает только нормаль ТРЕНИЯ (сдвиг
                                                    # берегов), распор остаётся в phys_reset.
                                                    FnF0 = Fn0
                                                    FnF1 = Fn1
                                                    if ti.abs(nx) > ti.abs(ny):
                                                        sv_i = sigVI[row_cell(by[i]) * COLS + clamp_cell(bx[i])]
                                                        sv_j = sigVI[row_cell(by[j]) * COLS + clamp_cell(bx[j])]
                                                        sv = sv_i if sv_i > sv_j else sv_j
                                                        coef = K0_RUBBLE
                                                        if nbond[i] > 0 or nbond[j] > 0:
                                                            coef = P[None].k0
                                                        closure = (sat[0, 2] + CONTACT_CLOSE_TOL) / CONTACT_CLOSE_TOL
                                                        if closure < 0.0:
                                                            closure = 0.0
                                                        elif closure > 1.0:
                                                            closure = 1.0
                                                        # нормаль сжатия закрытой трещины: σ_h·h = coef·sv·n
                                                        # (n — высота трещины в клетках = меньший блок); для n=1
                                                        # как раньше, для крупных — масштаб по грани, а не «на клетку»
                                                        hCrack = ti.min(bsz[i], bsz[j]) / LCELL
                                                        Fconf = sv * coef * closure * hCrack
                                                        FnTot = Fn0 + Fn1
                                                        if FnTot > 1e-6:
                                                            if Fconf > FnTot:
                                                                fsc = Fconf / FnTot
                                                                FnF0 = Fn0 * fsc
                                                                FnF1 = Fn1 * fsc
                                                        else:
                                                            FnF0 = Fconf * 0.5
                                                            FnF1 = Fconf * 0.5
                                                    apply_contact_point(i, j, key * 2,
                                                                        p0x, p0y, nx, ny,
                                                                        Fn0, FnF0, kpt, dt)
                                                    if npts == 2:
                                                        apply_contact_point(i, j, key * 2 + 1,
                                                                            p1x, p1y, nx, ny,
                                                                            Fn1, FnF1, kpt, dt)
                            j = hnext[j]
    # --- итог по опоре: supp/comIn из РЕАЛЬНЫХ контактов (а не из "сосед где-то снизу") ---
    # Пол — тоже опора (его реакция в phys_integrate приложена к центру, точек нет):
    # у лежащего на полу блока ЦТ всегда над опорой.
    for i in range(bcount[None]):
        if supLo[i] <= supHi[i]:
            supp[i] = 1
        rxp = block_prj(i)
        if by[i] >= ROWS * LCELL - rxp - 0.03 * LCELL:
            supp[i] = 1
            comIn[i] = 1
        elif supp[i] == 1:
            m = 0.05 * bsz[i]
            if bx[i] >= supLo[i] - m and bx[i] <= supHi[i] + m:
                comIn[i] = 1


@ti.func
def forces_rock():
    """Перекатывание свободного блока + тормоз вращения.

    Контакты блок-блок (phys_contact) теперь ТОЧНЫЕ: нормальная сила приложена
    в реальных точках пятна контакта, и её эксцентриситет относительно ЦТ сам
    даёт опрокидывающий/возвращающий момент — блок на краю опоры ЗАВАЛИВАЕТСЯ,
    блок на грани оседает плоско. Явный момент перекатывания нужен только для
    ПОЛА: пол — не блок, его реакция приложена к центру (кламп в phys_integrate)
    и момента не даёт. Для блока, лежащего НА ПОЛУ (supp==0):

      φ = brot, свёрнутый к ближайшей грани (−45°..45°);
      τ_rock = −ROCK_LIFT·m·g·r·(cosφ − sin|φ|)·tanh(10φ):
        честный момент тяжести вокруг нижнего ребра: максимум при малом φ
        (куб возвращается на грань — ЦТ опускается), НОЛЬ на ребре (45°) —
        неустойчивое равновесие, куб падает на следующую грань. tanh(10φ)
        гладко выключает момент у плоской посадки (φ=0 — опоры ребром нет).

    τ_roll = −μ_r·Fn·r·tanh(bw·2): сопротивление перекатыванию (тормоз вечного
    вращения). μ_r ОСЛАБЛЯЕТСЯ (×0.15), когда ЦТ вышел за интервал опоры
    (comIn==0): такой блок должен завалиться, и тормоз не может удерживать
    его кривым — иначе растут "пружинящие" столбы из блоков.

    На блоки внутри массива (nbond>0) не действует — их держат связи."""
    g = P[None].g
    half = PI * 0.5          # 90° — период симметрии квадрата
    for i in range(bcount[None]):
        if bfixed[i] == 1:
            continue
        if nbond[i] > 0:
            continue
        r = bsz[i] * 0.5
        m = bmass[i]
        rx = block_prj(i)
        on_floor = by[i] >= ROWS * LCELL - rx - 0.30 * LCELL
        near_ceil = by[i] <= rx + 0.30 * LCELL
        if supp[i] == 0 and inContact[i] == 0 and on_floor == 0 and near_ceil == 0:
            continue
        w = m * g * r         # масштаб: вес × рычаг полуграня (Н·м)
        if on_floor == 1:
            # качание по полу: сворачиваем угол к ближайшей грани, q ∈ (−0.5, 0.5]
            q = brot[i] / half
            q = q - ti.floor(q)
            if q > 0.5:
                q -= 1.0
            phi = q * half        # φ ∈ (−π/4, π/4]
            arm = ti.cos(phi) - ti.sin(ti.abs(phi))   # плечо ЦТ от нижнего ребра
            ti.atomic_add(btor[i], -ROCK_LIFT * w * arm * ti.tanh(phi * 10.0))
        # тормоз перекатывания: сила — реальная нормаль контакта (или минимум —
        # собственный вес, если блок опирается/у пола); неустойчивый блок (ЦТ вне
        # опоры) тормозится слабо — он обязан завалиться
        fn_roll = cnck[i]
        if supp[i] == 1 or on_floor == 1:
            if fn_roll < w:
                fn_roll = w
        mu_r = ROLL_MU
        if comIn[i] == 0:
            mu_r = ROLL_MU * 0.15
        ti.atomic_add(btor[i], -mu_r * fn_roll * r * ti.tanh(bw[i] * 2.0))


# ================================================================== ядра-обёртки
# Отдельные ядра сохранены для совместимости (тесты/отладка); рабочий путь
# решателя — phys_substep в solver.py (весь подшаг одним лаунчем).

@ti.kernel
def phys_reset():
    forces_zero()
    forces_reset()


@ti.kernel
def phys_bonds(dt: ti.f32):
    forces_bonds(dt)


@ti.kernel
def phys_contact(dt: ti.f32):
    forces_contact(dt)


@ti.kernel
def phys_rock():
    forces_rock()


@ti.kernel
def phys_forces(dt: ti.f32):
    """Все силы подшага ОДНИМ ядром (без интегрирования).

    Порядок: zero -> contact -> reset -> bonds. Контакт идёт ПЕРВЫМ из
    накопителей сил, потому что он пишет флаги граней (faceL/faceR/crack*),
    которые читает forces_reset: при старом порядке (reset первым) флаги
    запаздывали на подшаг. Проходы параллельные, между ними implicit-барьер."""
    forces_zero()
    forces_contact(dt)
    forces_reset()
    forces_bonds(dt)
