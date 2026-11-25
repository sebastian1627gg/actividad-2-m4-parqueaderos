# -*- coding: utf-8 -*-
"""
Análisis completo - Actividad 2-M4 (parqueaderos Supercentro).
Genera figuras (carpeta fig/) y resultados (resultados.json).
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy import stats
from sim_parqueadero import (simular_cajero, simular_salida, metricas, lindley,
                             TIPOS, MEDIA_SERV, MEDIA_LLEG, PROB, T_SIM)

SEMILLA = 20261124
os.makedirs("fig", exist_ok=True)
C = dict(azul="#2a78d6", naranja="#eb6834", aqua="#1baf7a", amarillo="#eda100",
         magenta="#e87ba4", verde="#008300", violeta="#6250d6", rojo="#e34948",
         gris="#52514e", tinta="#0b0b0b")
SERIE = [C["azul"], C["naranja"], C["aqua"], C["amarillo"], C["magenta"]]
plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
                     "axes.edgecolor": "#8a8984", "axes.axisbelow": True, "axes.titleweight": "bold",
                     "axes.titlesize": 10, "legend.frameon": False})
R = {}   # resultados para el documento


def guardar(nombre, rect=(0, 0, 1, 1)):
    plt.tight_layout(rect=rect); plt.savefig(f"fig/{nombre}.png", dpi=170); plt.close()


# ===========================================================================
# 0. Corrida piloto: 1000 réplicas (días) x 3 cajeros
# ===========================================================================
rng = np.random.default_rng(SEMILLA)
R_PIL = 1000
pil = [simular_salida(rng, 3) for _ in range(R_PIL)]          # pil[r][cajero]

# ===========================================================================
# 1. Estado estable - técnica de la media (dentro de la corrida)
#    Promedio acumulado de TA y Wq usuario a usuario (réplica 1, cajero 1)
# ===========================================================================
reg = pil[0][0]
pTA = np.cumsum(reg["serv"]) / np.arange(1, len(reg["serv"]) + 1)
pWq = np.cumsum(reg["wq"]) / np.arange(1, len(reg["wq"]) + 1)
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
for a, serie, nom in [(ax[0], pTA, "Tiempo de atención TA"), (ax[1], pWq, "Tiempo en cola Wq")]:
    m = serie.mean(); s = serie.std(ddof=1)
    a.plot(serie, color=C["azul"], lw=1.6, label="Promedio acumulado")
    a.axhline(m, color=C["rojo"], ls="--", lw=1.2, label=f"Media general = {m:.2f}")
    a.axhline(m + s, color=C["verde"], ls="--", lw=1, label="Media ± desv. estándar")
    a.axhline(m - s, color=C["verde"], ls="--", lw=1)
    a.set_title(f"{nom} (réplica 1, cajero 1)")
    a.set_xlabel("Usuario atendido (evento de salida)"); a.set_ylabel("Promedio acumulado (min)")
    a.legend(fontsize=7.5, loc="lower right")
guardar("f1_media_acumulada_una_corrida")

# Método de Welch: promedio de Wq del usuario i sobre las 3000 corridas (1000x3)
N = 130
M = np.full((R_PIL * 3, N), np.nan)
k = 0
for r in range(R_PIL):
    for reg in pil[r]:
        n = min(N, len(reg["wq"])); M[k, :n] = reg["wq"][:n]; k += 1
wel = np.nanmean(M, axis=0)
v = 5
ma = np.array([wel[max(0, i - v):i + v + 1].mean() for i in range(N - v)])
# criterio de fin del transitorio: primer usuario a partir del cual la media
# móvil queda dentro de ±5 % del nivel estable (media de usuarios 60-125)
nivel = ma[60:].mean()
dentro = np.abs(ma - nivel) <= 0.05 * nivel
L_WARM = int(next(i for i in range(len(dentro)) if dentro[i:].all()))
t_warm = np.nanmean([[reg["lleg"][L_WARM] for reg in d if len(reg["lleg"]) > L_WARM] for d in pil])
R["warmup"] = dict(L=L_WARM, nivel=float(nivel), t_min=float(t_warm))

fig, ax = plt.subplots(figsize=(8, 3.4))
ax.plot(wel, color="#b7d3f6", lw=1, label="Wq promedio del usuario i (3000 corridas)")
ax.plot(ma, color=C["azul"], lw=2, label=f"Media móvil (ventana {2*v+1})")
ax.axhline(nivel, color=C["rojo"], ls="--", lw=1.2, label=f"Nivel estable = {nivel:.2f} min")
ax.axhspan(nivel * 0.95, nivel * 1.05, color=C["verde"], alpha=0.12, label="Banda ±5 %")
ax.axvline(L_WARM, color=C["tinta"], ls=":", lw=1.2)
ax.annotate(f"Fin del transitorio\nusuario {L_WARM} (≈ {t_warm:.0f} min)",
            xy=(L_WARM, nivel * 0.95), xytext=(L_WARM + 12, nivel * 0.45),
            arrowprops=dict(arrowstyle="->", color=C["gris"]), fontsize=8)
ax.set_xlabel("Índice del usuario en la jornada"); ax.set_ylabel("Wq (min)")
ax.set_title("Identificación del estado transitorio (método de Welch sobre la media)")
ax.legend(fontsize=7.5, loc="lower right")
guardar("f2_welch_transitorio")

# ===========================================================================
# 2. Número de réplicas - media acumulada entre réplicas
# ===========================================================================
Wq_dia = np.array([np.mean([metricas(reg)["Wq"] for reg in d]) for d in pil])
TA_dia = np.array([np.mean([metricas(reg)["TA"] for reg in d]) for d in pil])
idx = np.arange(1, R_PIL + 1)
cmW = np.cumsum(Wq_dia) / idx
sdW = np.array([Wq_dia[:i].std(ddof=1) if i > 1 else np.nan for i in idx])
hw = stats.t.ppf(0.975, idx - 1) * sdW / np.sqrt(idx)
err_rel = hw / cmW
final = cmW[-1]
estab = np.abs(cmW - final) <= 0.02 * final
n_estab = int(next(i for i in range(R_PIL) if estab[i:].all())) + 1
n_err = int(np.argmax(err_rel <= 0.05)) + 1                  # semiancho ≤ 5 %
# fórmula clásica a partir de 30 réplicas piloto
s0, x0 = Wq_dia[:30].std(ddof=1), Wq_dia[:30].mean()
n_form = int(np.ceil((1.96 * s0 / (0.05 * x0)) ** 2))
N_REP = int(np.ceil(max(n_estab, n_err, n_form) / 50) * 50)
R["replicas"] = dict(n_estab=n_estab, n_err=n_err, n_form=n_form, N=N_REP,
                     s0=float(s0), x0=float(x0), final=float(final))

fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.5))
a = ax[0]
a.plot(idx, Wq_dia, color="#cde2fb", lw=0.6, label="Wq promedio de cada réplica")
a.plot(idx, cmW, color=C["azul"], lw=2, label="Media acumulada")
a.axhline(final, color=C["rojo"], ls="--", lw=1.2, label=f"Media general = {final:.2f}")
a.axhline(final * 1.02, color=C["verde"], ls="--", lw=1, label="±2 % de la media")
a.axhline(final * 0.98, color=C["verde"], ls="--", lw=1)
a.axvline(N_REP, color=C["tinta"], ls=":", lw=1.2)
a.text(N_REP + 12, 2, f"n = {N_REP}", fontsize=8, weight="bold")
a.set_ylim(0, final * 2.6)
a.set_xlabel("Número de réplicas"); a.set_ylabel("Wq (min)")
a.set_title("Media acumulada de Wq por réplicas"); a.legend(fontsize=7, loc="upper right", ncol=2)
a = ax[1]
a.plot(idx[1:], 100 * err_rel[1:], color=C["naranja"], lw=1.8)
a.axhline(5, color=C["rojo"], ls="--", lw=1.2, label="Meta: error relativo ≤ 5 %")
a.axvline(N_REP, color=C["tinta"], ls=":", lw=1.2)
a.set_ylim(0, 30); a.set_xlabel("Número de réplicas"); a.set_ylabel("Semiancho IC 95 % / media (%)")
a.set_title("Precisión del estimador de Wq"); a.legend(fontsize=7.5)
guardar("f3_numero_replicas")

# ===========================================================================
# 3. Corrida definitiva (N_REP réplicas) - escenario actual 3 cajeros
# ===========================================================================
fin = pil[:N_REP]
met = [[metricas(reg) for reg in d] for d in fin]          # sin eliminar transitorio
metW = [[metricas(reg, desde=L_WARM) for reg in d] for d in fin]   # con eliminación


def ic(x):
    x = np.asarray(x); m = x.mean(); h = stats.t.ppf(.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return float(m), float(m - h), float(m + h)


# --- P1: tiempo promedio de atención por cajero ---------------------------
p1 = {}
for c in range(3):
    p1[c + 1] = {k: ic([met[r][c][k] for r in range(N_REP)]) for k in
                 ["TA", "Wq", "W", "rho", "L", "n", "P5"]}
    p1[c + 1]["TA_tipo"] = [float(np.nanmean([met[r][c]["TA_tipo"][t] for r in range(N_REP)])) for t in range(4)]
TAs = [[met[r][c]["TA"] for r in range(N_REP)] for c in range(3)]
F, pval = stats.f_oneway(*TAs)
KW = stats.kruskal(*TAs)
orden = sorted(p1, key=lambda c: p1[c]["TA"][0])
R["P1"] = dict(tabla=p1, F=float(F), p=float(pval), KW_p=float(KW.pvalue),
               menor=orden[0], mayor=orden[-1])

fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.4))
a = ax[0]
for c in range(3):
    m, lo, hi = p1[c + 1]["TA"]
    a.bar(c, m, color=SERIE[c], width=0.55, edgecolor="white", linewidth=2)
    a.errorbar(c, m, yerr=[[m - lo], [hi - m]], color=C["tinta"], capsize=5, lw=1.2)
    a.text(c, hi + 0.03, f"{m:.3f}", ha="center", fontsize=8)
a.set_xticks(range(3), [f"Cajero {c+1}" for c in range(3)])
a.set_ylim(3.3, 3.85); a.set_ylabel("TA promedio (min)")
a.set_title("Tiempo promedio de atención (IC 95 %)")
a = ax[1]
bp = a.boxplot(TAs, tick_labels=[f"Cajero {c+1}" for c in range(3)], patch_artist=True,
               medianprops=dict(color=C["tinta"]), flierprops=dict(markersize=2))
for p, col in zip(bp["boxes"], SERIE): p.set_facecolor(col); p.set_alpha(0.55)
a.set_ylabel("TA promedio diario (min)")
a.set_title(f"Distribución por réplica (ANOVA p = {pval:.3f})")
guardar("f4_P1_atencion_cajeros")

# --- P2: promedio de usuarios por tipo ------------------------------------
cnt = np.array([[met[r][c]["n_tipo"] for c in range(3)] for r in range(N_REP)])  # R x 3 x 4
por_cajero = cnt.mean(axis=0)              # 3 x 4
total = cnt.sum(axis=1)                    # R x 4
R["P2"] = dict(por_cajero=por_cajero.tolist(),
               total=[ic(total[:, t]) for t in range(4)],
               total_dia=ic(total.sum(axis=1)),
               prom_cajero_tipo=[float(cnt[:, :, t].mean()) for t in range(4)],
               pct=(total.mean(0) / total.mean(0).sum()).tolist())
fig, ax = plt.subplots(figsize=(8, 3.4))
w = 0.2
for t in range(4):
    for c in range(3):
        ax.bar(c + (t - 1.5) * w, por_cajero[c, t], w - 0.02, color=SERIE[t],
               label=TIPOS[t] if c == 0 else None)
        ax.text(c + (t - 1.5) * w, por_cajero[c, t] + 0.6, f"{por_cajero[c,t]:.1f}", ha="center", fontsize=7)
ax.set_xticks(range(3), [f"Cajero {c+1}" for c in range(3)])
ax.set_ylabel("Usuarios por día (promedio)")
ax.set_title(f"Usuarios promedio por tipo y cajero ({N_REP} réplicas)")
ax.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12))
guardar("f5_P2_usuarios_tipo")

# ===========================================================================
# 4. P3 - Escenarios de mejora y criterio de decisión
# ===========================================================================
MS_UI = MEDIA_SERV * np.array([1, 1, 0.75, 0.75])   # pago asistido / QR para lentos
esc = [("E0: 3 cajeros (actual)", 3, MEDIA_SERV),
       ("E1: 4 cajeros", 4, MEDIA_SERV),
       ("E2: 5 cajeros", 5, MEDIA_SERV),
       ("E3: 3 cajeros + pago ágil", 3, MS_UI),
       ("E4: 4 cajeros + pago ágil", 4, MS_UI)]
rng_e = np.random.default_rng(SEMILLA + 1)
res_esc = []
for nom, c, ms in esc:
    rep = []
    for _ in range(N_REP):
        regs = simular_salida(rng_e, c, media_serv=ms)
        wq_all = np.concatenate([g["wq"][L_WARM * 3 // c:] for g in regs])
        mm = [metricas(g, desde=L_WARM * 3 // c) for g in regs]
        rep.append(dict(Wq=np.mean([m["Wq"] for m in mm]), P10=(wq_all > 10).mean(),
                        rho=np.mean([m["rho"] for m in mm]),
                        L=np.mean([m["L"] for m in mm]),
                        Wq_lento=np.nanmean([m["Wq_tipo"][3] for m in mm])))
    d = dict(nombre=nom, c=c, **{k: ic([x[k] for x in rep]) for k in rep[0]})
    # cumple: el IC 95 % completo queda dentro del criterio; marginal: el IC lo contiene
    lim = dict(Wq=5, P10=0.15, rho=0.80)
    if all(d[k][2] <= v for k, v in lim.items()): d["estado"] = "Cumple"
    elif all(d[k][1] <= v for k, v in lim.items()): d["estado"] = "Marginal"
    else: d["estado"] = "No cumple"
    d["cumple"] = d["estado"] == "Cumple"
    res_esc.append(d)
R["P3"] = res_esc

fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
nombres = [e["nombre"].split(":")[0] for e in res_esc]
for i, e in enumerate(res_esc):
    m, lo, hi = e["Wq"]
    col = {"Cumple": C["aqua"], "Marginal": C["amarillo"], "No cumple": C["naranja"]}[e["estado"]]
    ax[0].bar(i, m, color=col, width=0.6, edgecolor="white", linewidth=2)
    ax[0].errorbar(i, m, yerr=[[m - lo], [hi - m]], color=C["tinta"], capsize=4, lw=1)
    ax[0].text(i, hi + 0.4, f"{m:.1f}", ha="center", fontsize=8)
    m2 = 100 * e["P10"][0]
    ax[1].bar(i, m2, color=col, width=0.6, edgecolor="white", linewidth=2)
    ax[1].text(i, m2 + 0.8, f"{m2:.0f} %", ha="center", fontsize=8)
ax[0].axhline(5, color=C["rojo"], ls="--", lw=1.2, label="Criterio: Wq ≤ 5 min")
ax[1].axhline(15, color=C["rojo"], ls="--", lw=1.2, label="Criterio: P(Wq > 10) ≤ 15 %")
for a, t, yl in [(ax[0], "Tiempo promedio en cola", "Wq (min)"),
                 (ax[1], "Usuarios que esperan más de 10 min", "%")]:
    a.set_xticks(range(len(res_esc)), nombres); a.set_title(t); a.set_ylabel(yl); a.legend(fontsize=7.5)
fig.text(0.5, 0.01, "Verde: cumple (IC 95 % dentro del criterio) · Amarillo: marginal (el IC contiene el límite) · Naranja: no cumple", ha="center", fontsize=8, color=C["gris"])
guardar("f6_P3_escenarios", rect=(0, 0.05, 1, 1))

# ===========================================================================
# 5. P4 - Verificación, calibración y validación
# ===========================================================================
ver = {}
# V1 acoplamiento LEF vs Lindley
errs = [np.abs(lindley(reg["lleg"], reg["serv"]) - reg["wq"]).max() for d in fin[:100] for reg in d]
ver["lindley_max"] = float(max(errs))
# V2 conservación de usuarios y FIFO
ver["fifo_ok"] = bool(all((np.diff(reg["ini"]) >= -1e-9).all() and np.isfinite(reg["fin"]).all()
                          for d in fin for reg in d))
# V3 M/M/1 puro (un solo tipo: llegada 3, servicio 1) en corrida larga vs teoría
rng_v = np.random.default_rng(7)
mm1 = []
for _ in range(20):
    g = simular_cajero(rng_v, t_sim=20000, prob=np.array([1, 0, 0, 0.]))
    mm = metricas(g, t_sim=20000); mm1.append((mm["Wq"], mm["W"], mm["L"], mm["rho"]))
mm1 = np.array(mm1).mean(0)
lam, mu = 1 / 3, 1.0
teo = (lam / (mu * (mu - lam)), 1 / (mu - lam), lam / (mu - lam), lam / mu)
ver["mm1"] = dict(sim=mm1.tolist(), teo=list(teo))
# V4 bondad de ajuste de servicios por tipo (K-S) y mezcla de tipos (chi²)
serv_all = np.concatenate([reg["serv"] for d in fin for reg in d])
tipo_all = np.concatenate([reg["tipo"] for d in fin for reg in d])
ks = [float(stats.kstest(serv_all[tipo_all == t][:2000], "expon", args=(0, MEDIA_SERV[t])).pvalue) for t in range(4)]
obs = np.bincount(tipo_all, minlength=4)
# Nota: las llegadas de tipos con media entre llegadas mayor se "estiran"; la
# proporción de usuarios generados sí sigue PROB (sorteo por llegada)
chi = stats.chisquare(obs, PROB * obs.sum())
ver["ks_p"] = ks; ver["chi_p"] = float(chi.pvalue); ver["obs_pct"] = (obs / obs.sum()).tolist()
# V5 Ley de Little en la jornada: L ≈ λ W
Ls = np.mean([m["L"] for d in met for m in d])
lam_obs = np.mean([m["n"] for d in met for m in d]) / T_SIM
Ws = np.mean([m["W"] for d in met for m in d])
ver["little"] = dict(L=float(Ls), lamW=float(lam_obs * Ws))
# V6 utilización teórica
EA = (PROB * MEDIA_LLEG).sum(); ES = (PROB * MEDIA_SERV).sum()
ver["rho_teo"] = float(ES / EA); ver["rho_sim"] = ic([m["rho"] for d in metW for m in d])
ver["EA"] = float(EA); ver["ES"] = float(ES)
R["ver"] = ver

# Calibración: parámetros simulados vs. especificados (dato "observado")
lleg_all = np.concatenate([np.diff(np.r_[0, reg["lleg"]]) for d in fin for reg in d])
cal = []
for t in range(4):
    cal.append(dict(tipo=TIPOS[t], serv_esp=float(MEDIA_SERV[t]),
                    serv_sim=float(serv_all[tipo_all == t].mean()),
                    lleg_esp=float(MEDIA_LLEG[t]),
                    lleg_sim=float(lleg_all[tipo_all == t].mean()),
                    p_esp=float(PROB[t]), p_sim=float(obs[t] / obs.sum())))
R["cal"] = cal

# Calibración - test de sensibilidad ±10 % (un parámetro a la vez)
def wq_escenario(ms=MEDIA_SERV, ml=MEDIA_LLEG, pr=PROB, reps=200, seed=99):
    g = np.random.default_rng(seed)   # números aleatorios comunes
    out = []
    for _ in range(reps):
        for reg in simular_salida(g, 3, media_serv=ms, media_lleg=ml, prob=pr):
            out.append(reg["wq"][L_WARM:].mean() if len(reg["wq"]) > L_WARM else np.nan)
    return float(np.nanmean(out))

base = wq_escenario()
sens = []
for t in range(4):
    for nombre, vec in [("servicio", MEDIA_SERV), ("llegada", MEDIA_LLEG)]:
        res = []
        for f in (0.9, 1.1):
            v2 = vec.copy(); v2[t] *= f
            res.append(wq_escenario(ms=v2) if nombre == "servicio" else wq_escenario(ml=v2))
        sens.append(dict(par=f"Media de {nombre} – {TIPOS[t]}", menos=res[0], mas=res[1]))
# proporción lento/muy lento (incertidumbre del 55 %)
for lab, pr in [("Mezcla 35 % lento / 20 % muy lento", [0.25, 0.20, 0.35, 0.20]),
                ("Mezcla 20 % lento / 35 % muy lento", [0.25, 0.20, 0.20, 0.35])]:
    sens.append(dict(par=lab, valor=wq_escenario(pr=np.array(pr))))
R["sens"] = dict(base=base, items=sens)

fig, ax = plt.subplots(figsize=(8.5, 3.8))
it = sorted([s for s in sens if "menos" in s], key=lambda s: abs(s["mas"] - s["menos"]))
for i, s in enumerate(it):
    ax.barh(i, s["menos"] - base, color=C["azul"], height=0.6, label="−10 %" if i == 0 else None)
    ax.barh(i, s["mas"] - base, color=C["naranja"], height=0.6, label="+10 %" if i == 0 else None)
ax.set_yticks(range(len(it)), [s["par"] for s in it], fontsize=8)
ax.axvline(0, color=C["tinta"], lw=0.8)
ax.set_xlabel(f"Cambio en Wq frente a la base ({base:.2f} min)")
ax.set_title("Calibración – test de sensibilidad (±10 %, un parámetro a la vez)")
ax.legend(fontsize=8, loc="lower right")
guardar("f7_sensibilidad")

# Validación contra aproximación analítica G/G/1 (Kingman/Allen-Cunneen)
EA2 = (PROB * 2 * MEDIA_LLEG ** 2).sum(); ES2 = (PROB * 2 * MEDIA_SERV ** 2).sum()
ca2 = (EA2 - EA ** 2) / EA ** 2; cs2 = (ES2 - ES ** 2) / ES ** 2
rho = ES / EA
king = rho / (1 - rho) * (ca2 + cs2) / 2 * ES
# corrida larga (estado estable puro) para contrastar
rng_l = np.random.default_rng(11)
largas = [simular_cajero(rng_l, t_sim=50000) for _ in range(10)]
wq_largo = ic([g["wq"][500:].mean() for g in largas])
R["val"] = dict(ca2=float(ca2), cs2=float(cs2), rho=float(rho), kingman=float(king),
                wq_largo=wq_largo,
                wq_dia_sin=ic([np.mean([m["Wq"] for m in d]) for d in met]),
                wq_dia_con=ic([np.mean([m["Wq"] for m in d]) for d in metW]))

# Diagrama del proceso de revisión (verificación -> calibración -> validación)
fig, ax = plt.subplots(figsize=(9.5, 4.4)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 5)
cajas = [
    (0.1, 3.3, "Sistema real\nSupercentro\n(3 cajeros, 4 tipos)", "#f0efec"),
    (2.55, 3.3, "Modelo conceptual\nM/M/1 x 3, LEF,\nsupuestos", "#cde2fb"),
    (5.0, 3.3, "VERIFICACIÓN\n¿Está bien\nprogramado?", "#b7d3f6"),
    (7.45, 3.3, "CALIBRACIÓN\n¿Parámetros\najustados?", "#fde3b3"),
    (7.45, 0.6, "VALIDACIÓN\n¿Representa el\nsistema real?", "#c3eedc"),
    (5.0, 0.6, "Experimentación\nestado estable,\nn réplicas, escenarios", "#e8e6f8"),
    (2.55, 0.6, "Decisión\ncriterio Wq ≤ 5 min,\nP(Wq>10) ≤ 15 %", "#f0efec")]
for x, y, t, col in cajas:
    ax.add_patch(FancyBboxPatch((x, y), 2.2, 1.25, boxstyle="round,pad=0.05,rounding_size=0.15",
                                fc=col, ec="#8a8984", lw=1))
    ax.text(x + 1.1, y + 0.62, t, ha="center", va="center", fontsize=8.3)
fl = dict(arrowstyle="-|>", color=C["tinta"], lw=1.2)
for (x1, y1), (x2, y2) in [((2.3, 3.92), (2.55, 3.92)), ((4.75, 3.92), (5.0, 3.92)),
                           ((7.2, 3.92), (7.45, 3.92)), ((8.55, 3.3), (8.55, 1.9)),
                           ((7.45, 1.22), (7.2, 1.22)), ((5.0, 1.22), (4.75, 1.22))]:
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=fl)
# retroalimentaciones
ax.annotate("", xy=(6.1, 3.3), xytext=(6.1, 2.6), arrowprops=dict(arrowstyle="-|>", color=C["rojo"], lw=1, ls="--"))
ax.text(6.2, 2.85, "falla → corregir código", fontsize=7, color=C["rojo"])
ax.annotate("", xy=(8.0, 1.9), xytext=(8.0, 3.3), arrowprops=dict(arrowstyle="<|-", color=C["rojo"], lw=1, ls="--"))
ax.text(6.25, 2.3, "no valida → recalibrar", fontsize=7, color=C["rojo"])
ax.text(5.0, 4.85, "Proceso de revisión del modelo de cajeros", ha="center", fontsize=10, weight="bold")
txt = (f"V: LEF = Lindley (error {ver['lindley_max']:.0e}); FIFO; M/M/1 vs teoría; K-S; χ²; Little\n"
       f"C: parámetros simulados vs. tabla; sensibilidad ±10 %     V: G/G/1 analítico; datos reales 80/20")
ax.text(5.0, 0.12, txt, ha="center", fontsize=7.3, color=C["gris"])
guardar("f8_proceso_vcv")

# Gráfica de verificación: M/M/1 teórico vs simulado
fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.3))
labs = ["Wq", "W", "L", "ρ"]
x = np.arange(4)
ax[0].bar(x - 0.18, teo, 0.34, color=C["azul"], label="Teoría M/M/1")
ax[0].bar(x + 0.18, mm1, 0.34, color=C["naranja"], label="Simulación")
for i in range(4): ax[0].text(i + 0.18, mm1[i] + 0.03, f"{mm1[i]:.3f}", ha="center", fontsize=7)
ax[0].set_xticks(x, labs); ax[0].set_title("Verificación: M/M/1 (λ=1/3, μ=1)"); ax[0].legend(fontsize=7.5)
cats = [c["tipo"] for c in cal]
ax[1].bar(np.arange(4) - 0.18, [c["serv_esp"] for c in cal], 0.34, color=C["azul"], label="Especificado (tabla)")
ax[1].bar(np.arange(4) + 0.18, [c["serv_sim"] for c in cal], 0.34, color=C["naranja"], label="Simulado")
ax[1].set_xticks(range(4), cats); ax[1].set_ylabel("Media de servicio (min)")
ax[1].set_title("Calibración: entradas simuladas vs. tabla"); ax[1].legend(fontsize=7.5)
guardar("f9_verificacion_calibracion")

# ===========================================================================
# 6. P5 - Eliminación del estado transitorio: antes y después
# ===========================================================================
# (a) media acumulada usuario a usuario (Welch) antes / después
fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.5), sharey=True)
cum_antes = np.cumsum(wel) / np.arange(1, N + 1)
cum_desp = np.cumsum(wel[L_WARM:]) / np.arange(1, N - L_WARM + 1)
for a, serie, x0, tit in [(ax[0], cum_antes, 0, "ANTES: con estado transitorio"),
                          (ax[1], cum_desp, L_WARM, f"DESPUÉS: se eliminan los primeros {L_WARM} usuarios")]:
    m = serie.mean(); s = serie.std(ddof=1)
    xs = np.arange(x0, x0 + len(serie))
    a.plot(xs, serie, color=C["azul"], lw=1.8, label="Media acumulada de Wq")
    a.axhline(m, color=C["rojo"], ls="--", lw=1.2, label=f"Media = {m:.2f}")
    a.axhline(m + s, color=C["verde"], ls="--", lw=1, label=f"± desv. est. ({s:.2f})")
    a.axhline(m - s, color=C["verde"], ls="--", lw=1)
    a.set_title(tit); a.set_xlabel("Índice del usuario"); a.legend(fontsize=7.5, loc="lower right")
    a.set_xlim(0, N)
ax[0].set_ylabel("Wq (min)")
guardar("f10_antes_despues_usuarios")
R["P5"] = dict(antes=dict(m=float(cum_antes.mean()), s=float(cum_antes.std(ddof=1)), fin=float(cum_antes[-1])),
               despues=dict(m=float(cum_desp.mean()), s=float(cum_desp.std(ddof=1)), fin=float(cum_desp[-1])))

# (b) media acumulada entre réplicas antes / después y tabla por cajero
WqA = np.array([np.mean([m["Wq"] for m in d]) for d in met])
WqD = np.array([np.mean([m["Wq"] for m in d]) for d in metW])
TAA = np.array([np.mean([m["TA"] for m in d]) for d in met])
TAD = np.array([np.mean([m["TA"] for m in d]) for d in metW])
ii = np.arange(1, N_REP + 1)
fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.4))
for a, A, D, nom in [(ax[0], WqA, WqD, "Wq (min)"), (ax[1], TAA, TAD, "TA (min)")]:
    a.plot(ii, np.cumsum(A) / ii, color=C["gris"], lw=1.6, label=f"Antes (media {A.mean():.2f})")
    a.plot(ii, np.cumsum(D) / ii, color=C["azul"], lw=1.8, label=f"Después (media {D.mean():.2f})")
    a.axhline(A.mean(), color=C["gris"], ls="--", lw=0.9)
    a.axhline(D.mean(), color=C["rojo"], ls="--", lw=0.9)
    a.set_xlabel("Réplica"); a.set_ylabel(nom); a.legend(fontsize=7.5)
    a.set_title(f"Media acumulada de {nom.split()[0]} entre réplicas")
guardar("f11_antes_despues_replicas")

tabla5 = {}
for c in range(3):
    tabla5[c + 1] = {k: (ic([met[r][c][k] for r in range(N_REP)]),
                         ic([metW[r][c][k] for r in range(N_REP)])) for k in ["TA", "Wq", "W", "rho", "L"]}
R["P5"]["tabla"] = tabla5
R["P5"]["global"] = {k: (ic([np.mean([m[k] for m in d]) for d in met]),
                         ic([np.mean([m[k] for m in d]) for d in metW])) for k in ["TA", "Wq", "W", "rho", "L"]}

# Wq por tipo de usuario (después del transitorio)
R["wq_tipo"] = [float(np.nanmean([m["Wq_tipo"][t] for d in metW for m in d])) for t in range(4)]
R["W_tipo"] = [float(np.nanmean([m["Wq_tipo"][t] for d in metW for m in d]) + MEDIA_SERV[t]) for t in range(4)]

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(R, f, ensure_ascii=False, indent=1, default=float)
print(json.dumps(R, ensure_ascii=False, indent=1, default=float)[:6000])
