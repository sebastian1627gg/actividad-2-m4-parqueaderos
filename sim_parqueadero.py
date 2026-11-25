# -*- coding: utf-8 -*-
"""
Actividad didáctica 2-M4 - Solución de un problema de parqueaderos (Supercentro)
Simulación de eventos discretos de los cajeros de pago (cada cajero = M/M/1
con fila propia, sin cambio de fila ni deserción).

Supuestos (ver documento):
- "Exponencial de uso del servicio" = media del tiempo de servicio (min).
- "Media de llegada" = media del tiempo entre llegadas (min) cuando el usuario
  que llega es de ese tipo (misma lógica del código guía de la Actividad 2-M3).
- 25 % rápidos, 20 % normales, 55 % restante repartido en partes iguales:
  27,5 % lentos y 27,5 % muy lentos.
- Jornada de 720 min (10:00 a 22:00). Al cerrar no entran más usuarios y se
  atiende a quienes ya están en fila.
- Escenarios con c cajeros: la demanda total de la salida (la de 3 cajeros) se
  reparte entre c cajeros, es decir, las medias entre llegadas se multiplican
  por c/3.
"""
import heapq
import numpy as np

TIPOS = ["Rápido", "Normal", "Lento", "Muy lento"]
MEDIA_SERV = np.array([1.0, 3.0, 4.0, 6.0])     # min
MEDIA_LLEG = np.array([3.0, 3.0, 5.0, 7.0])     # min
PROB = np.array([0.25, 0.20, 0.275, 0.275])
T_SIM = 720.0                                     # min (12 h)


# ---------------------------------------------------------------------------
# 1. Modelo de un cajero con lista de eventos futuros (LEF)
# ---------------------------------------------------------------------------
def simular_cajero(rng, t_sim=T_SIM, escala_lleg=1.0, media_serv=MEDIA_SERV,
                   media_lleg=MEDIA_LLEG, prob=PROB):
    """Simula un cajero M/M/1 (fila FIFO). Devuelve un dict con un registro
    por usuario atendido y la traza del número en sistema."""
    lef = []                      # lista de eventos futuros (tiempo, tipo_evento, id)
    reloj = 0.0
    fila = []                     # ids en cola (FIFO)
    ocupado = False
    en_servicio = None
    # registros por usuario
    tipo, t_lleg, t_ini, t_fin, t_serv = [], [], [], [], []
    traza_t, traza_n = [0.0], [0]  # para L (número en el sistema)
    n_sis = 0

    def programar_llegada(t_actual):
        k = rng.choice(4, p=prob)                       # tipo del próximo usuario
        tll = rng.exponential(media_lleg[k] * escala_lleg)
        heapq.heappush(lef, (t_actual + tll, 0, len(tipo)))
        tipo.append(k); t_lleg.append(t_actual + tll)
        t_serv.append(rng.exponential(media_serv[k]))
        t_ini.append(np.nan); t_fin.append(np.nan)

    programar_llegada(0.0)
    while lef:
        reloj, ev, i = heapq.heappop(lef)
        if ev == 0:                                     # ---- LLEGADA
            if reloj > t_sim:                           # llegó con la puerta cerrada
                tipo.pop(); t_lleg.pop(); t_serv.pop(); t_ini.pop(); t_fin.pop()
                continue
            n_sis += 1; traza_t.append(reloj); traza_n.append(n_sis)
            programar_llegada(reloj)
            if not ocupado:
                ocupado = True; en_servicio = i; t_ini[i] = reloj
                heapq.heappush(lef, (reloj + t_serv[i], 1, i))
            else:
                fila.append(i)
        else:                                           # ---- SALIDA
            t_fin[i] = reloj
            n_sis -= 1; traza_t.append(reloj); traza_n.append(n_sis)
            if fila:
                j = fila.pop(0); en_servicio = j; t_ini[j] = reloj
                heapq.heappush(lef, (reloj + t_serv[j], 1, j))
            else:
                ocupado = False; en_servicio = None
    tipo = np.array(tipo); t_lleg = np.array(t_lleg); t_ini = np.array(t_ini)
    t_fin = np.array(t_fin); t_serv = np.array(t_serv)
    return dict(tipo=tipo, lleg=t_lleg, ini=t_ini, fin=t_fin, serv=t_serv,
                wq=t_ini - t_lleg, w=t_fin - t_lleg,
                traza_t=np.array(traza_t), traza_n=np.array(traza_n),
                t_final=reloj)


def metricas(reg, desde=0, t_sim=T_SIM):
    """Métricas de un cajero. 'desde' = nº de usuarios iniciales a descartar
    (eliminación del estado transitorio)."""
    s = slice(desde, None)
    wq, w, ts, tp = reg["wq"][s], reg["w"][s], reg["serv"][s], reg["tipo"][s]
    # utilización: tiempo ocupado dentro de la jornada / jornada
    # t0 = instante desde el que se mide (llegada del primer usuario retenido)
    t0 = reg["lleg"][desde] if desde > 0 else 0.0
    ini, fin = reg["ini"], reg["fin"]
    ocup = np.clip(np.minimum(fin, t_sim) - np.clip(ini, t0, t_sim), 0, None).sum()
    # L promedio en el sistema (área bajo la curva entre t0 y t_sim)
    tt, nn = reg["traza_t"], reg["traza_n"]
    m = tt <= t_sim
    tt2 = np.append(np.clip(tt[m], t0, None), t_sim); nn2 = nn[m]
    L = np.sum(np.diff(tt2) * nn2) / (t_sim - t0)
    return dict(n=len(wq), TA=ts.mean(), Wq=wq.mean(), W=w.mean(),
                rho=ocup / (t_sim - t0), L=L, P5=(wq > 5).mean(), Wq_max=wq.max(),
                n_tipo=np.bincount(tp, minlength=4),
                TA_tipo=np.array([ts[tp == k].mean() if (tp == k).any() else np.nan
                                  for k in range(4)]),
                Wq_tipo=np.array([wq[tp == k].mean() if (tp == k).any() else np.nan
                                  for k in range(4)]))


# ---------------------------------------------------------------------------
# 2. Verificación por acoplamiento: recursión de Lindley (implementación 2)
# ---------------------------------------------------------------------------
def lindley(lleg, serv):
    """Wq_{n+1} = max(0, Wq_n + S_n - (A_{n+1} - A_n))"""
    wq = np.zeros(len(lleg))
    for n in range(1, len(lleg)):
        wq[n] = max(0.0, wq[n - 1] + serv[n - 1] - (lleg[n] - lleg[n - 1]))
    return wq


def simular_salida(rng, c=3, **kw):
    """Simula los c cajeros independientes de una salida."""
    return [simular_cajero(rng, escala_lleg=c / 3, **kw) for _ in range(c)]
