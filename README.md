# Actividad didáctica 2-M4 — Solución de un problema de parqueaderos

Simulación de eventos discretos de los cajeros de pago del centro comercial
**Supercentro**. Cada salida tiene tres cajeros con filas independientes, sin
cambio de fila ni deserción. Cada cajero se modela como un sistema **M/M/1** y se
aplican las técnicas del Módulo 4: estado estable, número de réplicas,
escenarios de mejora, verificación/calibración/validación y eliminación del
estado transitorio.

## Estructura del repositorio

| Archivo | Descripción |
| --- | --- |
| `sim_parqueadero.py` | Modelo de eventos discretos (LEF con `heapq`), métricas y recursión de Lindley. |
| `analisis.py` | Corrida piloto, estado estable (media acumulada y Welch), número de réplicas, puntos A–E y generación de gráficas. |
| `fig/` | 11 figuras PNG generadas por el análisis (una por cada gráfica citada en el informe). |
| `resultados.json` | Resumen de los resultados del informe. Se regenera automáticamente al ejecutar `analisis.py`. |
| `Actividad2_M4_Problema_Parqueaderos.docx` | Informe con introducción, metodología, resultados, conclusiones, referencias y anexo de código. |
| `tarea.pdf` | Enunciado de la actividad y rúbrica de evaluación. |

## Requisitos

- Python 3.10 o superior
- `numpy`, `scipy`, `matplotlib`

```bash
pip install numpy scipy matplotlib
```

## Ejecución

Desde la raíz del repositorio:

```bash
python analisis.py
```

El script usa la semilla `20261124` (PCG64 de NumPy), por lo que los resultados
son reproducibles. Genera/sobrescribe la carpeta `fig/` y `resultados.json`.

Tiempo de simulación: 720 min (10:00–22:00). Al cierre no entran más usuarios,
pero se atiende a quienes ya están en fila.

## Resultados principales

- Transitorio: **52 usuarios por cajero** (≈ 246 min).
- Réplicas definitivas: **600** (estabilidad ±2 % y error relativo del IC 95 % < 5 %).
- Cajero más rápido: **Cajero 1** (3,580 min); más lento: **Cajero 3** (3,622 min).
  Diferencia no significativa (ANOVA p = 0,071; Kruskal-Wallis p = 0,125).
- Usuarios por salida: **≈ 466/día** (117 rápidos, 93 normales, 129 lentos, 128 muy lentos).
- Criterio acordado: `Wq ≤ 5 min`, `P(Wq > 10) ≤ 15 %`, `ρ ≤ 80 %`.
- Escenario recomendado: **E4 — 4 cajeros + pago ágil** (`Wq = 3,31 min`,
  `P(Wq > 10) = 11,3 %`, `ρ = 47,6 %`). Tres cajeros solos no cumplen.
- Al eliminar el transitorio, `Wq` sube de 13,16 a 14,36 min y `ρ` de 75,4 % a 77,6 %.

## Evidencias

Las figuras `f1`–`f11` están en `fig/` y también incrustadas en el documento
Word. `resultados.json` contiene los valores reportados en el informe; al volver
a ejecutar el análisis se sobrescribe con el detalle completo de todas las
réplicas.

## Pendientes de entrega (fuera del código)

1. Rellenar en el Word los marcadores de nombre de los dos estudiantes y del docente, y actualizar el índice.
2. Renombrar el informe a `Apellido_Nombre_problemaParqueadero.doc`.
3. Publicar este repositorio en GitHub y compartir el enlace.
4. Grabar el video de sustentación (problema, modelo, simulación, resultados y conclusiones).
5. Subir todo a la plataforma antes del domingo a las 23:59.
