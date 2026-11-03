"""B5/B6 sobre la segmentación propia (cells_pred): une color + textura, ajusta con train, valida con
leave-one-patient-out (LOPO) y evalúa en test contra el GT.

Reglas que se respetan:
- Modelos, umbrales y atributos se ajustan solo con células de train.
- test usa los modelos guardados, sin reajustar nada.
- El GT (clase_gt) solo se usa para ajustar en train y para medir; nunca para decidir en test.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from texture_eval.rules_classifier import (definir_estrategias, evaluar_lopo, ajustar_final,
                                           marcar, guardar_metodos, cargar_metodos)
from texture_eval.seg_metrics import evaluar_imagen, metricas, ponderado

RAIZ = Path(__file__).resolve().parents[1]
RESULTS = Path(__file__).resolve().parent / "results"
RUTA_RANKING_COLOR = RAIZ / "color_stain" / "results" / "c4_ranking_color.csv"      # lo genera C (C4)
RUTA_SEL_TEXTURA = RESULTS / "b2_atributos_seleccionados.csv"                       # lo genera el notebook B (B2)

CLASES = ["Parasitized", "Uninfected"]


def cargar_seleccion():
    """Atributos elegidos con train sobre el GT (color: C4, textura: B2). Quedan fijos."""
    for ruta in (RUTA_RANKING_COLOR, RUTA_SEL_TEXTURA):
        if not ruta.exists():
            raise FileNotFoundError(f"Falta {ruta}")
    rank = pd.read_csv(RUTA_RANKING_COLOR)
    sel_color = rank.loc[rank["seleccionada"], "caracteristica"].tolist()
    sel_tex = pd.read_csv(RUTA_SEL_TEXTURA)["feature"].tolist()
    return sel_color, sel_tex


def unir_features(textura, color):
    """Une textura y color por cell_id (sin etiquetas). n_pixeles ya viene en textura."""
    return textura.merge(color.drop(columns=["n_pixeles"]), on="cell_id", how="inner")


def con_etiquetas(tabla, etiquetas):
    """Solo train: agrega patient_id y y (1 = parasitada) usando etiquetas_train.csv (clase_gt).
    Se conservan las células pareadas con un eritrocito Parasitized o Uninfected del GT;
    las 'sin_pareja' y las de glóbulos blancos se dejan fuera del ajuste (etiqueta dudosa)."""
    m = tabla.merge(etiquetas[["cell_id", "patient_id", "clase_gt"]], on="cell_id", how="inner")
    m = m[m["clase_gt"].isin(CLASES)].reset_index(drop=True)
    m["y"] = (m["clase_gt"] == "Parasitized").astype(int)
    return m


def entrenar(m, sel_color, sel_tex, metodo):
    """LOPO de las 7 estrategias + modelos finales (color, textura, color+textura) con todo train.
    Guarda los modelos y las tablas en texture_eval/results/. Devuelve el resumen LOPO."""
    print(f"Ajuste con {len(m)} células pred de train | parasitadas: {int(m['y'].sum())} | "
          f"pacientes: {m['patient_id'].nunique()}")
    loo, res = evaluar_lopo(m, definir_estrategias(sel_color, sel_tex))

    metodos = {
        "General + color":         ajustar_final(m, sel_color),
        "General + textura":       ajustar_final(m, sel_tex),
        "General + color+textura": ajustar_final(m, sel_color + sel_tex),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    guardar_metodos(metodos, RESULTS / f"metodos_{metodo}.joblib")
    res.round(4).to_csv(RESULTS / f"b5_pred_{metodo}_lopo.csv")
    loo.to_csv(RESULTS / f"b5_pred_{metodo}_lopo_por_paciente.csv", index=False)
    return res


def evaluar_test(tabla_test, guardados, metodo):
    """Carga los modelos de train y los aplica a test, sin reajustar.
    tabla_test: textura + color de las células de test (sin etiquetas).
    guardados: una entrada por imagen con 'patient_id', 'image_id', 'celulas' (filas en el orden de las
    etiquetas 1..N), 'etiquetas' (mapa de la predicción), 'campo' y 'gt' (mapa GT de parasitadas)."""
    ruta = RESULTS / f"metodos_{metodo}.joblib"
    if not ruta.exists():
        raise FileNotFoundError(f"Falta {ruta}: se genera al correr main('train').")
    metodos = cargar_metodos(ruta)

    detalle = {}
    for nombre, M in metodos.items():
        marcadas = marcar(tabla_test, M)                       # cell_id marcados como parasitadas
        filas = []
        for g in guardados:
            ids = [k for k, f in enumerate(g["celulas"], start=1) if f["cell_id"] in marcadas]
            pred = np.where(np.isin(g["etiquetas"], ids), g["etiquetas"], 0)
            r = evaluar_imagen(g["gt"], pred, campo=g["campo"])
            filas.append({"paciente": g["patient_id"], "imagen": g["image_id"],
                          "TP": r["TP"], "FP": r["FP"], "FN": r["FN"],
                          "IoU_medio_TP": r["IoU_medio_TP"], "Dice_medio_TP": r["Dice_medio_TP"]})
        detalle[nombre] = pd.DataFrame(filas)

    total = pd.DataFrame({n: metricas(d) for n, d in detalle.items()}).T
    total["IoU_medio"] = [ponderado(d, "IoU_medio_TP") for d in detalle.values()]
    total["Dice_medio"] = [ponderado(d, "Dice_medio_TP") for d in detalle.values()]

    RESULTS.mkdir(parents=True, exist_ok=True)
    total.round(4).to_csv(RESULTS / f"b6_pred_{metodo}_test_total.csv")
    pd.concat(detalle, names=["metodo"]).to_csv(RESULTS / f"b6_pred_{metodo}_test_por_imagen.csv")
    return total
