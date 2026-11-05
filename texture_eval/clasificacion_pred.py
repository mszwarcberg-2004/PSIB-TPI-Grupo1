"""B5/B6 sobre la segmentación propia (cells_pred): aplica el método elegido en el notebook B (sección 8)
y lo evalúa contra el GT.

Reglas que se respetan:
- La elección del método y sus parámetros salen del notebook, solo con train, y quedan en b_metodo_elegido.json.
- main.py no compara ni reajusta nada: solo aplica ese archivo.
- El GT (clase_gt) solo se usa para medir; nunca para decidir en test.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from texture_eval.seg_metrics import evaluar_imagen, metricas, ponderado

RESULTS = Path(__file__).resolve().parent / "results"
RUTA_DECISION = RESULTS / "b_metodo_elegido.json"

NOMBRE_C5 = "Color C5 (n_manchas>=1)"
NOMBRE_AND = "C5 AND textura"
NOMBRE_OR = "C5 OR textura"


def cargar_decision(ruta=RUTA_DECISION):
    """Método elegido en el notebook (sección 8.4). Falla si es un método que este módulo no sabe aplicar."""
    if not ruta.exists():
        raise FileNotFoundError(f"Falta {ruta}: se genera con la celda 8.4 del notebook B.")
    with open(ruta, encoding="utf-8") as f:
        d = json.load(f)
    nombre = d["metodo"]
    if nombre not in (NOMBRE_C5, NOMBRE_AND, NOMBRE_OR) and not nombre.startswith("Textura ("):
        raise NotImplementedError(
            f"El método '{nombre}' necesita un modelo ajustado y el JSON no guarda coeficientes. "
            "Solo se soportan reglas: C5 sola, textura sola, C5 AND textura y C5 OR textura.")
    return d


def regla_textura(decision, atributos_tex):
    """Regla de textura con signo y umbral fijados en train. Sin textura calculada -> False."""
    if atributos_tex is None:
        return False
    r = decision["regla_textura"]
    return bool(r["signo"] * float(atributos_tex[r["atributo"]]) >= r["umbral"])


def decidir(decision, pred_c5, atributos_tex):
    """Decisión final (True = parasitada) de una célula, según el método elegido."""
    nombre = decision["metodo"]
    if nombre == NOMBRE_C5:
        return bool(pred_c5)
    tex = regla_textura(decision, atributos_tex)
    if nombre == NOMBRE_AND:
        return bool(pred_c5) and tex
    if nombre == NOMBRE_OR:
        return bool(pred_c5) or tex
    return tex            # "Textura (atributo)"


def evaluar_test(guardados, marcadas, metodo, carpeta=RESULTS):
    #carpeta (agregado): dónde se guardan los dos CSV. Por defecto, texture_eval/results como antes; main.py le pasa pred/resultados_test.
    """Evaluación end-to-end contra el GT de parasitadas (matching IoU >= 0,5), sin reajustar nada.
    guardados: una entrada por imagen con 'patient_id', 'image_id', 'celulas' (filas en el orden de las
    etiquetas 1..N), 'etiquetas' (mapa de la predicción), 'campo' y 'gt' (mapa GT de parasitadas).
    marcadas: {nombre del método: set de cell_id marcados como parasitadas}."""
    detalle = {}
    for nombre, ids_marcados in marcadas.items():
        filas = []
        for g in guardados:
            ids = [k for k, f in enumerate(g["celulas"], start=1) if f["cell_id"] in ids_marcados]
            pred = np.where(np.isin(g["etiquetas"], ids), g["etiquetas"], 0)
            r = evaluar_imagen(g["gt"], pred, campo=g["campo"])
            filas.append({"paciente": g["patient_id"], "imagen": g["image_id"],
                          "TP": r["TP"], "FP": r["FP"], "FN": r["FN"],
                          "IoU_medio_TP": r["IoU_medio_TP"], "Dice_medio_TP": r["Dice_medio_TP"]})
        detalle[nombre] = pd.DataFrame(filas)

    total = pd.DataFrame({n: metricas(d) for n, d in detalle.items()}).T
    total["IoU_medio"] = [ponderado(d, "IoU_medio_TP") for d in detalle.values()]
    total["Dice_medio"] = [ponderado(d, "Dice_medio_TP") for d in detalle.values()]

    carpeta.mkdir(parents=True, exist_ok=True)
    total.round(4).to_csv(carpeta / f"b6_pred_{metodo}_test_total.csv")
    pd.concat(detalle, names=["metodo"]).to_csv(carpeta / f"b6_pred_{metodo}_test_por_imagen.csv")
    return total