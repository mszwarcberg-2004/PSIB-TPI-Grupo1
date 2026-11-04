"""B3: métricas de segmentación y matching predicción-GT."""

import numpy as np
from scipy.optimize import linear_sum_assignment
import cv2
import pandas as pd

# 1. EVALUACIÓN GLOBAL DE LA SEGMENTACIÓN

def evaluar_segmentacion(mascara_pred, mascara_gt, campo=None): #mascara_pred: es la máscara obtenida mediante el algoritmo de segmentación que queremos evaluar, por ejemplo, Otsu
#Evalúa qué tan bien segmentaste los píxeles de todos los eritrocitos en conjunto. No distingue si detectaste correctamente cada célula individual.
    # Convertir ambas máscaras a booleanas
    pred = mascara_pred.astype(bool)
    gt = mascara_gt.astype(bool)

    # Verificar que las dimensiones coincidan para poder compararlas
    if pred.shape != gt.shape:
        raise ValueError("Las máscaras deben tener el mismo tamaño")

    # Evaluar únicamente dentro del campo circular
    if campo is not None:
        if campo.shape != pred.shape:
            raise ValueError("El campo debe tener el mismo tamaño")

        pred = pred & campo
        gt = gt & campo

    # Calcular la cantidad de píxeles coincidentes en la intersección
    interseccion = np.logical_and(pred, gt).sum()

    # Calcular píxeles que pertenecen al menos a una máscara (que están en blanco o 1)
    union = np.logical_or(pred, gt).sum()

    # Calcular IoU
    iou = interseccion / union if union > 0 else 1.0

    # Calcular Dice
    suma_areas = pred.sum() + gt.sum()
    dice = 2 * interseccion / suma_areas if suma_areas > 0 else 1.0

    return {
        "IoU": float(iou),
        "Dice": float(dice)
    } #devolvemos los resultados



# 2. COMPARACIÓN ENTRE CÉLULAS INDIVIDUALES (MATCHING)
#Porque para detectar células individualmente necesitamos saber qué célula predicha corresponde a qué célula real.



def matriz_iou(gt, pred, devolver_conteos=False):
    """gt, pred: mapas de etiquetas (0 = fondo, 1..N = instancias).
    Devuelve (iou[nG, nP], ids_gt, ids_pred).
    Con devolver_conteos=True devuelve además (inter[nG, nP], area_gt, area_pred) en píxeles."""

    # Solo se consideran los píxeles que pertenecen a alguna célula (GT o predicción).
    # El fondo es casi toda la imagen y no aporta a ninguna intersección, así que no hace falta ordenarlo
    celulas = (gt > 0) | (pred > 0)
    g = gt[celulas].astype(np.int64)
    p = pred[celulas].astype(np.int64)

    # Los ids van de 0 a N (el 0 es el fondo): se usan directamente como índices
    nG, nP = int(gt.max()) + 1, int(pred.max()) + 1

    # Cuenta cuántos píxeles comparte cada par (GT, pred): la intersección
    inter = np.bincount(
        g * nP + p,
        minlength=nG * nP
    ).reshape(nG, nP).astype(np.float64)

    # Áreas de cada célula
    area_g, area_p = inter.sum(1), inter.sum(0)

    # Ids que realmente aparecen en cada mapa (sin el fondo)
    ids_g = np.flatnonzero(area_g[1:] > 0) + 1
    ids_p = np.flatnonzero(area_p[1:] > 0) + 1

    # Quedarse solo con las filas y columnas de esas células
    inter = inter[np.ix_(ids_g, ids_p)]
    area_g, area_p = area_g[ids_g], area_p[ids_p]

    # Unión = A + B - intersección
    union = area_g[:, None] + area_p[None, :] - inter

    # Calcular IoU entre todas las combinaciones de células
    with np.errstate(divide="ignore", invalid="ignore"):
        M = np.where(union > 0, inter / union, 0.0)

    if devolver_conteos:
        return M, ids_g, ids_p, inter, area_g, area_p
    return M, ids_g, ids_p



def evaluar_imagen(gt, pred, umbral_iou=0.5, campo=None):
    # Evalúa qué tan bien se detectó cada célula individualmente. Utiliza el IoU para asociar cada célula predicha con una célula del GT.

    # Verificar que ambas imágenes tengan las mismas dimensiones
    if gt.shape != pred.shape:
        raise ValueError("Los mapas de etiquetas deben tener el mismo tamaño")

    # Evaluar solo dentro del campo circular, si se indica
    if campo is not None:
        gt = np.where(campo, gt, 0)
        pred = np.where(campo, pred, 0)

    # Calcular matriz de IoU entre todas las células.
    # Se piden también las intersecciones y áreas, que se usan abajo para el Dice de cada par
    M, ids_g, ids_p, inter, area_g, area_p = matriz_iou(gt, pred, devolver_conteos=True)

    nG, nP = len(ids_g), len(ids_p)

    pares = []

    if nG and nP:

        # Algoritmo húngaro: busca una asignación uno a uno
        # entre células GT y predichas que maximice el IoU total
        filas, cols = linear_sum_assignment(-M)

        for i, j in zip(filas, cols):

            # Considerar una detección correcta si el IoU supera el umbral
            if M[i, j] >= umbral_iou:

                # Dice del par, calculado con la intersección y las áreas que ya tenemos.
                # Así no se recorre toda la imagen por cada célula
                dice_par = 2 * inter[i, j] / (area_g[i] + area_p[j])

                # Guardar el par de células correctamente asociado
                pares.append({
                    "gt_id": int(ids_g[i]),
                    "pred_id": int(ids_p[j]),
                    "iou": float(M[i, j]),
                    "dice": float(dice_par)
                })

    # Verdaderos positivos: células correctamente emparejadas
    tp = len(pares)

    # Falsos positivos: células predichas sin una pareja válida en el GT
    fp = nP - tp

    # Falsos negativos: células reales que no fueron detectadas correctamente
    fn = nG - tp

    # Calcular precisión
    precision = tp / (tp + fp) if (tp + fp) else 0.0

    # Calcular sensibilidad (recall)
    sensibilidad = tp / (tp + fn) if (tp + fn) else 0.0

    # Calcular F1
    f1 = (
        2 * precision * sensibilidad / (precision + sensibilidad)
        if (precision + sensibilidad) else 0.0
    )

    # Calcular métricas globales de segmentación utilizando nuestra función
    metricas_globales = evaluar_segmentacion(pred > 0, gt > 0)

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": precision,
        "sensibilidad": sensibilidad,
        "F1": f1,
        "IoU_medio_TP": float(np.mean([p["iou"] for p in pares])) if pares else np.nan,
        "Dice_medio_TP": float(np.mean([p["dice"] for p in pares])) if pares else np.nan,
        "IoU_pixel": metricas_globales["IoU"],
        "Dice_pixel": metricas_globales["Dice"],
        "pares": pares
    }



# 3. MAPA DE ETIQUETAS DEL GROUND TRUTH
# Para comparar con evaluar_imagen necesitamos las células del GT en un mapa de etiquetas
# (igual formato que el de la predicción), no como polígonos sueltos.

def mapa_gt(img_shape, anotaciones, clases=None):
    # Genera un mapa de etiquetas: 0 = fondo, k = k-ésima célula (rellena su polígono con k).
    # Con 'clases' dibuja solo esas clases (ej: ["Parasitized"])

    # Imagen vacía (todo fondo) del mismo alto y ancho que la imagen original
    lab = np.zeros(img_shape[:2], np.int32)

    # Recorrer las anotaciones; k empieza en 1 porque el 0 es el fondo
    for k, a in enumerate(anotaciones, start=1):

        # Dibujar la célula solo si no se filtró por clases o si su clase está en la lista.
        # Se numera con k aunque no se dibuje, así el id de cada célula no depende del filtro
        if clases is None or a["label"] in clases:

            # Redondear los vértices a píxeles enteros y rellenar el polígono con el id k
            cv2.fillPoly(lab, [np.round(a["polygon"]).astype(np.int32)], k)

    return lab



# 4. AGREGACIÓN DE RESULTADOS
# evaluar_imagen da TP/FP/FN de UNA imagen. Para resultados por paciente, por método o
# totales se suman los conteos y se recalculan las métricas (micro-promedio).
# No se promedian los F1 de cada imagen: con pocas parasitadas por imagen serían muy ruidosos.

def metricas(tabla):
    # Recibe una tabla con columnas TP, FP y FN (una fila por imagen) y devuelve las métricas del conjunto

    # Sumar los conteos de todas las filas
    tp, fp, fn = tabla.TP.sum(), tabla.FP.sum(), tabla.FN.sum()

    # Precisión y sensibilidad con las sumas (0 si el denominador es 0, para evitar dividir por cero)
    p = tp / (tp + fp) if tp + fp else 0
    s = tp / (tp + fn) if tp + fn else 0

    # F1 con la precisión y la sensibilidad globales
    return pd.Series({"TP": tp, "FP": fp, "FN": fn, "precision": p, "sensibilidad": s,
                      "F1": 2 * p * s / (p + s) if p + s else 0})


def resumen(tabla, por):
    # Lo mismo que metricas, pero agrupando por una columna (estrategia, paciente, feature...)

    # Sumar TP, FP y FN dentro de cada grupo
    s = tabla.groupby(por)[["TP", "FP", "FN"]].sum()

    # Métricas de cada grupo; fillna(0) cubre los grupos donde el denominador es 0 (queda NaN)
    s["precision"] = (s.TP / (s.TP + s.FP)).fillna(0)
    s["sensibilidad"] = (s.TP / (s.TP + s.FN)).fillna(0)

    # F1 escrito en función de los conteos: 2·TP / (2·TP + FP + FN)
    s["F1"] = (2 * s.TP / (2 * s.TP + s.FP + s.FN)).fillna(0)
    return s


def ponderado(d, col):
    # Promedio de una métrica por imagen (IoU_medio_TP, Dice_medio_TP) ponderado por la cantidad de TP

    # Descartar las imágenes sin TP: no tienen IoU/Dice (son NaN) y no deben contar como 0
    d = d[d["TP"] > 0]

    # Cada imagen pesa según cuántas células detectadas correctamente aportó
    return np.average(d[col], weights=d["TP"]) if len(d) else np.nan