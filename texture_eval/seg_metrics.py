
# Función del archivo: evaluación de distintos métodos de segmentación.
# Permite comparar una máscara predicha con el Ground Truth usando IoU y Dice.
# Además, permite evaluar la detección individual de eritrocitos mediante matching.

import numpy as np
from scipy.optimize import linear_sum_assignment


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



def matriz_iou(gt, pred):
    """gt, pred: mapas de etiquetas (0 = fondo, 1..N = instancias).
    Devuelve (iou[nG, nP], ids_gt, ids_pred)."""

    ids_g, g_idx = np.unique(gt, return_inverse=True)
    ids_p, p_idx = np.unique(pred, return_inverse=True)

    # Fuerzo que el fondo (0) sea siempre el primer id
    if ids_g[0] != 0:
        ids_g = np.r_[0, ids_g]
        g_idx = g_idx + 1

    if ids_p[0] != 0:
        ids_p = np.r_[0, ids_p]
        p_idx = p_idx + 1

    nG, nP = len(ids_g), len(ids_p)

    # Cuenta cuántos píxeles comparte cada par (GT, pred): la intersección
    inter = np.bincount(
        g_idx.ravel() * nP + p_idx.ravel(),
        minlength=nG * nP
    ).reshape(nG, nP).astype(np.float64)

    # Áreas de cada célula
    area_g, area_p = inter.sum(1), inter.sum(0)

    # Unión = A + B - intersección
    union = area_g[:, None] + area_p[None, :] - inter

    # Calcular IoU entre todas las combinaciones de células
    with np.errstate(divide="ignore", invalid="ignore"):
        M = np.where(union > 0, inter / union, 0.0)

    # Devolver la matriz de IoU sin incluir el fondo
    return M[1:, 1:], ids_g[1:], ids_p[1:]




def evaluar_imagen(gt, pred, umbral_iou=0.5):
    # Evalúa qué tan bien se detectó cada célula individualmente. Utiliza el IoU para asociar cada célula predicha con una célula del GT.

    # Verificar que ambas imágenes tengan las mismas dimensiones
    if gt.shape != pred.shape:
        raise ValueError("Los mapas de etiquetas deben tener el mismo tamaño")

    # Calcular matriz de IoU entre todas las células
    M, ids_g, ids_p = matriz_iou(gt, pred)

    nG, nP = len(ids_g), len(ids_p)

    pares = []

    if nG and nP:

        # Algoritmo húngaro: busca una asignación uno a uno
        # entre células GT y predichas que maximice el IoU total
        filas, cols = linear_sum_assignment(-M)

        for i, j in zip(filas, cols):

            # Considerar una detección correcta si el IoU supera el umbral
            if M[i, j] >= umbral_iou:

                # Construir las máscaras binarias de las dos células
                a = gt == ids_g[i]
                b = pred == ids_p[j]

                # Calcular Dice entre ambas células utilizando la misma
                # función de evaluación global que ya teníamos
                metricas = evaluar_segmentacion(b, a)

                # Guardar el par de células correctamente asociado
                pares.append({
                    "gt_id": int(ids_g[i]),
                    "pred_id": int(ids_p[j]),
                    "iou": float(M[i, j]),
                    "dice": metricas["Dice"]
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
