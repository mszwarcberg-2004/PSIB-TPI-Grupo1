"""B3: métricas de segmentación y matching predicción-GT."""
import numpy as np
from scipy.optimize import linear_sum_assignment


def dice(a, b):
    #Dice entre dos máscaras
    a, b = a > 0, b > 0
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else 2 * (a & b).sum() / s


def iou(a, b):
    #IoU entre dos máscaras
    a, b = a > 0, b > 0
    u = (a | b).sum()
    return 1.0 if u == 0 else (a & b).sum() / u


def matriz_iou(gt, pred):
    """gt, pred: mapas de etiquetas (0 = fondo, 1..N = instancias).
    Devuelve (iou[nG, nP], ids_gt, ids_pred)."""
    ids_g, g_idx = np.unique(gt, return_inverse=True)
    ids_p, p_idx = np.unique(pred, return_inverse=True)
   
    #Fuerzo que el fondo (0) sea siempre el primer id
    if ids_g[0] != 0:
        ids_g = np.r_[0, ids_g]; g_idx = g_idx + 1
    if ids_p[0] != 0:
        ids_p = np.r_[0, ids_p]; p_idx = p_idx + 1

    nG, nP = len(ids_g), len(ids_p)
    
    # Cuenta de una vez cuántos píxeles comparte cada par (GT, pred): la intersección
    inter = np.bincount(g_idx.ravel() * nP + p_idx.ravel(), minlength=nG * nP).reshape(nG, nP).astype(np.float64)
    area_g, area_p = inter.sum(1), inter.sum(0) # áreas de cada célula
    union = area_g[:, None] + area_p[None, :] - inter # unión = A + B - intersección
    with np.errstate(divide="ignore", invalid="ignore"):
        M = np.where(union > 0, inter / union, 0.0)
    return M[1:, 1:], ids_g[1:], ids_p[1:] # sin fondo


def evaluar_imagen(gt, pred, umbral_iou=0.5):
    #Matching uno a uno y métricas de detección + segmentación de una imagen
    M, ids_g, ids_p = matriz_iou(gt, pred)
    nG, nP = len(ids_g), len(ids_p)

    pares = []
    if nG and nP:
        # Asignación uno a uno que maximiza el IoU total (algoritmo húngaro)
        filas, cols = linear_sum_assignment(-M)    
        for i, j in zip(filas, cols):
            if M[i, j] >= umbral_iou: # solo cuenta si IoU >= umbral
                a, b = gt == ids_g[i], pred == ids_p[j]
                pares.append({"gt_id": int(ids_g[i]), "pred_id": int(ids_p[j]),
                              "iou": M[i, j], "dice": dice(a, b)})

    tp = len(pares)  # predicciones emparejadas con una GT
    fp, fn = nP - tp, nG - tp # predicciones sin pareja, GT sin pareja
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    return {
        "TP": tp, "FP": fp, "FN": fn,
        "precision": prec, "sensibilidad": rec, "F1": f1,
        "IoU_medio_TP": np.mean([p["iou"] for p in pares]) if pares else np.nan,
        "Dice_medio_TP": np.mean([p["dice"] for p in pares]) if pares else np.nan,
        "Dice_pixel": dice(gt > 0, pred > 0),        # máscara global, sin instancias
        "pares": pares,
    }