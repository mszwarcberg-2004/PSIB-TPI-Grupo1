
# Función del archivo: evaluación genérica de distintos métodos de segmentación para luego poder ver cual es el mejor. 
# Compara una máscara predicha con el Ground Truth usando IoU y Dice.

import numpy as np


def evaluar_segmentacion(mascara_pred, mascara_gt, campo=None): #mascara_pred: es la máscara obtenida mediante el algoritmo de segmentación que queremos evaluar, por ejemplo, Otsu

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
