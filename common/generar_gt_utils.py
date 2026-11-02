
# Función del archivo: funciones compartidas para trabajar con el Ground Truth (GT).
# Permiten obtener recortes y máscaras individuales de las célulasn y construir mapas de etiquetas para evaluar la segmentación.

from pathlib import Path
import cv2
import numpy as np
from common.io_utils import cargar_subset, cargar_imagen, cargar_gt


def celulas_desde_gt(split=None):
    # Generador que devuelve (metadatos, recorte RGB, máscara) para cada célula anotada en el Ground Truth.

    for reg in cargar_subset(split):
        pid = reg["patient_id"]
        iid = reg["image_id"]
        sp = reg["split"]

        # Cargar la imagen original y sus anotaciones
        img = cargar_imagen(pid, iid)
        anotaciones, _ = cargar_gt(pid, iid)

        alto, ancho = img.shape[:2]

        for a in anotaciones:

            # Obtener los vértices del polígono en píxeles enteros
            poly = np.round(a["polygon"]).astype(np.int32)

            if len(poly) < 3:
                continue

            # Calcular el bounding box y limitarlo a la imagen
            x0 = max(0, int(poly[:, 0].min()))
            y0 = max(0, int(poly[:, 1].min()))
            x1 = min(ancho, int(poly[:, 0].max()) + 1)
            y1 = min(alto, int(poly[:, 1].max()) + 1)

            # Descartar recortes fuera de la imagen o vacíos
            if x1 <= x0 or y1 <= y0:
                continue

            # Recortar la imagen RGB sin modificar la original
            crop = img[y0:y1, x0:x1].copy()

            # Crear una máscara del tamaño del recorte
            mask = np.zeros(crop.shape[:2], dtype=np.uint8)

            # Trasladar el polígono a las coordenadas del recorte
            poly_local = poly - np.array([x0, y0], dtype=np.int32)

            # Rellenar de blanco la región de la célula
            cv2.fillPoly(mask, [poly_local], 255)

            if not np.any(mask):
                continue

            # Guardar los metadatos de la célula
            meta = {
                "cell_id": f"{pid}_{Path(iid).stem}_{a['cell_id']}",
                "image_id": iid,
                "patient_id": pid,
                "split": sp,
                "label": a["label"],
            }

            yield meta, crop, mask


def mapa_gt(img_shape, anotaciones, clases=None):
    # Construye un mapa de etiquetas a partir de los polígonos GT.
    # Fondo = 0; cada célula tiene un identificador diferente.
    # Si se especifican clases, solo se incluyen esas clases.

    mapa = np.zeros(img_shape[:2], dtype=np.int32)

    for k, a in enumerate(anotaciones, start=1):

        # Filtrar las clases si corresponde
        if clases is not None and a["label"] not in clases:
            continue

        # Convertir los vértices a coordenadas enteras
        poly = np.round(a["polygon"]).astype(np.int32)

        if len(poly) < 3:
            continue

        # Rellenar el polígono con el identificador de la célula
        cv2.fillPoly(mapa, [poly], k)

    return mapa
