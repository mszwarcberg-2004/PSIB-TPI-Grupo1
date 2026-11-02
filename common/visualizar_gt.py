
# Función del archivo: trabajar con los polígonos del GT.
# 1. Construir máscaras binarias para evaluar la segmentación.
# 2. Visualizar los contornos del GT sobre las imágenes originales.

import cv2
import numpy as np
from common.io_utils import cargar_subset, cargar_imagen, cargar_gt
from common.config import CHECK_DIR, LABEL_PARASITIZED, LABEL_UNINFECTED, LABEL_WBC

FACTOR = 0.25 #compresión: tamaño al que voy a reducir las imágenes para que no pesen demasiado y las pueda guardar. 

COLORES = {
    LABEL_PARASITIZED: (255, 0, 0),
    LABEL_UNINFECTED: (0, 255, 0),
    LABEL_WBC: (0, 0, 255),
}
#dibujo el contorno de eritrocitos parasitados, no infectados y glóbulos blancos con un color distinto: rojo, azul y verde respectivametne


# 1. CONSTRUIR MÁSCARA BINARIA DEL GT --> para después compararla con la máscara que obtengamos mediante distintos métodos de segmentación.
def construir_mascara_gt(imagen_rgb, anotaciones):
    # Crear una máscara negra del tamaño de la imagen original
    mascara_gt = np.zeros(imagen_rgb.shape[:2], dtype=np.uint8)

    for a in anotaciones:

        # Incluir únicamente eritrocitos parasitados y no parasitados
        if a["label"] not in [LABEL_PARASITIZED, LABEL_UNINFECTED]:
            continue

        # Saltear anotaciones sin polígono válido
        if a["polygon"] is None or len(a["polygon"]) < 3:
            continue

        # Obtener las coordenadas del polígono
        puntos = np.round(a["polygon"]).astype(np.int32)

        # Rellenar el interior del eritrocito con valor 1
        cv2.fillPoly(mascara_gt, [puntos], 1)

    return mascara_gt.astype(bool) #devuelve una matriz que tiene: 
    #True (1): píxel perteneciente a un eritrocito según el GT.
    # #False (0): píxel que no pertenece a un eritrocito anotado.


# 2. VISUALIZAR LOS POLÍGONOS DEL GT
def visualizar_gt():

    CHECK_DIR.mkdir(parents=True, exist_ok=True)

    for imagen in cargar_subset():

        patient_id = imagen["patient_id"]
        image_id = imagen["image_id"]

        img = cargar_imagen(patient_id, image_id)
        anotaciones, _ = cargar_gt(patient_id, image_id)

        # Dibujar los contornos de las células
        for a in anotaciones:

            if a["polygon"] is None:
                continue

            puntos = a["polygon"].astype(np.int32).reshape(-1, 1, 2)

            cv2.polylines(
                img,
                [puntos],
                isClosed=True,
                color=COLORES.get(a["label"], (255, 255, 0)),
                thickness=3
            )

        # Reducir tamaño de la imagen
        img = cv2.resize(
            img,
            None,
            fx=FACTOR,
            fy=FACTOR,
            interpolation=cv2.INTER_AREA
        )

        # Guardar imagen
        salida = CHECK_DIR / f"{patient_id}__{image_id}.png"

        cv2.imwrite(str(salida), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        print("Guardado:", salida)


if __name__ == "__main__":
    visualizar_gt()
