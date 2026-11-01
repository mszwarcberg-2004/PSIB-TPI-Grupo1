# Función del archivo: dibuja los polígonos del GT sobre cada imagen para verificarlos a ojo.
# Guarda las imágenes (reducidas) en data/shared/gt_check/.
# Ejecutar desde la carpeta principal del repo:  python -m common.visualizar_gt

import cv2
import numpy as np
from common.io_utils import cargar_subset, cargar_imagen, cargar_gt
from common.config import CHECK_DIR, LABEL_PARASITIZED, LABEL_UNINFECTED, LABEL_WBC

FACTOR = 0.25  # compresión: se guarda al 25% del tamaño para que los archivos no pesen tanto

#Se crea un diccionario llamado COLORES, que relaciona cada tipo de célula con un color.
# Colores en RGB: parasitadas en rojo, no parasitadas en verde, glóbulos blancos en azul --> se dibuja el contorno de cada color.
COLORES = {
    LABEL_PARASITIZED: (255, 0, 0),
    LABEL_UNINFECTED: (0, 255, 0),
    LABEL_WBC: (0, 0, 255),
}

CHECK_DIR.mkdir(parents=True, exist_ok=True) #creo la carpeta de salida dentro de data/shared/

for imagen in cargar_subset():
    #Obtener el paciente y el nombre de la imagen
    patient_id, image_id = imagen["patient_id"], imagen["image_id"]

    img = cargar_imagen(patient_id, image_id)
    anotaciones, _ = cargar_gt(patient_id, image_id)

    #En cada iteración, a es un diccionario correspondiente a una célula.
    for a in anotaciones:
        if a["polygon"] is None:  # fila sin polígono (puede pasar con los WBC), las saltea
            continue
        puntos = a["polygon"].astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [puntos], isClosed=True, color=COLORES.get(a["label"], (255, 255, 0)), thickness=3)

    img = cv2.resize(img, None, fx=FACTOR, fy=FACTOR, interpolation=cv2.INTER_AREA)
    salida = CHECK_DIR / f"{patient_id}__{image_id}.png"
    cv2.imwrite(str(salida), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))  # OpenCV guarda en BGR
    print("Guardado:", salida)
