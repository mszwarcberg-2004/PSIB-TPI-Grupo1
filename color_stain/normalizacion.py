# Función del archivo: normalización del color de las imágenes por el color del fondo (balance de blancos).
# Las funciones mediana_fondo y normalizar_fondo vienen de los notebooks C2 y C3 (Bloque C), sin cambios en la lógica.
# Ejecutar desde la raíz del repo:  python -m color_stain.normalizacion   (calcula y guarda la referencia de train)

import json

import cv2
import numpy as np

from common.config import ROOT
from common.io_utils import cargar_subset, cargar_imagen
from segmentation.preprocess import mascara_circular

# Archivo donde se guarda el color de fondo de referencia, así no hay que recalcularlo cada vez
RUTA_REFERENCIA = ROOT / "color_stain" / "results" / "referencia_fondo.json"


#Idea: el fondo (plasma) casi no toma colorante, así que su color depende sobre todo de la luz del microscopio y de la cámara.
#Si llevamos el fondo de todas las imágenes a un mismo color, corregimos esas diferencias sin tocar lo que es propio de las células.
#En C2 se vio que para segmentar no hace falta, pero sí ayuda con el canal b (lo violeta), que es el que usamos para los glóbulos blancos.


#La función estima el color de fondo de cada imagen 
def mediana_fondo(imagen_rgb, campo):
    # Mediana R, G, B del fondo. El fondo se estima sin GT: píxeles del campo con G mayor al umbral de Otsu (igual que en C2)
    verde = imagen_rgb[:, :, 1]
    umbral, _ = cv2.threshold(verde[campo].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return np.median(imagen_rgb[campo & (verde > umbral)].astype(float), axis=0)

# balance de blancos para corregir las diferencias entre las distintas imágenes y hacer más comparables las características cromáticas de los eritrocitos.
def normalizar_fondo(imagen_rgb, fondo_imagen, fondo_ref):
    # Multiplica cada canal R, G, B por (fondo de referencia / fondo de esta imagen), para que el fondo quede igual en todas
    return np.clip(imagen_rgb * (fondo_ref / fondo_imagen), 0, 255).astype(np.uint8)


#Función calcular_referencia: el fondo de referencia es el promedio del fondo de las imágenes de TRAIN.
def calcular_referencia():

    fondos = []
    for reg in cargar_subset("train"):
        img = cargar_imagen(reg["patient_id"], reg["image_id"])
        fondos.append(mediana_fondo(img, mascara_circular(img)))

    referencia = np.mean(fondos, axis=0)

    # La guardamos en un archivo .json (un archivo de texto simple) para no volver a cargar las 12 imágenes cada vez
    RUTA_REFERENCIA.parent.mkdir(parents=True, exist_ok=True)
    with open(RUTA_REFERENCIA, "w") as archivo:
        json.dump({"fondo_rgb": referencia.tolist(), "calculada_con": "train"}, archivo, indent=2)

    return referencia

#Carga el fondo RGB de referencia previamente guardado o, si no existe, lo calcula utilizando las imágenes de entrenamiento.
def cargar_referencia():
    # Si la referencia ya está guardada, la leemos; si no, la calculamos con train
    if RUTA_REFERENCIA.exists():
        with open(RUTA_REFERENCIA) as archivo:
            return np.array(json.load(archivo)["fondo_rgb"])
    return calcular_referencia()


#Automatiza el proceso completo
def normalizar_imagen(imagen_rgb, campo=None):
    # Atajo para normalizar una imagen entera (de train o de test) con la referencia de train
    if campo is None:
        campo = mascara_circular(imagen_rgb)
    return normalizar_fondo(imagen_rgb, mediana_fondo(imagen_rgb, campo), cargar_referencia())


#Cuando ejecutas directamente el archivo, este bloque calcula y muestra el fondo RGB de referencia de las imágenes del train
if __name__ == "__main__":
    print("Fondo de referencia (R, G, B), calculado con train:", calcular_referencia().round(1))

