# Función del archivo: detección de regiones candidatas a parásito (manchas oscuras chicas) sobre la imagen completa.
# Sale de la sección 7 del notebook C3 (Bloque C). No usa el GT ni necesita conocer las células.
# La usa el Bloque A para la segmentación dirigida por color (A6): A busca el eritrocito que contiene cada región.

import cv2
import numpy as np

from segmentation.preprocess import mascara_circular
from color_stain.normalizacion import normalizar_imagen
from color_stain.wbc_detection import detectar_wbc

# Parámetros elegidos en C3 con los 6 pacientes de train (validación anidada por paciente: exactitud balanceada 0,933)
RADIO_TOPHAT = 20       # radio del disco del top-hat: más grande que un parásito, más chico que un eritrocito (~130 px)
UMBRAL_TOPHAT = 61      # un píxel es candidato si es al menos 61 niveles más oscuro que su entorno (deja el 0,02 % de los píxeles sanos)
AREA_MIN_CANDIDATA = 25 # nos quedamos con las manchas de al menos 25 píxeles


#Idea: el top-hat negro (cierre - imagen) "rellena" las zonas oscuras más chicas que el disco y las resta de la imagen.
#Así quedan resaltadas solo las manchas oscuras chicas (como un parásito), con un valor igual a cuánto más oscuras son
#que lo que las rodea. Un eritrocito entero es más grande que el disco, así que no se resalta.


def tophat_negro(verde, radio=RADIO_TOPHAT):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radio + 1, 2 * radio + 1))
    return cv2.morphologyEx(verde, cv2.MORPH_BLACKHAT, kernel)


def detectar_candidatas(imagen_rgb, campo=None): #recibe la imagen RGB original y devuelve una máscara booleana: True en las regiones candidatas

    if campo is None:
        campo = mascara_circular(imagen_rgb)

    # 1. Normalizar el color con el fondo de referencia de train (balance de blancos, ver normalizacion.py)
    imagen_norm = normalizar_imagen(imagen_rgb, campo)

    # 2. Top-hat negro sobre G: cuánto más oscuro es cada píxel que su entorno
    th = tophat_negro(imagen_norm[:, :, 1])

    # 3. Píxeles mucho más oscuros que su entorno, dentro del campo y fuera de los núcleos de glóbulos blancos
    #    (el borde del núcleo de un WBC también da top-hat alto, por eso lo sacamos)
    candidatos = (th > UMBRAL_TOPHAT) & campo & ~detectar_wbc(imagen_rgb, campo)

    # 4. Nos quedamos solo con las manchas de tamaño suficiente (borra puntitos sueltos de ruido)
    n, etiquetas, stats, _ = cv2.connectedComponentsWithStats(candidatos.astype(np.uint8), connectivity=8)
    grandes = [k for k in range(1, n) if stats[k, cv2.CC_STAT_AREA] >= AREA_MIN_CANDIDATA]

    return np.isin(etiquetas, grandes)
#Ojo: las regiones son candidatas, no parásitos verificados. Algunas caen fuera de los eritrocitos
#(objetos oscuros sueltos entre las células): A6 las descarta si no encuentra un eritrocito que las contenga.
