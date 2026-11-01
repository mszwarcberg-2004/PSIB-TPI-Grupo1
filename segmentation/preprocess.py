# Función del archivo: preprocesamiento básico. 

import cv2
import numpy as np
from scipy import ndimage as ndi

UMBRAL_NEGRO = 20  # por debajo de este gris, el píxel pertenece al borde negro
MARGEN = 5         # píxeles que se achica el círculo (evita artefactos JPEG en el borde)


#Función mascara_circular: identificar el campo circular del microscopio y excluir el borde negro de la imagen,
# para que después los algoritmos de segmentación no confundan ese fondo negro con estructuras celulares.

def mascara_circular(imagen_rgb): #imagen RGB, es decir, una matriz NumPy de dimensiones (alto, ancho, 3).

    gris = cv2.cvtColor(imagen_rgb, cv2.COLOR_RGB2GRAY) #convierte a escala de grises


    #Devuelve una máscara booleana: True dentro del campo de visión circular.
    campo = (gris > UMBRAL_NEGRO).astype(np.uint8)  #compara cada píxel con el umbral de 20 y si su intensidad es mayor que 20 asigna true y sino false
    # 1 = no es negro

    # Se queda con la región conexa más grande (el círculo)--> agrupa los píxeles con valor 1 que están conectados entre sí.
    n, etiquetas, stats, _ = cv2.connectedComponentsWithStats(campo, connectivity=8)

    #Comprueba si se detectó más de una región distinta al fondo
    if n > 1:
        mayor = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]) #sleecciona sólamente el componente más grande
        campo = (etiquetas == mayor).astype(np.uint8)
            #Lo convierte en una máscara binaria de 0 y 1


    # Rellena huecos (por ejemplo, células muy oscuras dentro del círculo)
    campo = ndi.binary_fill_holes(campo).astype(np.uint8)

    # Erosionamos para achicar el circulo unos píxeles cerca del borde circular, donde pueden aparecer artefactos de la imagen.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * MARGEN + 1, 2 * MARGEN + 1))
    return cv2.erode(campo, kernel).astype(bool)


# CORRECCIÓN DE ILUMINACIÓN: para que la imagen tenga iluminación uniforme.
# Si una zona de la imagen está más oscura que otra por la iluminación del microscopio, esto puede afectar el umbral global de Otsu.
def corregir_iluminacion(verde, campo):

    # Convertimos la imagen y la máscara a float
    imagen = verde.astype(np.float32)
    mascara = campo.astype(np.float32)

    # Estimamos la iluminación local mediante un suavizado amplio.
    # Normalizamos por la máscara para no incluir el borde negro.
    numerador = cv2.GaussianBlur(imagen * mascara, (0, 0), sigmaX=60)
    denominador = cv2.GaussianBlur(mascara, (0, 0), sigmaX=60)

    fondo = numerador / np.maximum(denominador, 1e-6)

    # Compensamos las diferencias de brillo respecto al fondo estimado para que la imagen tenga iluminación más uniforme
    referencia = np.median(fondo[campo])
    corregida = imagen - fondo + referencia

    # Mantener intensidades válidas entre 0 y 255
    return np.clip(corregida, 0, 255).astype(np.uint8) 
#La función devuelve una imagen en escala de grises (canal verde) con las variaciones de iluminación compensadas.


# MEJORA DE CONTRASTE LOCAL MEDIANTE CLAHE, considerando distintas zonas de la imagen.
def mejorar_contraste_clahe(verde, campo):

    # Evitar que el borde negro afecte el contraste
    imagen = verde.copy()
    imagen[~campo] = int(np.median(verde[campo]))

    # Crear y aplicar CLAHE
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    return clahe.apply(imagen)
#La función devuelve una imagen en escala de grises (canal verde) con el contraste local modificado.