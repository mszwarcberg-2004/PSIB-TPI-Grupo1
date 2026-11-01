# Función del archivo: segmentación semántica de eritrocitos (eritrocito vs fondo).

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu

KERNEL_APERTURA = 5   # tamaño del kernel de la apertura (elimina ruido y uniones finas)
KERNEL_CIERRE = 7     # tamaño del kernel del cierre (une pedacitos de una misma célula)
HUECO_MAX = 1500      # área máxima (en píxeles) de un hueco que se rellena (centro claro del eritrocito) --> tiene un límite porque áreas más grandes pertenecen al fondo.

# MÉTODO: OTSU

#Otsu es el método de referencia. 
#Aplicamos otsu sobre el canal verde porque en imágenes de frotis sanguíneos teñidos con Giemsa, el canal verde puede presentar un contraste útil entre los eritrocitos rosados y el fondo claro.
def segmentar_otsu_verde(imagen_rgb, campo):

    # Extraer el canal verde de la imagen ya preprocesada
    verde = imagen_rgb[:, :, 1]

    # Calcular el umbral de Otsu dentro del campo circular
    umbral = threshold_otsu(verde[campo])

    # Generar la máscara binaria
    mascara = (verde < umbral) & campo

    return mascara, umbral


def _kernel(tamano): #Esta función crea un elemento estructurante elíptico del tamaño indicado --> para luego aplicar operaciones morfológicas sobre la segmentación hecha con Otsu
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tamano, tamano))

#Importante: en las imágenes teñidas, los eritrocitos muchas veces presentan una periferia más oscura y un centro más claro. 
# El problema es que el centro claro sigue perteneciendo al eritrocito y al segmentarlos, ese centro podría quedar como un hueco negro.
def rellenar_huecos_chicos(mascara, area_max): #completa el interior de los eritrocitos.

    #Idea: 
    # 1. busca huecos negros completamente encerrados por regiones blancas
    # 2. calcula cuántos píxeles tiene cada hueco.
    # 3. Rellena solamente los que tienen un área menor o igual a HUECO_MAX = 1500.
    # Devuelve la máscara con esos huecos rellenados.

    # identifica regiones negras completamente rodeadas de blanco y las convierte en blanco. Por ahora, las rellena todas, independientemente del tamaño.
    mascara_rellena = ndi.binary_fill_holes(mascara)
    
    # Identificar los píxeles que se rellenaron: los que están blancos en máscara_rellena y negros en máscara. 
    huecos = mascara_rellena & ~mascara

    # Etiquetar los huecos y calcular sus áreas
    etiquetas, n = ndi.label(huecos)
    areas = ndi.sum(huecos, etiquetas, index=np.arange(1, n + 1))

    # Seleccionar los huecos que no superen el área máxima
    chicos = np.isin(etiquetas, np.where(areas <= area_max)[0] + 1)

    # Incorporar solamente los huecos pequeños a la máscara original
    return mascara | chicos


def limpiar_mascara(mascara): #Objetivo: mejorar la máscara binaria obtenida mediante Otsu, eliminando ruido mediante cierre y apertura
    m = mascara.astype(np.uint8) #Convierte una máscara booleana (True y False) a una matriz de enteros (1 y 0).
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, _kernel(KERNEL_APERTURA)) #Aplicamos apertura: elimina pequeñas regiones blancas aisladas que Otsu detectó erróneamente como eritrocitos. También puede romper conexiones muy finas entre células.
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, _kernel(KERNEL_CIERRE)) #Aplicamos cierre: une pequeños fragmentos blancos cercanos y cierra pequeñas discontinuidades dentro de las regiones segmentadas.
    return rellenar_huecos_chicos(m.astype(bool), HUECO_MAX) 

