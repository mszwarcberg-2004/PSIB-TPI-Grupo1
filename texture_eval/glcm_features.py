#características de textura GLCM por eritrocito
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops

sys.path.append(str(Path(__file__).resolve().parents[1]))
from common.config import ROOT

# ---- Parámetros -----
LEVELS = 32 # Reduce las intensidades de 256 a 32 niveles de gris paara simplificar la glcm y reducir costo computacional. 
DISTANCES = [1, 2, 3] #Permite analizar la textura a distintas escalas.
ANGLES = [0, np.pi/4, np.pi/2, 3*np.pi/4] #Permite considerar distintas orientaciones de la textura.
CHANNEL = "green" # Utiliza el canal verde de RGB, que se eligió por su contraste en las imágenes teñidas.
PROPS = ["contrast", "homogeneity", "energy", "correlation"] #Define las características que se calculan a partir de la GLCM. También está entropía que se calcula aparte. 


def _a_gris(crop_rgb): #recibe un corte rgb de un eritrocito, y se queda sólo con el canal verde --> la glcm puede trabajar con intensidades de 0 a 255. 
    if CHANNEL == "green":
        return crop_rgb[:, :, 1]
    return cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)


def _cuantizar(g): #reduce la cantidad de niveles de gris de 255 a 32 para simplificar el cálculo de la glcm y reducir costo computacional. 
    #Reduce 256 niveles a LEVELS, con el mismo mapeo para todas las células.
    return (g.astype(np.uint16) * LEVELS // 256).astype(np.uint8)


def _entropia(glcm): #calcula la entropía
    #Entropía de la GLCM normalizada. Entrada (L, L, D, A), salida (D, A).
    p = glcm
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.where(p > 0, np.log2(p), 0.0)
    return -(p * logp).sum(axis=(0, 1))


def glcm_celula(crop_rgb, mask): #Funcion principal: recibe una célula recortada y su máscara y calcula caracteristicasa que describan su textura. 
    #Devuelve dict de descriptores promediados sobre los ángulos, por distancia

    #se queda con los niveles de intensidad verde y cuantiza
    q = _cuantizar(_a_gris(crop_rgb))

    #excluye el fondo
    mask = mask > 0

    # El fondo recibe un nivel reservado (LEVELS) que luego se descarta,
    # así solo cuentan pares de píxeles que están ambos dentro de la célula
    q = np.where(mask, q, LEVELS).astype(np.uint8)
    #asigna el valor 32 a los píxeles que están fuera de la célula, porque las intensidades van de 0 a 31

    # Calculo GLCM simétrica para todas las distancias y ángulos
    glcm = graycomatrix(q, DISTANCES, ANGLES, levels=LEVELS + 1, symmetric=True, normed=False)
    glcm = glcm[:LEVELS, :LEVELS].astype(np.float64)  # saco fila/columna del fondo
    # se construyen 12 matrices GLCM: con las distancias [1,2,3] y en las 4 orientaciones

    # Normalización: cada matriz (distancia, ángulo) pasa a sumar 1
    totales = glcm.sum(axis=(0, 1), keepdims=True)
    n_pares = int(totales.min())
    if n_pares == 0: #None si no hay pares válidos
        return None
    glcm /= totales

    out = {"n_pixeles": int(mask.sum()), "n_pares_min": n_pares}
    #Resultado: cada matriz representa una distribución de probabilidades de pares de intensidades.
    
    # Propiedades de skimage y promediando los 4 ángulos para cada distancia
    for prop in PROPS:
        vals = graycoprops(glcm, prop)          # (D, A)
        for i, d in enumerate(DISTANCES):
            out[f"{prop}_d{d}"] = vals[i].mean()
        #Para cada propiedad, obtiene un resultado por distancia y ángulo. Después promedia los cuatro ángulos de cada distancia.
   
    #Entropía y mismo promedio por ángulo
    ent = _entropia(glcm)
    for i, d in enumerate(DISTANCES):
        out[f"entropy_d{d}"] = ent[i].mean()
    return out