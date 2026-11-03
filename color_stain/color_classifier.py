# Función del archivo: identificación de eritrocitos parasitados por color, parasitemia y máscara final de infectados (Bloque C).
# El criterio sale del notebook C5 (results/c5_criterio_color.json): una célula es parasitada si tiene al menos una mancha oscura
# de la máscara de C3 (n_manchas >= 1). Las características se calculan antes con color_features.caracteristicas_celula.
#
# Uso en el pipeline (para cada imagen):
#     atributos = caracteristicas_celula(crop, mask, fondo)        # de color_features.py, para cada célula
#     es_parasitada = clasificar_color(atributos)                  # True / False
#     porcentaje = parasitemia(lista_de_decisiones)                # % de eritrocitos infectados de la imagen
#     mascara = mascara_infectados(etiquetas, ids_parasitadas)     # máscara con los eritrocitos parasitados completos

import json

import numpy as np

from common.config import ROOT
from color_stain.color_features import PARAMS_C3

# Archivo con el criterio elegido en C5 (ajustado solo con train)
RUTA_CRITERIO = ROOT / "color_stain" / "results" / "c5_criterio_color.json"


def cargar_criterio():
    with open(RUTA_CRITERIO) as archivo:
        criterio = json.load(archivo)
    # Las características tienen que calcularse con los mismos parámetros de la máscara de C3 que se usaron para elegir el criterio
    if any(criterio["params_mascara_c3"][k] != v for k, v in PARAMS_C3.items()):
        raise ValueError("Los parámetros de C3 de color_features.py no coinciden con los del criterio de C5")
    return criterio


#Función clasificar_color: recibe el diccionario de características de una célula (caracteristicas_celula) y devuelve True si es parasitada
def clasificar_color(atributos, criterio=None):
    if atributos is None:  # máscara vacía: no se puede clasificar
        return False
    if criterio is None:
        criterio = cargar_criterio()
    if criterio["regla"] != "Tiene mancha de C3":
        raise ValueError(f"Regla no implementada: {criterio['regla']}")
    return atributos["n_manchas"] >= 1


#Función parasitemia: porcentaje de eritrocitos clasificados como parasitados (sobre todas las células analizadas)
def parasitemia(es_parasitada):
    es_parasitada = np.asarray(es_parasitada, dtype=bool)
    return 100 * es_parasitada.mean() if len(es_parasitada) else 0.0


#Función mascara_infectados: máscara binaria con los eritrocitos parasitados completos.
#etiquetas: imagen de etiquetas de las células segmentadas (0 = fondo, 1, 2, 3... = células).
#ids_parasitadas: etiquetas de las células clasificadas como parasitadas.
def mascara_infectados(etiquetas, ids_parasitadas):
    return np.isin(etiquetas, list(ids_parasitadas)) & (etiquetas > 0)
