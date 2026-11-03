# Función del archivo: separar eritrocitos que quedaron pegados en la máscara binaria, mediante Watershed.
# Se usa dentro de procesar_imagen() (a1_segmentacion_base.py), después de la limpieza morfológica y en lugar del etiquetado simple.

import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.measure import label
from skimage.segmentation import watershed

FACTOR_DISTANCIA = 0.7  # distancia mínima entre dos centros de células, como fracción del radio típico de un eritrocito
#Se eligió comparando 0.5, 0.7 y 0.9 sobre train (detección célula a célula): 0.7 fue el mejor para K-means y casi igual al mejor para Otsu.


#Función estimar_radio: calcula automáticamente el radio típico de un eritrocito en ESA imagen, sin usar el GT.
def estimar_radio(distancia, etiquetas):

    # Para cada objeto, el valor máximo de la distancia es aproximadamente su radio (la distancia del centro al borde).
    # Si son dos células pegadas, el máximo sigue siendo el radio de una de ellas, porque el centro de cada una está a un radio del fondo.
    radios = ndi.maximum(distancia, etiquetas, index=np.arange(1, etiquetas.max() + 1))

    # Usamos la mediana (y no el promedio) para que manchas chicas o grupos raros no la muevan demasiado
    return float(np.median(radios)) if len(radios) else 0.0


def separar_celulas(mascara): #recibe la máscara binaria ya limpia (True = eritrocito) y devuelve una imagen de etiquetas con las células separadas

    # 1. Etiquetado inicial (igual que etiquetar() en etiquetado.py): cada objeto conectado recibe un número
    etiquetas = label(mascara, connectivity=2)

    # Si la máscara está vacía, no hay nada que separar
    if etiquetas.max() == 0:
        return etiquetas, 0.0

    # 2. Transformada de distancia: distancia de cada píxel blanco al píxel negro más cercano (en píxeles)
    distancia = ndi.distance_transform_edt(mascara)

    # 3. Radio típico de un eritrocito en esta imagen, y distancia mínima entre dos centros
    radio = estimar_radio(distancia, etiquetas)
    distancia_minima = max(1, int(FACTOR_DISTANCIA * radio))
    #Así el parámetro se adapta solo a cada imagen: si las células se ven más grandes o más chicas, la distancia mínima cambia con ellas.

    # 4. Marcadores: máximos locales de la distancia (centros de las células)
    # labels=etiquetas: busca los máximos dentro de cada objeto, así cada objeto tiene al menos un marcador
    # min_distance: dos máximos más cerca que esto se consideran la misma célula (evita cortar una célula en pedazos)
    # exclude_border=False: también acepta máximos cerca del borde de la imagen
    centros = peak_local_max(distancia, min_distance=distancia_minima, labels=etiquetas, exclude_border=False)
    #centros es una lista de coordenadas (fila, columna), una por cada célula encontrada

    # Pasamos los centros a una imagen de marcadores: 0 en todos lados y 1, 2, 3... en cada centro
    marcadores = np.zeros(mascara.shape, dtype=np.int32)
    marcadores[centros[:, 0], centros[:, 1]] = np.arange(1, len(centros) + 1)

    # 5. Watershed sobre la distancia invertida (-distancia): los centros pasan a ser los "pozos" desde donde se inunda.
    # mask=mascara: solo se reparten los píxeles blancos; el fondo sigue siendo 0.
    etiquetas_separadas = watershed(-distancia, marcadores, mask=mascara)

    # Devolvemos la imagen de etiquetas (0 = fondo, 1, 2, 3... = células) y el radio estimado, por si se quiere revisar
    return etiquetas_separadas, radio
