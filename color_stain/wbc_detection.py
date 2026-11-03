# Función del archivo: detección de los núcleos de los glóbulos blancos (WBC) por color.
# Sale de la máscara de WBC del notebook C3 (Bloque C), con un criterio de color más estricto (ver abajo).
# La usa el Bloque A para sacar los glóbulos blancos de la segmentación de eritrocitos.

import cv2
import numpy as np

from segmentation.preprocess import mascara_circular
from color_stain.normalizacion import normalizar_imagen

UMBRAL_B_WBC = 127     # violeta si b (Lab) < 127. Sale de C1: umbral de Youden para separar WBC de eritrocitos (solo train)
UMBRAL_G_WBC = 100     # además, oscuro si G < 100 (ver explicación abajo)
AREA_MIN_WBC = 6000    # una región violeta y oscura se considera núcleo si tiene al menos 6000 píxeles
RADIO_APERTURA = 3     # apertura para borrar puntitos sueltos antes de medir las regiones


#Por qué agregamos la condición de G (oscuro):
#En C3 la máscara era solo "violeta" (b < 127 y más de 12000 píxeles). Probándola en nuestra segmentación de train,
#en las imágenes del paciente 209 (el de tinción más oscura) también marcaba eritrocitos, y se perdían 3 células PARASITADAS.
#Eso es grave, porque bajaría la parasitemia. Según C1, el núcleo del glóbulo blanco no solo es violeta: también es muy oscuro
#(mediana de G = 52), mientras que el 95% de los píxeles de células parasitadas tiene G mayor a 124. Por eso pedimos G < 100.
#Probado sobre train: con G < 80 o G < 100 y área mínima entre 3000 y 12000 el resultado es el mismo
#(0 glóbulos blancos aceptados y 0 eritrocitos perdidos), así que el valor elegido no depende de un ajuste fino.


def detectar_wbc(imagen_rgb, campo=None): #recibe la imagen RGB original y devuelve una máscara booleana: True en los núcleos de glóbulos blancos

    if campo is None:
        campo = mascara_circular(imagen_rgb)

    # 1. Normalizar el color con el fondo de referencia de train (balance de blancos, ver normalizacion.py)
    imagen_norm = normalizar_imagen(imagen_rgb, campo)

    # 2. Canales que usamos: b de Lab (violeta = valores bajos) y G de RGB (oscuro = valores bajos)
    b = cv2.cvtColor(imagen_norm, cv2.COLOR_RGB2LAB)[:, :, 2]
    verde = imagen_norm[:, :, 1]

    # 3. Píxeles violetas Y oscuros dentro del campo circular
    candidatos = (b < UMBRAL_B_WBC) & (verde < UMBRAL_G_WBC) & campo

    # 4. Apertura: borra puntitos sueltos y uniones finas, así no se juntan regiones que no tienen nada que ver
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * RADIO_APERTURA + 1, 2 * RADIO_APERTURA + 1))
    candidatos = cv2.morphologyEx(candidatos.astype(np.uint8), cv2.MORPH_OPEN, kernel)

    # 5. Nos quedamos solo con las regiones grandes: un parásito es chico, un núcleo de glóbulo blanco es grande
    n, etiquetas, stats, _ = cv2.connectedComponentsWithStats(candidatos, connectivity=8)
    grandes = [k for k in range(1, n) if stats[k, cv2.CC_STAT_AREA] >= AREA_MIN_WBC]

    return np.isin(etiquetas, grandes)
#Ojo: la máscara marca el NÚCLEO del glóbulo blanco, no el citoplasma (que es mucho más claro).
#Por eso en a1_segmentacion_base.py no borramos solo estos píxeles: eliminamos el objeto segmentado entero que contiene el núcleo.
