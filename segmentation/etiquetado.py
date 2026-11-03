
# Función del archivo: etiquetado (Merge Labeling), filtrado de objetos y extracción de células.

import cv2
import numpy as np
from skimage.measure import label, regionprops
from common.io_utils import construir_cell_id

FACTOR_AREA_MIN = 0.6   # se descarta si mide menos de 0.6 células típicas (pedacitos, ruido)
FACTOR_AREA_MAX = 2.0   # se descarta si mide más de 2 células típicas (células pegadas que no se separaron)
#El filtro de área ahora es RELATIVO: el tamaño típico de una célula se calcula en cada imagen (ver area_tipica), sin usar el GT.
#Así no depende de la escala: si las células se ven más grandes o más chicas, los límites cambian con ellas.
#Los factores se eligieron sobre train: probamos mínimo 0.3 a 0.8 y máximo 1.4 a 2.2, con validación dejando un paciente afuera,
#y 0.6 - 2.0 fue el mejor en las 6 vueltas (F1 de detección 0.946). Los vecinos (0.6 - 1.8 o 2.2) dan casi lo mismo (0.946).
#Historia de los valores fijos anteriores (en píxeles): antes eran 800 y 12000, pero en el GT de train un eritrocito mide en promedio ~14000 píxeles (el 81% supera 12000), así que se descartaban la mayoría.
#Los nuevos valores salen de mirar las áreas del GT de train (percentil 1 ≈ 8200 y percentil 99 ≈ 19500), dejando algo de margen porque
#nuestras máscaras salen un poco más chicas que los polígonos del GT. Probándolo con Otsu sobre train: F1 de detección 0.36 --> 0.64, precisión 0.95.
#Lo que sigue quedando afuera por grande son sobre todo células pegadas, que se van a separar con Watershed.
DESCARTAR_BORDE = True  # descarta células cortadas por el borde del círculo
MARGEN_BORDE = 3        # ancho (en píxeles) de la franja que se considera borde


#Función area_tipica: estima cuánto mide un eritrocito en ESTA imagen, sin usar el GT.
#No sirve la mediana común de las áreas: en algunas imágenes (paciente 209) hay cientos de pedacitos chicos de ruido
#y la mediana da ~2000 píxeles en vez de ~12000. Por eso usamos la mediana PONDERADA por área: ordenamos los objetos
#de menor a mayor y buscamos el objeto en el que se completa la mitad de todos los píxeles de célula.
#Los pedacitos chicos tienen pocos píxeles, así que casi no pesan. En train da entre 10000 y 14600 píxeles en todas las imágenes.
def area_tipica(areas):
    areas = np.sort(areas)               # de menor a mayor
    acumulada = np.cumsum(areas)         # píxeles acumulados
    mitad = acumulada[-1] / 2            # la mitad de todos los píxeles de célula
    return areas[np.searchsorted(acumulada, mitad)]


def etiquetar(mascara): #Recibe la máscara binaria que recibimos luego de segmentar la imagen y hace merge labeling.
    return label(mascara, connectivity=2)


def filtrar_y_extraer(imagen_rgb, etiquetas, campo, patient_id, image_id, split, rutas):

    # Función: revisa cada objeto etiquetado y guardar solamente los que cumplen los criterios definidos.
    # Filtra objetos por área y borde, y guarda recorte y máscara de cada célula.
    # Devuelve (filas para cells_pred.csv, imagen de etiquetas solo con las células aceptadas).

    #Preparamos las carpetas de salida
    #Las rutas dependen del método de segmentación (otsu, kmeans, gmm, etc.)
    crops_dir = rutas["crops"]
    masks_dir = rutas["masks"]

    crops_dir.mkdir(parents=True, exist_ok=True) #Se guardarán los recortes RGB de las células.
    masks_dir.mkdir(parents=True, exist_ok=True) #Se guardan máscaras binarias de esas mismas células.

    #Idea: detectar los eritrocitos que están demasiado cerca del borde circular del microscopio, para poder descartarlos si están cortados.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * MARGEN_BORDE + 1, 2 * MARGEN_BORDE + 1))
    interior = cv2.erode(campo.astype(np.uint8), kernel).astype(bool)

    filas = [] #filas: lista donde guardaremos la información de cada célula aceptada
    etiquetas_ok = np.zeros(etiquetas.shape, dtype=np.int32) #nueva imagen de etiquetas, inicialmente toda en cero, donde solamente aparecerán las células aceptadas.
    indice = 0 #inicializo el contador que cuenta las células aceptadas

    # Tamaño típico de una célula en esta imagen (con los objetos que no tocan el borde) y límites del filtro de área
    areas = [r.area for r in regionprops(etiquetas) if interior[r.coords[:, 0], r.coords[:, 1]].all()]
    if not areas:
        return filas, etiquetas_ok  # no hay objetos para analizar
    tipica = area_tipica(areas)
    area_min = FACTOR_AREA_MIN * tipica
    area_max = FACTOR_AREA_MAX * tipica

    for r in regionprops(etiquetas): #regionprops sirve para medir las propiedades de cada objeto etiquetado

        #En primer lugar, aplicamos filtros para ver si las células son aceptadas o no
        # Se descartan objetos demasiados chicos o grandes
        if r.area < area_min or r.area > area_max:
            continue

        #Se descartan células cortadas por el borde
        if DESCARTAR_BORDE and not interior[r.coords[:, 0], r.coords[:, 1]].all():
            continue

        #Si las células atravesaron los filtros, quiere decir que son aceptadas y sumo +1 al contador.
        indice += 1
        cell_id = construir_cell_id(patient_id, image_id, indice)

        #Obtengo el bounding box
        min_f, min_c, max_f, max_c = r.bbox

        #Luego, en máscara_celula, se queda sólo con la región de la célula comprendida dentro del bounding box
        mascara_celula = etiquetas[min_f:max_f, min_c:max_c] == r.label
        #Resultado: una máscara binaria individual, donde esa célula está en blanco y todo lo demás en negro.

        #Enmascaramos para extraer el recorte a color, en rgb
        recorte = imagen_rgb[min_f:max_f, min_c:max_c] * mascara_celula[..., None]  # fondo en negro

        # Se guardan en PNG dos archivos por cada célula:
            # - Recorte RGB: la célula en color, con fondo negro.
            # - Máscara individual: la silueta de la célula en blanco sobre fondo negro.

        cv2.imwrite(
            str(crops_dir / f"{cell_id}.png"),
            cv2.cvtColor(recorte, cv2.COLOR_RGB2BGR)
        )

        cv2.imwrite(
            str(masks_dir / f"{cell_id}.png"),
            mascara_celula.astype(np.uint8) * 255
        )

        #Guarda en etiquetas_ok la etiqueta de la célula aceptada.
        etiquetas_ok[min_f:max_f, min_c:max_c][mascara_celula] = indice
        #Al finalizar, esta imagen contendrá:
            # - 0: fondo y objetos descartados.
            # - 1, 2, 3...: células aceptadas.

        # Circularidad: 1 = círculo perfecto. Cuanto más irregular o alargado sea el objeto, generalmente menor será su circularidad. Sólamente lo guardamos como una característica.
        circularidad = min(1.0, 4 * np.pi * r.area / r.perimeter ** 2) if r.perimeter > 0 else 0.0

        #Agrega un diccionario a la lista filas con los datos de la célula aceptada.
        filas.append({
            "cell_id": cell_id,
            "image_id": image_id,
            "patient_id": patient_id,
            "split": split,
            "bbox_x": min_c,
            "bbox_y": min_f,
            "bbox_w": max_c - min_c,
            "bbox_h": max_f - min_f,
            "area": int(r.area),
            "circularidad": round(float(circularidad), 4),
        })

    return filas, etiquetas_ok


#Devuelve:
#filas: Lista de diccionarios con los datos de todas las células aceptadas
#etiquetas_ok: Imagen etiquetada que contiene solamente las células aceptadas
