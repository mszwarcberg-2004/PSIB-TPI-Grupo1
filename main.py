
# Función del archivo: integra el pipeline completo (los tres bloques), cada bloque va agregando sus pasos.
# Ejecutar desde la carpeta principal del repo: python main.py

import numpy as np

from common.config import SEED, rutas_segmentacion
from common.io_utils import cargar_subset, cargar_imagen

# Bloque A: segmentación de eritrocitos
from segmentation.a1_segmentacion_base import procesar_imagen_otsu

# Bloque B: extracción de características de textura
from texture_eval.glcm_features import glcm_celula


# Cuando cada bloque tenga sus funciones, se importan acá, por ejemplo:
# from color_stain.wbc_detection import detectar_wbc                 # Bloque C
# from segmentation.cluster_seg import segmentar_eritrocitos         # Bloque A
# from segmentation.watershed_split import separar_celulas           # Bloque A
# from color_stain.color_features import calcular_color              # Bloque C
# from texture_eval.auto_classifier import identificar_infectados    # Bloque B
# from color_stain.infected_seg import construir_mascara_infectados  # Bloque C


# Defino una función a la que entra una imagen y, mediante distintas funciones,
# se lleva a cabo todo el procesamiento.
def procesar_imagen(imagen_rgb, patient_id, image_id, split, rutas):

    # 1. BLOQUE A: segmentación de eritrocitos mediante Otsu
    # Esta función aplica preprocesamiento, Otsu, limpieza morfológica,
    # etiquetado, filtrado y extracción de las células.
    filas, mascara = procesar_imagen_otsu(
        imagen_rgb, patient_id, image_id, split, rutas
    )

    # 2. BLOQUE B: características de textura de cada célula detectada
    # Los recortes y las máscaras individuales fueron generados por el Bloque A.
    caracteristicas_textura = []

    for fila in filas:

        # Todavía falta conectar la lectura de los recortes y máscaras
        # generados por el Bloque A con glcm_celula().
        #
        # crop = ...
        # mask = ...
        # atributos = glcm_celula(crop, mask)
        #
        # caracteristicas_textura.append({
        #     **fila,
        #     **atributos
        # })

        pass

    # 3. PASOS PENDIENTES
    # Detección de glóbulos blancos
    # Segmentación mediante K-means / GMM
    # Separación de células adheridas mediante Watershed
    # Extracción de características cromáticas
    # Clasificación de eritrocitos infectados
    # Construcción de la máscara final de células infectadas

    return {
        "mascara": mascara,
        "celulas": filas,
        "textura": caracteristicas_textura
    }


# La función main va a encargarse de recorrer las imágenes del dataset
# y llamar a las funciones de procesamiento que desarrollen los tres bloques.
def main(split="train"):

    np.random.seed(SEED)  # Fijar la semilla para reproducibilidad

    imagenes = cargar_subset(split)

    print(f"Imágenes a procesar: {len(imagenes)}")

    rutas = rutas_segmentacion("otsu")

    for imagen in imagenes:

        # Cargar la imagen RGB original
        imagen_rgb = cargar_imagen(
            imagen["patient_id"],
            imagen["image_id"]
        )

        # Procesar la imagen con el pipeline
        resultado = procesar_imagen(
            imagen_rgb,
            imagen["patient_id"],
            imagen["image_id"],
            imagen["split"],
            rutas
        )

        print(
            f"{imagen['image_id']}: "
            f"{len(resultado['celulas'])} células detectadas"
        )


# Cuando corremos el archivo main.py, se ejecuta sobre train.
if __name__ == "__main__":
    main("train")
