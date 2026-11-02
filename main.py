# Función del archivo: integra el pipeline completo (los tres bloques), cada bloque va agregando sus pasos.
# Ejecutar desde la carpeta principal del repo: python main.py

import numpy as np
import pandas as pd

from common.config import SEED, FEATURES_DIR, rutas_segmentacion
from common.io_utils import cargar_subset, cargar_imagen, cargar_celula

# Bloque A: segmentación de eritrocitos (se importa con otro nombre para no pisar la función de este archivo)
from segmentation.a1_segmentacion_base import procesar_imagen as segmentar_imagen
from segmentation.a1_segmentacion_base import limpiar_resultados_anteriores

# Bloque B: extracción de características de textura
from texture_eval.glcm_features import glcm_celula


# Cuando cada bloque tenga sus funciones, se importan acá, por ejemplo:
# from color_stain.wbc_detection import detectar_wbc                 # Bloque C
# from segmentation.watershed_split import separar_celulas           # Bloque A
# from color_stain.color_features import calcular_color              # Bloque C
# from texture_eval.auto_classifier import identificar_infectados    # Bloque B
# from color_stain.infected_seg import construir_mascara_infectados  # Bloque C


# Método de segmentación que usa todo el pipeline: "otsu", "kmeans" o "gmm".
# Cuando termine la comparación (comparar_metodos.py), se cambia por el método ganador.
METODO = "otsu"

# Defino una función a la que entra una imagen y, mediante distintas funciones, se lleva a cabo todo el procesamiento.
def procesar_imagen(imagen_rgb, patient_id, image_id, split, rutas):

    # 0. BLOQUE C: máscara de exclusión de glóbulos blancos (todavía no existe)
    mascara_wbc = None  # cuando esté: mascara_wbc = detectar_wbc(imagen_rgb), con True donde hay un glóbulo blanco

    # 1. BLOQUE A: segmentación de eritrocitos con el método elegido.
    # Aplica preprocesamiento, segmentación, limpieza morfológica, etiquetado, filtrado
    # y guarda recortes y máscaras de cada célula. Devuelve: células, máscara, etiquetas y campo.
    filas, mascara, _, _ = segmentar_imagen(
        imagen_rgb, patient_id, image_id, split, rutas, METODO, mascara_wbc
    )

    # 2. BLOQUE B: características de textura de cada célula detectada
    caracteristicas_textura = []

    for fila in filas:
        # Leer el recorte y la máscara que guardó el Bloque A
        crop, mask = cargar_celula(rutas, fila["cell_id"])

        # Calcular los atributos GLCM de esa célula
        atributos = glcm_celula(crop, mask)

        # glcm_celula devuelve None si la célula no tiene pares de píxeles válidos: se saltea
        if atributos is None:
            continue

        # Se guarda con cell_id, para poder combinar después con las características de color
        caracteristicas_textura.append({"cell_id": fila["cell_id"], **atributos})

    # 3. PASOS PENDIENTES
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

    print(f"Imágenes a procesar: {len(imagenes)} | método: {METODO}")

    rutas = rutas_segmentacion(METODO)
    limpiar_resultados_anteriores(rutas)  # borra recortes, máscaras y CSV de corridas anteriores

    celulas_total = []   # todas las células detectadas
    textura_total = []   # características de textura de todas las células

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

        celulas_total.extend(resultado["celulas"])
        textura_total.extend(resultado["textura"])

        print(
            f"{imagen['image_id']}: "
            f"{len(resultado['celulas'])} células detectadas"
        )

    # Guardar la lista de células (cells_pred.csv) y las características de textura
    rutas["csv"].parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(celulas_total).to_csv(rutas["csv"], index=False)

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(textura_total).to_csv(FEATURES_DIR / f"features_texture_{METODO}.csv", index=False)

    print(f"Total: {len(celulas_total)} células -> {rutas['csv']}")


# Cuando corremos el archivo main.py, se ejecuta sobre train.
if __name__ == "__main__":
    main("train")