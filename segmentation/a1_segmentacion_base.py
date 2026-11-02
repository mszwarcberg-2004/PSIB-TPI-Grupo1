# Función del archivo: A1, segmentación base de punta a punta para cualquier método (otsu, kmeans, gmm).
# Ejecutar desde la raíz del repo:  python -m segmentation.a1_segmentacion_base

import cv2
import numpy as np
import pandas as pd
from common.config import rutas_segmentacion
from common.io_utils import cargar_subset, cargar_imagen, cargar_gt
from common.generar_gt_utils import mapa_gt
from segmentation.preprocess import preprocesar_imagen
from segmentation.cluster_segmentation import segmentar, limpiar_mascara
from segmentation.etiquetado import etiquetar, filtrar_y_extraer, MARGEN_BORDE
from texture_eval.seg_metrics import evaluar_segmentacion, evaluar_imagen

CLASES_ERITROCITOS = ["Parasitized", "Uninfected"]  # el GT de glóbulos blancos no se incluye


def procesar_imagen(imagen_rgb, patient_id, image_id, split, rutas, metodo="otsu", mascara_wbc=None): #procesa una imagen
#mascara_wbc: máscara booleana del tamaño de la imagen que va a entregar el Bloque C, con True donde hay un glóbulo blanco.
#(Antes se llamaba exclusion_mask y no quedaba claro si True significaba "excluir" o "conservar").

    # 1. Preprocesamiento (igual para todos los métodos) y máscara circular
    imagen_preprocesada, campo = preprocesar_imagen(
        imagen_rgb,
        aplicar_clahe=True,
        aplicar_gaussiano=True
    )

    # 2. Segmentación semántica con el método elegido (otsu, kmeans o gmm)
    mascara_inicial, _ = segmentar(metodo, imagen_preprocesada, campo)

    # 3. Excluir glóbulos blancos (cuando C entregue la máscara; hasta entonces es None)
    if mascara_wbc is not None:
        mascara_inicial = mascara_inicial & ~mascara_wbc  # ~ invierte la máscara: nos quedamos con lo que NO es glóbulo blanco

    # 4. Apertura, cierre y relleno de huecos (igual para todos los métodos)
    mascara_limpia = limpiar_mascara(mascara_inicial)

    # 5. Etiquetar los objetos de la máscara limpia
    etiquetas = etiquetar(mascara_limpia)

    # 6. Filtrar y extraer las células individuales
    # Usamos la imagen RGB ORIGINAL para conservar sus colores reales
    filas, etiquetas_ok = filtrar_y_extraer(
        imagen_rgb, etiquetas, campo, patient_id, image_id, split, rutas
    )

    # Devolvemos las células, la máscara final, las etiquetas aceptadas y el campo circular
    return filas, mascara_limpia, etiquetas_ok, campo


def limpiar_resultados_anteriores(rutas): #que se borren las máscaras anteriores si estaban mal y lo estoy corriendo de nuevo
    # Eliminar recortes RGB anteriores
    rutas["crops"].mkdir(parents=True, exist_ok=True)
    for archivo in rutas["crops"].glob("*.png"): #busca los png de la carpeta
        archivo.unlink() #unlink: elimina cada archivo

    # Eliminar máscaras anteriores
    rutas["masks"].mkdir(parents=True, exist_ok=True)
    for archivo in rutas["masks"].glob("*.png"): #busca los png de la carpeta
        archivo.unlink() #unlink: elimina cada archivo

    # Eliminar CSV anterior
    if rutas["csv"].exists():
        rutas["csv"].unlink()


#Función gt_sin_borde: saca del mapa de etiquetas del GT las células que tocan el borde del círculo.
#Nuestro filtrado descarta las células cortadas por el borde, así que si las dejamos en el GT se cuentan como FN aunque las hayamos descartado a propósito.
#Usamos el mismo criterio que filtrar_y_extraer (etiquetado.py): la franja de MARGEN_BORDE píxeles.
def gt_sin_borde(mapa, campo):

    # Interior del campo: el círculo achicado MARGEN_BORDE píxeles (igual que en etiquetado.py)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * MARGEN_BORDE + 1, 2 * MARGEN_BORDE + 1))
    interior = cv2.erode(campo.astype(np.uint8), kernel).astype(bool)

    # Etiquetas del GT que tienen algún píxel fuera del interior
    ids_borde = np.unique(mapa[~interior])
    ids_borde = ids_borde[ids_borde > 0]  # el 0 es el fondo, no es una célula

    # Esas células pasan a ser fondo (0); el resto queda igual
    return np.where(np.isin(mapa, ids_borde), 0, mapa)


#Objetivo: ejecutar la segmentación A1 con un método sobre todas las imágenes seleccionadas y guardar la información de las células detectadas.
def segmentacion_base(split="train", metodo="otsu"): #procesa todas las imágenes

    # Evaluamos solamente train, reservando test para la evaluación final.
    if split != "train":
        raise ValueError("Por ahora solamente se permite ejecutar train.")

    # Elegir las rutas del método de segmentación
    rutas = rutas_segmentacion(metodo)

    # Borrar resultados de ejecuciones anteriores
    limpiar_resultados_anteriores(rutas)

    filas_total = [] #lista vacía en la que vamos a guardar la información de todas las células detectadas
    resultados_metricas = [] #guardamos las métricas para luego calcular el promedio
    tp_total, fp_total, fn_total = 0, 0, 0 #contadores de la evaluación individual (matching), sumando todas las imágenes

    for imagen in cargar_subset(split): #consulta el archivo subset.csv y devuelve las imágenes correspondientes al conjunto seleccionado.
        patient_id, image_id = imagen["patient_id"], imagen["image_id"]
        imagen_rgb = cargar_imagen(patient_id, image_id)

        #Acá procesamos la imagen
        filas, mascara_limpia, etiquetas_ok, campo = procesar_imagen(
            imagen_rgb, patient_id, image_id, imagen["split"], rutas, metodo
        )

        filas_total.extend(filas) #Agrega a filas_total las células detectadas y aceptadas en la imagen actual.
        print(f"{patient_id} / {image_id}: {len(filas)} células") #Muestra en la terminal cuantas células se aceptaron de esa imagen.

        # Evaluar la máscara contra el GT de train
        anotaciones, _ = cargar_gt(patient_id, image_id)
        mascara_gt = mapa_gt(imagen_rgb.shape, anotaciones, clases=CLASES_ERITROCITOS) > 0  # máscara global de eritrocitos
        resultados_metricas.append(evaluar_segmentacion(mascara_limpia, mascara_gt, campo))

        # Evaluación individual (matching): compara cada célula aceptada con las células del GT.
        # Usamos etiquetas_ok (solo las células que pasaron el filtro) y el GT sin las células del borde.
        etiquetas_gt = gt_sin_borde(mapa_gt(imagen_rgb.shape, anotaciones, clases=CLASES_ERITROCITOS), campo)
        r = evaluar_imagen(etiquetas_gt, etiquetas_ok, campo=campo)
        tp_total += r["TP"]
        fp_total += r["FP"]
        fn_total += r["FN"]

    #Guardamos todas las células ya procesadas en un csv
    rutas["csv"].parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(filas_total).to_csv(rutas["csv"], index=False)

    print(f"Total: {len(filas_total)} células -> {rutas['csv']}")

    if not resultados_metricas:
        raise ValueError("No se encontraron imágenes de train para evaluar.")

    promedio = pd.DataFrame(resultados_metricas)[["IoU", "Dice"]].mean()

    print(f"\nMÉTRICAS GENERALES ({metodo.upper()}, TRAIN):")
    print(f"IoU promedio: {promedio['IoU']:.4f}")
    print(f"Dice promedio: {promedio['Dice']:.4f}")

    # Métricas de detección sumando TP, FP y FN de todas las imágenes (no promediando por imagen, para que cada célula pese lo mismo)
    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) else 0.0
    sensibilidad = tp_total / (tp_total + fn_total) if (tp_total + fn_total) else 0.0
    f1 = 2 * precision * sensibilidad / (precision + sensibilidad) if (precision + sensibilidad) else 0.0

    print(f"\nDETECCIÓN DE CÉLULAS INDIVIDUALES ({metodo.upper()}, TRAIN, IoU >= 0.5):")
    print(f"TP: {tp_total} | FP: {fp_total} | FN: {fn_total}")
    print(f"Precisión: {precision:.4f} | Sensibilidad: {sensibilidad:.4f} | F1: {f1:.4f}")


#Si ejecuto directamente este archivo, se corre Otsu (cambiar "otsu" por "kmeans" o "gmm" para probar los otros).
if __name__ == "__main__":
    segmentacion_base("train", "otsu")