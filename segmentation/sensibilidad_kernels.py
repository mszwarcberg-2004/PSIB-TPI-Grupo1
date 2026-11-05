# Función del archivo: tabla de sensibilidad de los kernels de la segmentación (apertura, cierre y filtro gaussiano), solo con train.
# Ejecutar desde la carpeta principal del repo: python -m segmentation.sensibilidad_kernels
#
#Idea: el profesor pidió que cada parámetro tenga un criterio y que se muestre cuánto cambia el resultado alrededor del valor elegido.
#Acá movemos UN parámetro por vez (los otros quedan en el valor elegido) y medimos la segmentación y la detección de células en train.
#No modifica nada del pipeline: usa las mismas funciones que a1_segmentacion_base.py y main.py, solo cambia el tamaño de los kernels.
#Los recortes que genera el filtrado se guardan en una carpeta temporal que se borra sola, así no se pisan los resultados de main.py.

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from common.io_utils import cargar_subset, cargar_imagen, cargar_gt
from common.generar_gt_utils import mapa_gt
from segmentation.preprocess import mascara_circular, mejorar_contraste_clahe
from segmentation.cluster_segmentation import (
    segmentar_otsu_verde, rellenar_huecos_chicos, _kernel, KERNEL_APERTURA, KERNEL_CIERRE, HUECO_MAX
)
from segmentation.watershed_split import separar_celulas
from segmentation.etiquetado import filtrar_y_extraer
from segmentation.a1_segmentacion_base import excluir_wbc, gt_sin_borde, CLASES_ERITROCITOS
from color_stain.wbc_detection import detectar_wbc
from texture_eval.seg_metrics import evaluar_segmentacion, evaluar_imagen

GAUSSIANO = 5  # tamaño del filtro gaussiano que usa el pipeline (está fijo en preprocesar_imagen, preprocess.py)

# Valores que probamos para cada parámetro (incluye el elegido). Gaussiano 0 = sin filtro gaussiano.
VALORES_APERTURA = [3, 5, 7, 9]
VALORES_CIERRE = [5, 7, 9, 11]
VALORES_GAUSSIANO = [0, 3, 5, 7]


#Armamos la lista de configuraciones: cada una cambia un solo parámetro respecto del valor elegido
def configuraciones():
    elegido = {"apertura": KERNEL_APERTURA, "cierre": KERNEL_CIERRE, "gaussiano": GAUSSIANO}
    lista = [("elegido", dict(elegido))]
    for nombre, valores in [("apertura", VALORES_APERTURA), ("cierre", VALORES_CIERRE), ("gaussiano", VALORES_GAUSSIANO)]:
        for v in valores:
            if v != elegido[nombre]:
                lista.append((nombre, {**elegido, nombre: v}))
    return lista


#Igual que preprocesar_imagen (CLAHE en el verde y después gaussiano), pero con el tamaño del gaussiano como dato
def preprocesar(imagen_rgb, campo, gaussiano):
    imagen = imagen_rgb.copy()
    imagen[:, :, 1] = mejorar_contraste_clahe(imagen[:, :, 1], campo)
    if gaussiano > 0:
        imagen = cv2.GaussianBlur(imagen, (gaussiano, gaussiano), 0)
    return imagen


#Igual que limpiar_mascara (apertura, cierre y relleno de huecos), pero con el tamaño de los kernels como dato
def limpiar(mascara, apertura, cierre):
    m = mascara.astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, _kernel(apertura))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, _kernel(cierre))
    return rellenar_huecos_chicos(m.astype(bool), HUECO_MAX)


def sensibilidad_kernels():

    lista = configuraciones()
    conteos = {i: {"TP": 0, "FP": 0, "FN": 0, "IoU": [], "Dice": []} for i in range(len(lista))}

    with tempfile.TemporaryDirectory() as carpeta_temporal:  #se borra sola al terminar
        rutas = {"crops": Path(carpeta_temporal) / "crops", "masks": Path(carpeta_temporal) / "masks"}
        rutas["crops"].mkdir()
        rutas["masks"].mkdir()

        for imagen in cargar_subset("train"):  #solo train
            patient_id, image_id = imagen["patient_id"], imagen["image_id"]
            imagen_rgb = cargar_imagen(patient_id, image_id)
            campo = mascara_circular(imagen_rgb)
            mascara_wbc = detectar_wbc(imagen_rgb)  #no depende de los kernels, se calcula una vez por imagen

            # GT de la imagen: máscara global de eritrocitos y mapa sin las células del borde (igual que en main.py)
            anotaciones, _ = cargar_gt(patient_id, image_id)
            mapa = mapa_gt(imagen_rgb.shape, anotaciones, clases=CLASES_ERITROCITOS)
            mapa_sin_borde = gt_sin_borde(mapa, campo)

            otsu_por_gaussiano = {}  #Otsu solo depende del gaussiano, así no se repite para cada apertura y cierre
            for i, (_, p) in enumerate(lista):
                if p["gaussiano"] not in otsu_por_gaussiano:
                    otsu_por_gaussiano[p["gaussiano"]], _ = segmentar_otsu_verde(preprocesar(imagen_rgb, campo, p["gaussiano"]), campo)

                # Mismos pasos que el pipeline: limpieza, Watershed, exclusión de glóbulos blancos y filtro de área y borde
                mascara = limpiar(otsu_por_gaussiano[p["gaussiano"]], p["apertura"], p["cierre"])
                etiquetas, _ = separar_celulas(mascara)
                etiquetas = excluir_wbc(etiquetas, mascara_wbc)
                _, etiquetas_ok = filtrar_y_extraer(imagen_rgb, etiquetas, campo, patient_id, image_id, "train", rutas)

                globales = evaluar_segmentacion(mascara, mapa > 0, campo)
                deteccion = evaluar_imagen(mapa_sin_borde, etiquetas_ok, campo=campo)
                c = conteos[i]
                c["IoU"].append(globales["IoU"])
                c["Dice"].append(globales["Dice"])
                for k in ("TP", "FP", "FN"):
                    c[k] += deteccion[k]

            print(f"Evaluada: {image_id}")

    # Tabla: IoU y Dice promediados por imagen; TP, FP y FN sumados (igual que el resumen de main.py)
    filas = []
    for i, (cambia, p) in enumerate(lista):
        c = conteos[i]
        precision = c["TP"] / (c["TP"] + c["FP"]) if (c["TP"] + c["FP"]) else 0.0
        sensibilidad = c["TP"] / (c["TP"] + c["FN"]) if (c["TP"] + c["FN"]) else 0.0
        f1 = 2 * precision * sensibilidad / (precision + sensibilidad) if (precision + sensibilidad) else 0.0
        filas.append({"cambia": cambia, **p, "IoU": np.mean(c["IoU"]), "Dice": np.mean(c["Dice"]),
                      "TP": c["TP"], "FP": c["FP"], "FN": c["FN"],
                      "precision": precision, "sensibilidad": sensibilidad, "F1": f1})

    tabla = pd.DataFrame(filas)
    print("\nSENSIBILIDAD DE LOS KERNELS (TRAIN):")
    print(tabla.round(4).to_string(index=False))

    salida = Path(__file__).resolve().parent / "results" / "sensibilidad_kernels.csv"
    tabla.round(4).to_csv(salida, index=False)
    print(f"\nTabla guardada en: {salida}")


if __name__ == "__main__":
    sensibilidad_kernels()
