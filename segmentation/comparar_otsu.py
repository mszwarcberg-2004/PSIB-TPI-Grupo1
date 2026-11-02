
# Comparación de distintas configuraciones de Otsu contra el GT.
# Se evalúan filtros, operaciones morfológicas y preprocesamiento.
# Ejecutar: python -m segmentation.comparar_otsu

import cv2
import pandas as pd
from skimage.filters import threshold_otsu

from common.config import ROOT
from common.io_utils import cargar_subset, cargar_imagen, cargar_gt
from common.visualizar_gt import construir_mascara_gt

from segmentation.preprocess import (
    mascara_circular,
    corregir_iluminacion,
    mejorar_contraste_clahe
)

from segmentation.cluster_segmentation import (
    segmentar_otsu_verde,
    limpiar_mascara,
    KERNEL_APERTURA,
    KERNEL_CIERRE,
    HUECO_MAX,
    rellenar_huecos_chicos
)

from texture_eval.seg_metrics import evaluar_segmentacion

def comparar_otsu():

    resultados = []

    # Evaluar únicamente las imágenes de entrenamiento
    for imagen in cargar_subset("train"):

        patient_id = imagen["patient_id"]
        image_id = imagen["image_id"]

        # 1. Cargar imagen y Ground Truth
        imagen_rgb = cargar_imagen(patient_id, image_id)
        anotaciones, _ = cargar_gt(patient_id, image_id)

        mascara_gt = construir_mascara_gt(imagen_rgb, anotaciones)
        campo = mascara_circular(imagen_rgb)

        # 2. Seleccionar canal verde
        verde = imagen_rgb[:, :, 1]

        # 3. Otsu sin filtro gaussiano
        umbral_sin = threshold_otsu(verde[campo])
        mascara_sin_filtro = (verde < umbral_sin) & campo

        # 4. Otsu con filtro gaussiano
        # El gaussiano ya no está dentro de segmentar_otsu_verde (se pasó a preprocesar_imagen en preprocess.py),
        # así que lo aplicamos acá. Si no, esta fila y las 3 a 6 serían Otsu SIN gaussiano aunque el nombre diga lo contrario.
        imagen_suavizada = cv2.GaussianBlur(imagen_rgb, (5, 5), 0)
        mascara_otsu, _ = segmentar_otsu_verde(imagen_suavizada, campo)

        # 5. Apertura morfológica
        kernel_apertura = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (KERNEL_APERTURA, KERNEL_APERTURA)
        )

        mascara_apertura = cv2.morphologyEx(
            mascara_otsu.astype("uint8"),
            cv2.MORPH_OPEN,
            kernel_apertura
        ).astype(bool)

        # 6. Cierre después de apertura
        kernel_cierre = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (KERNEL_CIERRE, KERNEL_CIERRE)
        )

        mascara_cierre = cv2.morphologyEx(
            mascara_apertura.astype("uint8"),
            cv2.MORPH_CLOSE,
            kernel_cierre
        ).astype(bool)

        # 7. Relleno después de apertura y cierre
        mascara_final = rellenar_huecos_chicos(
            mascara_cierre, HUECO_MAX
        )

        # 8. Solo relleno, sin apertura ni cierre
        mascara_solo_relleno = rellenar_huecos_chicos(
            mascara_otsu, HUECO_MAX
        )

        # 9. Generar versiones con distintos preprocesamientos
        verde_iluminacion = corregir_iluminacion(verde, campo)

        verde_clahe = mejorar_contraste_clahe(verde, campo)

        verde_ambos = mejorar_contraste_clahe(
            verde_iluminacion, campo
        )

        # 10. Aplicar Otsu con gaussiano y limpieza completa
        # a las tres versiones preprocesadas
        mascaras_preprocesamiento = {}

        versiones = {
            "7. Otsu + iluminacion": verde_iluminacion,
            "8. Otsu + CLAHE": verde_clahe,
            "9. Otsu + iluminacion + CLAHE": verde_ambos
        }

        for nombre, canal in versiones.items():

            suavizado = cv2.GaussianBlur(canal, (5, 5), 0)

            umbral = threshold_otsu(suavizado[campo])

            mascara = (suavizado < umbral) & campo

            mascaras_preprocesamiento[nombre] = limpiar_mascara(mascara)

        # 11. NUEVO: aplicar Otsu a CLAHE sin limpieza morfológica
        clahe_suavizado = cv2.GaussianBlur(verde_clahe, (5, 5), 0)

        umbral_clahe = threshold_otsu(clahe_suavizado[campo])

        mascara_clahe = (clahe_suavizado < umbral_clahe) & campo

        # 12. NUEVO: CLAHE + Otsu + solo relleno de huecos
        mascara_clahe_relleno = rellenar_huecos_chicos(
            mascara_clahe, HUECO_MAX
        )

        # 13. NUEVO: CLAHE + Otsu + limpieza completa
        mascara_clahe_completa = limpiar_mascara(mascara_clahe)

        # 14. Reunir todas las alternativas
        alternativas = {
            "1. Otsu sin filtro": mascara_sin_filtro,
            "2. Otsu con gaussiano": mascara_otsu,
            "3. Otsu + apertura": mascara_apertura,
            "4. Otsu + apertura + cierre": mascara_cierre,
            "5. Otsu + limpieza completa": mascara_final,
            "6. Otsu + solo relleno": mascara_solo_relleno
        }

        alternativas.update(mascaras_preprocesamiento)

        alternativas.update({
            "10. CLAHE + Otsu sin limpieza": mascara_clahe,
            "11. CLAHE + Otsu + solo relleno": mascara_clahe_relleno,
            "12. CLAHE + Otsu + limpieza completa": mascara_clahe_completa
        })

        # 15. Evaluar cada alternativa con el GT
        for metodo, mascara_pred in alternativas.items():

            metricas = evaluar_segmentacion(
                mascara_pred, mascara_gt, campo
            )

            resultados.append({
                "patient_id": patient_id,
                "image_id": image_id,
                "metodo": metodo,
                "IoU": metricas["IoU"],
                "Dice": metricas["Dice"]
            })

        print(f"Evaluada: {image_id}")

    # 16. Crear tabla y calcular promedios
    df = pd.DataFrame(resultados)

    resumen = df.groupby("metodo", sort=False)[["IoU", "Dice"]].mean()

    print("\nRESULTADOS PROMEDIO:")
    print(resumen.round(4))

    # 17. Guardar resultados en CSV
    carpeta = ROOT / "segmentation" / "results"
    carpeta.mkdir(parents=True, exist_ok=True)

    salida = carpeta / "comparacion_otsu.csv"
    df.to_csv(salida, index=False)

    print(f"\nResultados guardados en: {salida}")


if __name__ == "__main__":
    comparar_otsu()
