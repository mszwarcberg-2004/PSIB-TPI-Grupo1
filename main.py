# Función del archivo: integra el pipeline completo (los tres bloques), cada bloque va agregando sus pasos.
# Ejecutar desde la carpeta principal del repo: python main.py

# PIPELINE (lo que charlamos en el grupo):
#  Para cada imagen (sin usar el GT):
#   1. Preprocesamiento: campo circular, CLAHE y filtro gaussiano.                  (Bloque A)
#   2. Segmentación con el método elegido (Otsu; K-means y GMM quedaron para comparar). (Bloque A)
#   3. Limpieza morfológica y Watershed para separar células pegadas.               (Bloque A)
#   4. Exclusión de glóbulos blancos (la máscara la entrega el Bloque C).           (A + C)
#   5. Filtrado por área y borde, recorte y máscara de cada célula.                 (Bloque A)
#   6. Atributos de textura de cada célula (GLCM).                                  (Bloque B)
#   7. Atributos de color de cada célula.                                           (Bloque C) -> PENDIENTE
#   8. Clasificación de cada célula en parasitada / sana, con umbrales.             (B + C)   -> PENDIENTE
#   9. Parasitemia de la imagen y máscara final de células infectadas.              (Bloque C) -> PENDIENTE
#  Después, solo para medir (acá recién se usa el GT):
#  10. Evaluación contra el GT: segmentación (IoU, Dice, detección célula a célula); clasificación y parasitemia -> PENDIENTE
#
# Todos los parámetros se eligen solo con train. Test se corre una única vez al final, con todo fijo.

import numpy as np
import pandas as pd

from common.config import SEED, FEATURES_DIR, rutas_segmentacion
from common.io_utils import cargar_subset, cargar_imagen, cargar_gt, cargar_celula
from common.generar_gt_utils import mapa_gt

# Bloque A: segmentación de eritrocitos (se importa con otro nombre para no pisar la función de este archivo)
from segmentation.a1_segmentacion_base import procesar_imagen as segmentar_imagen
from segmentation.a1_segmentacion_base import limpiar_resultados_anteriores
from segmentation.a1_segmentacion_base import gt_sin_borde, CLASES_ERITROCITOS

# Bloque C: detección de glóbulos blancos por color (núcleos violetas y oscuros)
from color_stain.wbc_detection import detectar_wbc

# Bloque B: extracción de características de textura y métricas de evaluación
from texture_eval.glcm_features import glcm_celula
from texture_eval.seg_metrics import evaluar_segmentacion, evaluar_imagen


# Cuando cada bloque tenga sus funciones, se importan acá:
# from color_stain.color_features import calcular_color              # Bloque C (paso 7)
# from color_stain.clasificador import clasificar_celula             # Bloques B + C (paso 8)
# from color_stain.infected_seg import construir_mascara_infectados  # Bloque C (paso 9)


# Método de segmentación que usa todo el pipeline: "otsu", "kmeans" o "gmm".
# Elegimos Otsu: con Watershed fue el mejor sobre train (F1 de detección 0.88, contra 0.84 de K-means y GMM). Ver report.md, sección 8.5.
METODO = "otsu"

# Candado para test: hasta el final del trabajo test no se toca (indicación del profesor).
# Solo cuando todos los parámetros estén fijos, se cambia a True para correr main("test") una única vez.
EVALUACION_FINAL = False


# Defino una función a la que entra una imagen y, mediante distintas funciones, se lleva a cabo todo el procesamiento.
# IMPORTANTE: esta función NO usa el GT, así que sirve para cualquier imagen nueva (por ejemplo, los "casos incógnita" de la cátedra).
def procesar_imagen(imagen_rgb, patient_id, image_id, split, rutas):

    # PASOS 1 a 5. BLOQUE A: segmentación de eritrocitos con el método elegido.
    # segmentar_imagen (a1_segmentacion_base.py) hace: preprocesamiento, segmentación, limpieza morfológica,
    # Watershed, filtrado por área y borde, y guarda el recorte y la máscara de cada célula.
    # Devuelve: células aceptadas (filas), máscara binaria, imagen de etiquetas de las células aceptadas y campo circular.

    # PASO 4 (Bloques C + A): exclusión de glóbulos blancos.
    # detectar_wbc (color_stain/wbc_detection.py) marca los núcleos violetas y oscuros (True = núcleo de glóbulo blanco).
    # Después, dentro de segmentar_imagen, se elimina el OBJETO entero que contiene cada núcleo (después de Watershed),
    # no solo los píxeles del núcleo, porque si no el citoplasma queda como pedazos que pasan por eritrocitos.
    mascara_wbc = detectar_wbc(imagen_rgb)

    filas, mascara, etiquetas, campo = segmentar_imagen(
        imagen_rgb, patient_id, image_id, split, rutas, METODO, mascara_wbc
    )

    # PASO 6. BLOQUE B: características de textura de cada célula detectada
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

    # PASO 7 (PENDIENTE, Bloque C): características de color de cada célula.
    # La función ya existe en el notebook C4 (caracteristicas_color), pero falta pasarla a color_stain/color_features.py.
    # Antes de calcularla hay que normalizar el recorte con el balance de blancos de C2 (referencia calculada solo con train).
    # Iría dentro del mismo for de arriba, por ejemplo:
    #     color = calcular_color(crop, mask)
    #     caracteristicas_color.append({"cell_id": fila["cell_id"], **color})
    caracteristicas_color = []

    # PASO 8 (PENDIENTE, Bloques B + C): clasificación de cada célula.
    # Con los atributos elegidos (textura: contrast_d3, correlation_d1, energy_d1; color: b_rel_p1, contraste_G_mancha, a_p95)
    # se aplican umbrales para decidir si cada célula está parasitada. Los umbrales se ajustan sobre las células
    # SEGMENTADAS de train (la etiqueta de cada una sale del GT con el matching), validando con un paciente afuera.
    # Iría acá, por ejemplo: fila["parasitada"] = clasificar_celula(atributos_textura, atributos_color)

    # PASO 9 (PENDIENTE, Bloque C): parasitemia y máscara final de infectados.
    # parasitemia = cantidad de células clasificadas como parasitadas / cantidad total de eritrocitos detectados.
    # La máscara final sería la unión de las máscaras de las células parasitadas:
    #     mascara_infectados = construir_mascara_infectados(etiquetas, filas)
    parasitemia = None

    return {
        "mascara": mascara,          # máscara binaria de eritrocitos (antes del filtrado por área y borde)
        "etiquetas": etiquetas,      # imagen de etiquetas: 0 = fondo, 1, 2, 3... = células aceptadas
        "campo": campo,              # campo circular del microscopio
        "celulas": filas,            # una fila por célula aceptada (lo que va a cells_pred.csv)
        "textura": caracteristicas_textura,
        "color": caracteristicas_color,
        "parasitemia": parasitemia
    }


# PASO 10. EVALUACIÓN CONTRA EL GT (solo para medir; no cambia nada del algoritmo).
# Compara el resultado de una imagen con sus anotaciones GT y devuelve las métricas de esa imagen.
def evaluar_contra_gt(imagen_rgb, resultado, patient_id, image_id):

    anotaciones, _ = cargar_gt(patient_id, image_id)

    # Mapa de etiquetas del GT con eritrocitos parasitados y no parasitados (sin glóbulos blancos)
    etiquetas_gt = mapa_gt(imagen_rgb.shape, anotaciones, clases=CLASES_ERITROCITOS)

    # a) Segmentación global: IoU y Dice entre la máscara y el GT, dentro del campo circular
    globales = evaluar_segmentacion(resultado["mascara"], etiquetas_gt > 0, resultado["campo"])

    # b) Detección célula a célula: cada célula aceptada contra cada célula del GT (IoU >= 0.5 = acierto).
    # Se sacan del GT las células cortadas por el borde, porque nosotros las descartamos a propósito.
    deteccion = evaluar_imagen(gt_sin_borde(etiquetas_gt, resultado["campo"]), resultado["etiquetas"], campo=resultado["campo"])

    # c) PENDIENTE: clasificación (sensibilidad, especificidad y F1 de las parasitadas), usando los pares
    #    célula predicha - célula del GT que devuelve evaluar_imagen en deteccion["pares"].
    # d) PENDIENTE: parasitemia estimada vs. parasitemia real del GT (calculada también sin las células del borde).

    return {
        "patient_id": patient_id,
        "image_id": image_id,
        "IoU": globales["IoU"],
        "Dice": globales["Dice"],
        "TP": deteccion["TP"],
        "FP": deteccion["FP"],
        "FN": deteccion["FN"]
    }


# La función main va a encargarse de recorrer las imágenes del dataset
# y llamar a las funciones de procesamiento que desarrollen los tres bloques.
def main(split="train"):

    # Candado: test solo se puede correr en la evaluación final
    if split == "test" and not EVALUACION_FINAL:
        raise ValueError("Test se corre una única vez al final. Poner EVALUACION_FINAL = True cuando todo esté fijo.")

    np.random.seed(SEED)  # Fijar la semilla para reproducibilidad

    imagenes = cargar_subset(split)

    print(f"Imágenes a procesar: {len(imagenes)} | split: {split} | método: {METODO}")

    rutas = rutas_segmentacion(METODO)
    limpiar_resultados_anteriores(rutas)  # borra recortes, máscaras y CSV de corridas anteriores

    celulas_total = []   # todas las células detectadas
    textura_total = []   # características de textura de todas las células
    metricas_total = []  # métricas de cada imagen contra el GT

    for imagen in imagenes:

        # Cargar la imagen RGB original
        imagen_rgb = cargar_imagen(
            imagen["patient_id"],
            imagen["image_id"]
        )

        # Procesar la imagen con el pipeline (sin GT)
        resultado = procesar_imagen(
            imagen_rgb,
            imagen["patient_id"],
            imagen["image_id"],
            imagen["split"],
            rutas
        )

        celulas_total.extend(resultado["celulas"])
        textura_total.extend(resultado["textura"])

        # Evaluar contra el GT (solo para medir)
        metricas = evaluar_contra_gt(imagen_rgb, resultado, imagen["patient_id"], imagen["image_id"])
        metricas_total.append(metricas)

        print(
            f"{imagen['image_id']}: "
            f"{len(resultado['celulas'])} células detectadas | "
            f"IoU {metricas['IoU']:.3f} | TP {metricas['TP']} FP {metricas['FP']} FN {metricas['FN']}"
        )

    # Guardar la lista de células (cells_pred.csv), las características de textura y las métricas por imagen
    rutas["csv"].parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(celulas_total).to_csv(rutas["csv"], index=False)

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(textura_total).to_csv(FEATURES_DIR / f"features_texture_{METODO}.csv", index=False)

    tabla_metricas = pd.DataFrame(metricas_total)
    ruta_metricas = rutas["csv"].parent / f"metricas_{split}.csv"
    tabla_metricas.to_csv(ruta_metricas, index=False)

    # Resumen general.
    # IoU y Dice: promedio de las imágenes. Detección: se suman TP, FP y FN de todas las imágenes (cada célula pesa lo mismo).
    tp, fp, fn = tabla_metricas[["TP", "FP", "FN"]].sum()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    sensibilidad = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * sensibilidad / (precision + sensibilidad) if (precision + sensibilidad) else 0.0

    print(f"\nTotal: {len(celulas_total)} células -> {rutas['csv']}")
    print(f"Textura: {len(textura_total)} células -> {FEATURES_DIR / f'features_texture_{METODO}.csv'}")
    print(f"Métricas por imagen -> {ruta_metricas}")
    print(f"\nSEGMENTACIÓN ({split.upper()}): IoU {tabla_metricas['IoU'].mean():.4f} | Dice {tabla_metricas['Dice'].mean():.4f}")
    print(f"DETECCIÓN DE CÉLULAS ({split.upper()}): TP {tp} | FP {fp} | FN {fn} | "
          f"Precisión {precision:.4f} | Sensibilidad {sensibilidad:.4f} | F1 {f1:.4f}")
    print("CLASIFICACIÓN Y PARASITEMIA: pendientes (pasos 7 a 9)")


# Cuando corremos el archivo main.py, se ejecuta sobre train.
if __name__ == "__main__":
    main("train")
