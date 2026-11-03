# Función del archivo: integra el pipeline completo de los tres bloques.
# Ejecutar desde la carpeta principal del repo: python main.py

#Pipeline, para cada imagen (sin usar el GT):
#   1. Preprocesamiento: campo circular, CLAHE y filtro gaussiano.        (Bloque A)
#   2. Segmentación con Otsu y limpieza morfológica.                       (Bloque A)
#   3. Watershed para separar células pegadas.                             (Bloque A)
#   4. Exclusión de glóbulos blancos.                                      (A + C)
#   5. Filtrado por área y borde, recorte y máscara de cada célula.        (Bloque A)
#   6. Atributos de textura de cada célula (GLCM).                         (Bloque B)
#   7. Métricas de color para cada célula.                                  (Bloque C)
#   8. Definir umbrales para identificar células parasitadas vs no parasitadas
#   9. Hacer las predicciones para el test y comparar con el gt. 

#Todos los parámetros se eligen con train. Test se corre una única vez al final.

import numpy as np
import pandas as pd

from common.config import SEED, FEATURES_DIR, rutas_segmentacion
from common.io_utils import cargar_subset, cargar_imagen, cargar_gt, cargar_celula
from common.generar_gt_utils import mapa_gt

# Bloque A (procesar_imagen se importa como segmentar_imagen para no confundirla con la de este archivo)
from segmentation.a1_segmentacion_base import procesar_imagen as segmentar_imagen
from segmentation.a1_segmentacion_base import limpiar_resultados_anteriores
from segmentation.a1_segmentacion_base import gt_sin_borde, CLASES_ERITROCITOS

# Bloque C
from color_stain.color_features import (mediana_fondo, calcular_fondo_ref, guardar_fondo_ref,
                                        cargar_fondo_ref, color_celula)
from color_stain.wbc_detection import detectar_wbc

# Bloque B
from texture_eval import clasificacion_pred as clf
from texture_eval.glcm_features import glcm_celula
from texture_eval.seg_metrics import evaluar_segmentacion, evaluar_imagen


METODO = "otsu"  #método de segmentación: elegimos otsu porque tuvo los mejores resultados en train. 

EVALUACION_FINAL = False  #va en falso hasta que terminemos todo: test no se corre hasta que todo esté fijo. Recién ahí se pone en True

#Función procesar_imagen: recibe una imagen y le aplica todo el pipeline. No usa el GT, así que sirve para cualquier imagen nueva.
def procesar_imagen(imagen_rgb, patient_id, image_id, split, rutas):

    #Devuelve una máscara bianrai del tamaño de la imagen, que contiene 1 donde están los núcleos 
    # de los glóbulos blancos y 0 donde no están. 
    # Dentro de segmentar_imagen se saca la células entera que los contiene
    mascara_wbc = detectar_wbc(imagen_rgb)

    # Pasos 1 a 5: preprocesamiento, segmentación, limpieza, Watershed, exclusión de glóbulos blancos y filtrado.
    # Guarda el recorte y la máscara de cada célula y devuelve: células aceptadas, máscara binaria, etiquetas y campo circular
    filas, mascara, etiquetas, campo = segmentar_imagen(
        imagen_rgb, patient_id, image_id, split, rutas, METODO, mascara_wbc
    )

    # Paso 6: textura de cada célula
    caracteristicas_textura = []

    for fila in filas:
        crop, mask = cargar_celula(rutas, fila["cell_id"])  #recorte y máscara que guardó el Bloque A
        atributos = glcm_celula(crop, mask)

        if atributos is None:  #la célula no tiene pares de píxeles válidos, se saltea
            continue

        caracteristicas_textura.append({"cell_id": fila["cell_id"], **atributos})  #con cell_id para unirla después con el color

    
    parasitemia = None          #pendiente (ver abajo de todo)

    return {
        "mascara": mascara,          # máscara binaria de eritrocitos (antes del filtrado por área y borde)
        "etiquetas": etiquetas,      # 0 = fondo, 1, 2, 3... = células aceptadas
        "campo": campo,              # campo circular del microscopio
        "celulas": filas,            # una fila por célula aceptada (va a cells_pred.csv)
        "textura": caracteristicas_textura,
        "fondo": mediana_fondo(imagen_rgb, campo),
        "parasitemia": parasitemia
    }


#Función evaluar_contra_gt: comprueba qué tan bien funcionó el algoritmo, comparando los eritrocitos detectados con los del GT.
def evaluar_contra_gt(imagen_rgb, resultado, patient_id, image_id):

    anotaciones, _ = cargar_gt(patient_id, image_id)

    # Mapa de etiquetas del GT, solo con eritrocitos (sin glóbulos blancos)
    etiquetas_gt = mapa_gt(imagen_rgb.shape, anotaciones, clases=CLASES_ERITROCITOS)

    # IoU y Dice de toda la máscara, dentro del campo circular
    globales = evaluar_segmentacion(resultado["mascara"], etiquetas_gt > 0, resultado["campo"])

    # Detección célula a célula (IoU >= 0.5 = acierto). Sacamos del GT las células del borde, porque nosotros las descartamos a propósito
    deteccion = evaluar_imagen(gt_sin_borde(etiquetas_gt, resultado["campo"]), resultado["etiquetas"], campo=resultado["campo"])

    # Clase del GT de cada célula detectada, con los mismos pares de la detección (para elegir después los umbrales del clasificador).
    # pred_id es el número de la célula detectada y gt_id el lugar del polígono en las anotaciones (los dos empiezan en 1)
    clase_gt = {p["pred_id"]: anotaciones[p["gt_id"] - 1]["label"] for p in deteccion["pares"]}
    iou_gt = {p["pred_id"]: round(p["iou"], 4) for p in deteccion["pares"]}

    etiquetas_celulas = []
    for k, fila in enumerate(resultado["celulas"], start=1):
        etiquetas_celulas.append({
            "cell_id": fila["cell_id"],
            "image_id": image_id,
            "patient_id": patient_id,
            "clase_gt": clase_gt.get(k, "sin_pareja"),  #sin_pareja: no coincide con ningún eritrocito del GT
            "iou_gt": iou_gt.get(k, np.nan)
        })

    metricas = {
        "patient_id": patient_id,
        "image_id": image_id,
        "IoU": globales["IoU"],
        "Dice": globales["Dice"],
        "TP": deteccion["TP"],
        "FP": deteccion["FP"],
        "FN": deteccion["FN"]
    }

    return metricas, etiquetas_celulas


#Función main: recorre las imágenes del split, las procesa, las evalúa y guarda los resultados.
def main(split="train"):

    if split == "test" and not EVALUACION_FINAL:
        raise ValueError("Test se corre una única vez al final. Poner EVALUACION_FINAL = True cuando todo esté fijo.")

    np.random.seed(SEED)  #semilla para que los resultados se repitan

    imagenes = cargar_subset(split)

    print(f"Imágenes a procesar: {len(imagenes)} | split: {split} | método: {METODO}")

    rutas = rutas_segmentacion(METODO)
    limpiar_resultados_anteriores(rutas)  #borra recortes, máscaras y CSV de la corrida anterior

    celulas_total = []
    textura_total = []
    metricas_total = []
    etiquetas_total = []
    fondos = {}
    guardados = []
    for imagen in imagenes:

        imagen_rgb = cargar_imagen(
            imagen["patient_id"],
            imagen["image_id"]
        )

        resultado = procesar_imagen(
            imagen_rgb,
            imagen["patient_id"],
            imagen["image_id"],
            imagen["split"],
            rutas
        )

        celulas_total.extend(resultado["celulas"])
        textura_total.extend(resultado["textura"])

        fondos[imagen["image_id"]] = resultado["fondo"]
        if split == "test":
            anot, _ = cargar_gt(imagen["patient_id"], imagen["image_id"])
            gt_par = gt_sin_borde(mapa_gt(imagen_rgb.shape, anot, clases=["Parasitized"]), resultado["campo"])
            guardados.append({"patient_id": imagen["patient_id"], "image_id": imagen["image_id"],
                            "celulas": resultado["celulas"], "campo": resultado["campo"],
                            "etiquetas": resultado["etiquetas"].astype(np.int16), "gt": gt_par.astype(np.int16)})
    
        metricas, etiquetas_celulas = evaluar_contra_gt(imagen_rgb, resultado, imagen["patient_id"], imagen["image_id"])
        metricas_total.append(metricas)
        etiquetas_total.extend(etiquetas_celulas)

        print(
            f"{imagen['image_id']}: "
            f"{len(resultado['celulas'])} células detectadas | "
            f"IoU {metricas['IoU']:.3f} | TP {metricas['TP']} FP {metricas['FP']} FN {metricas['FN']}"
        )

    # Guardamos: células detectadas, textura, métricas por imagen y clase del GT de cada célula
    rutas["csv"].parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(celulas_total).to_csv(rutas["csv"], index=False)

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(textura_total).to_csv(FEATURES_DIR / f"features_texture_{METODO}.csv", index=False)

    tabla_metricas = pd.DataFrame(metricas_total)
    ruta_metricas = rutas["csv"].parent / f"metricas_{split}.csv"
    tabla_metricas.to_csv(ruta_metricas, index=False)

    tabla_etiquetas = pd.DataFrame(etiquetas_total)
    ruta_etiquetas = rutas["csv"].parent / f"etiquetas_{split}.csv"
    tabla_etiquetas.to_csv(ruta_etiquetas, index=False)
    
    # Paso 7: color (el fondo de referencia se fija con train y test lo reutiliza)
    if split == "train":
        fondo_ref = calcular_fondo_ref(fondos)
        guardar_fondo_ref(fondo_ref)
        print("Fondo de referencia (R, G, B):", fondo_ref.round(1), "| el de C4 era [220.3 181.1 166.4]")
    else:
        fondo_ref = cargar_fondo_ref()

    color_total = []
    for fila in celulas_total:
        crop, mask = cargar_celula(rutas, fila["cell_id"])
        color_total.append({"cell_id": fila["cell_id"],
                            **color_celula(crop, mask, fondos[fila["image_id"]], fondo_ref)})
    color = pd.DataFrame(color_total)
    color.to_csv(FEATURES_DIR / f"features_color_{METODO}_{split}.csv", index=False)

    # Pasos 8 y 9: color + textura
    tabla = clf.unir_features(pd.DataFrame(textura_total), color)
    sel_color, sel_tex = clf.cargar_seleccion()
    if split == "train":
        print(clf.entrenar(clf.con_etiquetas(tabla, tabla_etiquetas), sel_color, sel_tex, METODO).round(3))
    else:
        print(clf.evaluar_test(tabla, guardados, METODO).round(3))
        
    # Resumen: IoU y Dice se promedian por imagen; TP, FP y FN se suman (cada célula pesa lo mismo)
    tp, fp, fn = tabla_metricas[["TP", "FP", "FN"]].sum()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    sensibilidad = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * sensibilidad / (precision + sensibilidad) if (precision + sensibilidad) else 0.0

    print(f"\nTotal: {len(celulas_total)} células -> {rutas['csv']}")
    print(f"Textura: {len(textura_total)} células -> {FEATURES_DIR / f'features_texture_{METODO}.csv'}")
    print(f"Métricas por imagen -> {ruta_metricas}")
    print(f"Clase del GT de cada célula -> {ruta_etiquetas}")
    print(f"\nSEGMENTACIÓN ({split.upper()}): IoU {tabla_metricas['IoU'].mean():.4f} | Dice {tabla_metricas['Dice'].mean():.4f}")
    print(f"DETECCIÓN DE CÉLULAS ({split.upper()}): TP {tp} | FP {fp} | FN {fn} | "
          f"Precisión {precision:.4f} | Sensibilidad {sensibilidad:.4f} | F1 {f1:.4f}")
    conteo = tabla_etiquetas["clase_gt"].value_counts()
    print(f"CLASE DEL GT DE LAS CÉLULAS DETECTADAS: " + " | ".join(f"{clase} {n}" for clase, n in conteo.items()))
    print("CLASIFICACIÓN Y PARASITEMIA: pendientes (pasos 7 a 9)")


if __name__ == "__main__":
    main("train")


#PENDIENTE:
#7. Color de cada célula (Bloque C): pasar caracteristicas_color del notebook C4 a color_stain/color_features.py
#   y llamarla en el for de la textura, con el recorte normalizado (balance de blancos de normalizacion.py).
#8. Clasificación parasitada / sana (B + C): umbrales sobre textura y color, elegidos con etiquetas_train.csv.
#9. Parasitemia (parasitadas / total) y máscara final de infectados (Bloque C), en lugar de parasitemia = None.
#10. En evaluar_contra_gt: métricas de la clasificación y parasitemia estimada vs. la del GT.
