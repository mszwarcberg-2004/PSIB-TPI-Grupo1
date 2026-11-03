# Función del archivo: integra el pipeline completo de los tres bloques.
# Ejecutar desde la carpeta principal del repo: python main.py
 
#Pipeline, para cada imagen (sin usar el GT):
#   1. Preprocesamiento: campo circular, CLAHE y filtro gaussiano.        (Bloque A)
#   2. Segmentación con Otsu y limpieza morfológica.                       (Bloque A)
#   3. Watershed para separar células pegadas.                             (Bloque A)
#   4. Exclusión de glóbulos blancos.                                      (A + C)
#   5. Filtrado por área y borde, recorte y máscara de cada célula.        (Bloque A)
#   6. Atributos de textura de cada célula (GLCM).                         (Bloque B)
#   7. Métricas de color para cada célula.                                 (Bloque C)
#   8. Regla de color de C5 (se guarda como referencia).                   (Bloque C)
#   9. Clasificación final con el método elegido en B (b_metodo_elegido.json). (B + C)
#  10. Parasitemia y máscara de eritrocitos infectados.                    (Bloque C)
#  11. Evaluación contra el GT.                                            (B + C)
 
#Todos los parámetros se eligen con train. Test se corre una única vez al final.
#El método de clasificación NO se elige acá: lo elige el notebook B (sección 8) con train y lo guarda
#en texture_eval/results/b_metodo_elegido.json. main.py solo lo lee y lo aplica.
 
import numpy as np
import pandas as pd
 
from common.config import SEED, FEATURES_DIR, rutas_segmentacion
from common.io_utils import cargar_subset, cargar_imagen, cargar_gt, cargar_celula
from common.generar_gt_utils import mapa_gt
 
# Bloque A (procesar_imagen se importa como segmentar_imagen para no confundirla con la de este archivo)
from segmentation.a1_segmentacion_base import procesar_imagen as segmentar_imagen
from segmentation.a1_segmentacion_base import limpiar_resultados_anteriores
from segmentation.a1_segmentacion_base import gt_sin_borde, CLASES_ERITROCITOS
from segmentation.preprocess import mascara_circular
 
# Bloque C
from color_stain.wbc_detection import detectar_wbc
from color_stain.normalizacion import mediana_fondo
from color_stain.color_features import caracteristicas_celula
from color_stain.color_classifier import clasificar_color, parasitemia, mascara_infectados
 
# Bloque B
from texture_eval import clasificacion_pred as clf
from texture_eval.glcm_features import glcm_celula
from texture_eval.seg_metrics import evaluar_segmentacion, evaluar_imagen
 
 
METODO = "otsu"  #método de segmentación: elegimos otsu porque tuvo los mejores resultados en train.
 
EVALUACION_FINAL = False  #va en falso hasta que terminemos todo: test no se corre hasta que todo esté fijo. Recién ahí se pone en True
 
CLASE_PARASITADA = "Parasitized"  #nombre de la clase parasitada en el GT
 
 
#Función prf: precisión, sensibilidad y F1 a partir de TP, FP y FN (devuelve 0 si el denominador es 0).
def prf(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    sensibilidad = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * sensibilidad / (precision + sensibilidad) if (precision + sensibilidad) else 0.0
    return precision, sensibilidad, f1
 
 
#Función procesar_imagen: recibe una imagen y le aplica todo el pipeline. No usa el GT, así que sirve para cualquier imagen nueva.
#decision: método de clasificación elegido en B (se carga una sola vez en main con clf.cargar_decision()).
def procesar_imagen(imagen_rgb, patient_id, image_id, split, rutas, decision):
 
    #Máscara binaria del tamaño de la imagen: 1 en los núcleos de glóbulos blancos, 0 en el resto.
    # Dentro de segmentar_imagen se saca la célula entera que los contiene
    mascara_wbc = detectar_wbc(imagen_rgb)
 
    # Pasos 1 a 5: preprocesamiento, segmentación, limpieza, Watershed, exclusión de glóbulos blancos y filtrado.
    # Guarda el recorte y la máscara de cada célula y devuelve: células aceptadas, máscara binaria, etiquetas y campo circular
    filas, mascara, etiquetas, campo = segmentar_imagen(
        imagen_rgb, patient_id, image_id, split, rutas, METODO, mascara_wbc
    )
 
    # Color de fondo de esta imagen (una vez por imagen). Se usa mascara_circular, igual que C4, C5 y C7
    fondo = mediana_fondo(imagen_rgb, mascara_circular(imagen_rgb))
 
    caracteristicas_textura = []
    caracteristicas_color = []
    pred_color = {}        # cell_id -> True/False (regla de C5 sola, queda como referencia)
    pred_final = {}        # cell_id -> True/False (método elegido en B)
    ids_parasitadas = []   # número de etiqueta (1, 2, 3...) de las células parasitadas según el método elegido
    decisiones = []        # una decisión por célula (método elegido), para la parasitemia
 
    for k, fila in enumerate(filas, start=1):  #k es el número de etiqueta de la célula en "etiquetas"
        crop, mask = cargar_celula(rutas, fila["cell_id"])  #recorte y máscara que guardó el Bloque A
 
        # Paso 6: textura
        atributos_tex = glcm_celula(crop, mask)
        if atributos_tex is not None:  #si no tiene pares de píxeles válidos, se saltea
            caracteristicas_textura.append({"cell_id": fila["cell_id"], **atributos_tex})
 
        # Paso 7: color (el recorte va SIN normalizar: caracteristicas_celula lo normaliza con la referencia de train)
        atributos_col = caracteristicas_celula(crop, mask, fondo)
        if atributos_col is not None:
            caracteristicas_color.append({"cell_id": fila["cell_id"], **atributos_col})
 
        # Paso 8: regla de color de C5 (None -> False)
        es_c5 = bool(clasificar_color(atributos_col))
        pred_color[fila["cell_id"]] = es_c5
 
        # Paso 9: método elegido en B (por ejemplo C5 AND textura). Sin textura calculada, la parte de textura da False
        es_parasitada = clf.decidir(decision, es_c5, atributos_tex)
        pred_final[fila["cell_id"]] = es_parasitada
        decisiones.append(es_parasitada)
        if es_parasitada:
            ids_parasitadas.append(k)
 
    # Paso 10: parasitemia (%) y máscara con los eritrocitos infectados completos
    porcentaje = parasitemia(decisiones)
    mascara_inf = mascara_infectados(etiquetas, ids_parasitadas)
 
    return {
        "mascara": mascara,          # máscara binaria de eritrocitos (antes del filtrado por área y borde)
        "etiquetas": etiquetas,      # 0 = fondo, 1, 2, 3... = células aceptadas
        "campo": campo,              # campo circular del microscopio
        "celulas": filas,            # una fila por célula aceptada (va a cells_pred.csv)
        "textura": caracteristicas_textura,
        "color": caracteristicas_color,
        "pred_color": pred_color,    # regla de C5 sola
        "pred_final": pred_final,    # método elegido en B
        "fondo": fondo,
        "parasitemia": porcentaje,
        "mascara_infectados": mascara_inf
    }
 
 
#Función evaluar_contra_gt: comprueba qué tan bien funcionó el algoritmo, comparando con el GT.
def evaluar_contra_gt(imagen_rgb, resultado, patient_id, image_id):
 
    anotaciones, _ = cargar_gt(patient_id, image_id)
 
    # Mapa de etiquetas del GT, solo con eritrocitos (sin glóbulos blancos)
    etiquetas_gt = mapa_gt(imagen_rgb.shape, anotaciones, clases=CLASES_ERITROCITOS)
 
    # IoU y Dice de toda la máscara, dentro del campo circular
    globales = evaluar_segmentacion(resultado["mascara"], etiquetas_gt > 0, resultado["campo"])
 
    # Detección célula a célula (IoU >= 0.5 = acierto). Sacamos del GT las células del borde, porque nosotros las descartamos a propósito
    etiquetas_gt_sb = gt_sin_borde(etiquetas_gt, resultado["campo"])
    deteccion = evaluar_imagen(etiquetas_gt_sb, resultado["etiquetas"], campo=resultado["campo"])
 
    # Clase del GT de cada célula detectada, con los mismos pares de la detección.
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
            "iou_gt": iou_gt.get(k, np.nan),
            "pred_color": resultado["pred_color"][fila["cell_id"]],   #decisión de la regla de color de C5 sola
            "pred_final": resultado["pred_final"][fila["cell_id"]]    #decisión del método elegido en B
        })
 
    # Clasificación sobre células que tienen pareja en el GT. Se cuenta para el método elegido y para C5 sola (referencia)
    def conteos(campo):
        tp = fp = fn = tn = 0
        for e in etiquetas_celulas:
            if e["clase_gt"] == "sin_pareja":
                continue
            real = e["clase_gt"] == CLASE_PARASITADA
            pred = e[campo]
            tp += int(real and pred)
            fp += int(not real and pred)
            fn += int(real and not pred)
            tn += int(not real and not pred)
        return tp, fp, fn, tn
 
    tp_c, fp_c, fn_c, tn_c = conteos("pred_final")   # método elegido
    tp_5, fp_5, fn_5, _ = conteos("pred_color")      # C5 sola
 
    # Parasitemia real: parasitadas / eritrocitos del GT (sin los del borde, igual que la detección)
    ids_gt = [i for i in np.unique(etiquetas_gt_sb) if i > 0]
    n_par_gt = sum(anotaciones[i - 1]["label"] == CLASE_PARASITADA for i in ids_gt)
    parasitemia_gt = 100 * n_par_gt / len(ids_gt) if ids_gt else 0.0
 
    metricas = {
        "patient_id": patient_id,
        "image_id": image_id,
        "IoU": globales["IoU"],
        "Dice": globales["Dice"],
        "TP": deteccion["TP"],
        "FP": deteccion["FP"],
        "FN": deteccion["FN"],
        "TP_cls": tp_c,      # método elegido
        "FP_cls": fp_c,
        "FN_cls": fn_c,
        "TN_cls": tn_c,
        "TP_c5": tp_5,       # C5 sola (referencia)
        "FP_c5": fp_5,
        "FN_c5": fn_5,
        "parasitemia_pred": resultado["parasitemia"],
        "parasitemia_gt": parasitemia_gt
    }
 
    return metricas, etiquetas_celulas
 
 
#Función main: recorre las imágenes del split, las procesa, las evalúa y guarda los resultados.
def main(split="train"):
 
    if split == "test" and not EVALUACION_FINAL:
        raise ValueError("Test se corre una única vez al final. Poner EVALUACION_FINAL = True cuando todo esté fijo.")
 
    np.random.seed(SEED)  #semilla para que los resultados se repitan
 
    # Método de clasificación elegido en el notebook B con train (no se compara ni se reajusta acá)
    decision = clf.cargar_decision()
    print(f"Método de clasificación: {decision['metodo']} (F1 LOPO en train: {decision['f1_lopo']})")
 
    imagenes = cargar_subset(split)
 
    print(f"Imágenes a procesar: {len(imagenes)} | split: {split} | método de segmentación: {METODO}")
 
    rutas = rutas_segmentacion(METODO)
    limpiar_resultados_anteriores(rutas)  #borra recortes, máscaras y CSV de la corrida anterior
 
    celulas_total = []
    textura_total = []
    color_total = []
    metricas_total = []
    etiquetas_total = []
    guardados = []
    pred_color_total = {}   # cell_id -> decisión de C5 sola
    pred_final_total = {}   # cell_id -> decisión del método elegido
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
            rutas,
            decision
        )
 
        celulas_total.extend(resultado["celulas"])
        textura_total.extend(resultado["textura"])
        color_total.extend(resultado["color"])
        pred_color_total.update(resultado["pred_color"])
        pred_final_total.update(resultado["pred_final"])
 
        if split == "test":
            anot, _ = cargar_gt(imagen["patient_id"], imagen["image_id"])
            gt_par = gt_sin_borde(mapa_gt(imagen_rgb.shape, anot, clases=[CLASE_PARASITADA]), resultado["campo"])
            guardados.append({"patient_id": imagen["patient_id"], "image_id": imagen["image_id"],
                              "celulas": resultado["celulas"], "campo": resultado["campo"],
                              "etiquetas": resultado["etiquetas"].astype(np.int16), "gt": gt_par.astype(np.int16)})
 
        metricas, etiquetas_celulas = evaluar_contra_gt(imagen_rgb, resultado, imagen["patient_id"], imagen["image_id"])
        metricas_total.append(metricas)
        etiquetas_total.extend(etiquetas_celulas)
 
        print(
            f"{imagen['image_id']}: "
            f"{len(resultado['celulas'])} células detectadas | "
            f"IoU {metricas['IoU']:.3f} | TP {metricas['TP']} FP {metricas['FP']} FN {metricas['FN']} | "
            f"parasitemia {metricas['parasitemia_pred']:.1f}% (GT {metricas['parasitemia_gt']:.1f}%)"
        )
 
    # Guardamos: células detectadas, textura, color, métricas por imagen y clase del GT de cada célula
    rutas["csv"].parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(celulas_total).to_csv(rutas["csv"], index=False)
 
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    ruta_textura = FEATURES_DIR / f"features_texture_{METODO}_{split}.csv"   # lleva el split para que test no pise a train
    pd.DataFrame(textura_total).to_csv(ruta_textura, index=False)
 
    # Paso 7: color (la referencia de fondo la maneja normalizacion.py; sale de train y test la reutiliza)
    color = pd.DataFrame(color_total)
    ruta_color = FEATURES_DIR / f"features_color_{METODO}_{split}.csv"
    color.to_csv(ruta_color, index=False)
 
    tabla_metricas = pd.DataFrame(metricas_total)
    ruta_metricas = rutas["csv"].parent / f"metricas_{split}.csv"
    tabla_metricas.to_csv(ruta_metricas, index=False)
 
    tabla_etiquetas = pd.DataFrame(etiquetas_total)
    ruta_etiquetas = rutas["csv"].parent / f"etiquetas_{split}.csv"
    tabla_etiquetas.to_csv(ruta_etiquetas, index=False)
 
    # Evaluación end-to-end contra el GT de parasitadas (matching IoU >= 0.5). Solo en test, una única vez.
    # Se evalúa el método elegido y, como referencia, la regla de C5 sola. No se reajusta nada.
    if split == "test":
        marcadas = {
            decision["metodo"]: {c for c, v in pred_final_total.items() if v},
            "C5 sola": {c for c, v in pred_color_total.items() if v},
        }
        print(clf.evaluar_test(guardados, marcadas, METODO).round(3))
 
    # Resumen de detección: IoU y Dice se promedian por imagen; TP, FP y FN se suman (cada célula pesa lo mismo)
    tp, fp, fn = tabla_metricas[["TP", "FP", "FN"]].sum()
    precision, sensibilidad, f1 = prf(tp, fp, fn)
 
    # Resumen de clasificación con el método elegido (sobre células con pareja en el GT)
    tpc, fpc, fnc = tabla_metricas[["TP_cls", "FP_cls", "FN_cls"]].sum()
    prec_c, sens_c, f1_c = prf(tpc, fpc, fnc)
 
    # Referencia: regla de C5 sola, con las mismas células
    tp5, fp5, fn5 = tabla_metricas[["TP_c5", "FP_c5", "FN_c5"]].sum()
    prec_5, sens_5, f1_5 = prf(tp5, fp5, fn5)
 
    # F1 por paciente del método elegido (para ver el peor caso)
    f1_pacientes = {}
    for pid, g in tabla_metricas.groupby("patient_id"):
        a, b, c = g[["TP_cls", "FP_cls", "FN_cls"]].sum()
        f1_pacientes[pid] = prf(a, b, c)[2]
 
    # Parasitemia global: parasitadas (método elegido) / total, sumando todas las imágenes
    total_celulas = len(tabla_etiquetas)
    total_par_pred = int(tabla_etiquetas["pred_final"].sum()) if total_celulas else 0
    parasitemia_global = 100 * total_par_pred / total_celulas if total_celulas else 0.0
 
    print(f"\nTotal: {len(celulas_total)} células -> {rutas['csv']}")
    print(f"Textura: {len(textura_total)} células -> {ruta_textura}")
    print(f"Color: {len(color_total)} células -> {ruta_color}")
    print(f"Métricas por imagen -> {ruta_metricas}")
    print(f"Clase del GT de cada célula -> {ruta_etiquetas}")
    print(f"\nSEGMENTACIÓN ({split.upper()}): IoU {tabla_metricas['IoU'].mean():.4f} | Dice {tabla_metricas['Dice'].mean():.4f}")
    print(f"DETECCIÓN DE CÉLULAS ({split.upper()}): TP {tp} | FP {fp} | FN {fn} | "
          f"Precisión {precision:.4f} | Sensibilidad {sensibilidad:.4f} | F1 {f1:.4f}")
    conteo = tabla_etiquetas["clase_gt"].value_counts()
    print(f"CLASE DEL GT DE LAS CÉLULAS DETECTADAS: " + " | ".join(f"{clase} {n}" for clase, n in conteo.items()))
    print(f"CLASIFICACIÓN, {decision['metodo']} ({split.upper()}): TP {tpc} | FP {fpc} | FN {fnc} | "
          f"Precisión {prec_c:.4f} | Sensibilidad {sens_c:.4f} | F1 {f1_c:.4f}")
    print(f"CLASIFICACIÓN, regla C5 sola, referencia ({split.upper()}): TP {tp5} | FP {fp5} | FN {fn5} | "
          f"Precisión {prec_5:.4f} | Sensibilidad {sens_5:.4f} | F1 {f1_5:.4f}")
    print("F1 por paciente: " + " | ".join(f"{pid} {v:.3f}" for pid, v in f1_pacientes.items()))
    print(f"PARASITEMIA ({split.upper()}): estimada {parasitemia_global:.2f}% | "
          f"GT promedio por imagen {tabla_metricas['parasitemia_gt'].mean():.2f}%")
 
 
if __name__ == "__main__":
    main("train")
 
 
#PENDIENTE:
#- Cuando todo esté fijo: EVALUACION_FINAL = True y main("test") una sola vez, sin tocar nada después.