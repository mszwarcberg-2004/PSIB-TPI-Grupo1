
# Función del archivo: A1, segmentación base de punta a punta (Otsu + etiquetado + recortes + cells_pred.csv).

import pandas as pd
from common.config import rutas_segmentacion
from common.io_utils import cargar_subset, cargar_imagen, cargar_gt
from common.visualizar_gt import construir_mascara_gt
from segmentation.preprocess import mascara_circular, mejorar_contraste_clahe
from segmentation.cluster_segmentation import segmentar_otsu_verde, limpiar_mascara
from segmentation.etiquetado import etiquetar, filtrar_y_extraer
from evaluacion.evaluacion import evaluar_segmentacion


def procesar_imagen_base(imagen_rgb, patient_id, image_id, split, rutas): #procesa una imagen

    # 1. Obtener la máscara circular del microscopio
    campo = mascara_circular(imagen_rgb)

    # 2. Extraer el canal verde
    verde = imagen_rgb[:, :, 1]

    # 3. Mejorar el contraste local mediante CLAHE
    verde_clahe = mejorar_contraste_clahe(verde, campo)

    # 4. Construir una imagen RGB auxiliar para usar la función de Otsu que ya tenemos programada
    imagen_clahe = imagen_rgb.copy() #trabajamos con una copia de la imagen original
    imagen_clahe[:, :, 1] = verde_clahe

    # 5. Segmentar mediante Otsu
    # Esta función selecciona el canal verde y aplica el filtro gaussiano
    mascara_inicial, umbral = segmentar_otsu_verde(imagen_clahe, campo)

    # 6. Aplicar apertura, cierre y relleno de huecos
    mascara_limpia = limpiar_mascara(mascara_inicial)

    # 7. Etiquetar los objetos de la máscara limpia
    etiquetas = etiquetar(mascara_limpia)

    # 8. Filtrar y extraer las células individuales
    # Usamos la imagen RGB ORIGINAL para conservar sus colores reales
    filas, etiquetas_ok = filtrar_y_extraer(
        imagen_rgb, etiquetas, campo, patient_id, image_id, split, rutas
    )

    return filas, mascara_limpia #devolvemos las células detectadas y la máscara final de Otsu que queremos evaluar.


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


#Objetivo: ejecutar la segmentación A1 sobre todas las imágenes seleccionadas del dataset y guardar la información de las células detectadas.
def segmentacion_base(split="train"): #procesa todas las imágenes

    # Evaluamos solamente train, reservando test para la evaluación final.
    if split != "train":
        raise ValueError("Por ahora solamente se permite ejecutar train.")

    # Elegir las rutas del método de segmentación
    rutas = rutas_segmentacion("otsu")

    # Borrar resultados de ejecuciones anteriores
    limpiar_resultados_anteriores(rutas)

    filas_total = [] #lista vacía en la que vamos a guardar la información de todas las células detectadas en la imagen
    resultados_metricas = [] #guardamos las métricas para luego calcular el promedio

    for imagen in cargar_subset(split): #consulta el archivo subset.csv y devuelve las imágenes correspondientes al conjunto seleccionado.
        patient_id, image_id = imagen["patient_id"], imagen["image_id"]
        imagen_rgb = cargar_imagen(patient_id, image_id)

        #Acá procesamos la imagen
        filas, mascara_limpia = procesar_imagen_base(
            imagen_rgb, patient_id, image_id, imagen["split"], rutas
        ) #Ejecuta las etapas de segmentación sobre esa imagen

        #Acá no necesitamos las etiquetas, sólo las filas.
        filas_total.extend(filas) #Agrega a filas_total la información de todas las células detectadas y aceptadas en la imagen actual.
        print(f"{patient_id} / {image_id}: {len(filas)} células") #Muestra en la terminal cuantas células se aceptaron de esa imagen.

        # Evaluar la máscara de Otsu contra el GT de train
        anotaciones, _ = cargar_gt(patient_id, image_id)
        mascara_gt = construir_mascara_gt(imagen_rgb, anotaciones)

        campo = mascara_circular(imagen_rgb)
        metricas = evaluar_segmentacion(mascara_limpia, mascara_gt, campo)

        # Guardamos las métricas en memoria para calcular el promedio general
        resultados_metricas.append(metricas)

    #Guardamos todas las células ya procesadas en un cvs
    rutas["csv"].parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(filas_total).to_csv(rutas["csv"], index=False) #convertimos la lista de registros en una tabla
    #to_csv(...) guarda esa tabla en el archivo cells_pred.csv.

    print(f"Total: {len(filas_total)} células -> {rutas['csv']}") #print(f"Total: {len(filas_total)} células -> {CELLS_PRED_CSV}")

    # Calcular únicamente el IoU y Dice promedio de todas las imágenes de train
    if not resultados_metricas:
        raise ValueError("No se encontraron imágenes de train para evaluar.")

    promedio = pd.DataFrame(resultados_metricas)[["IoU", "Dice"]].mean()

    print("\nMÉTRICAS GENERALES DE OTSU (TRAIN):")
    print(f"IoU promedio: {promedio['IoU']:.4f}")
    print(f"Dice promedio: {promedio['Dice']:.4f}")


#Si ejecuto directamente este archivo, llamar a la función segmentacion_base().
if __name__ == "__main__":
    segmentacion_base("train")
