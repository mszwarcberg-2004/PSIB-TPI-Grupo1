# Función del archivo: A1, segmentación base de punta a punta (Otsu + etiquetado + recortes + cells_pred.csv).

import pandas as pd
from common.config import SEED, CELLS_PRED_CSV
from common.io_utils import cargar_subset, cargar_imagen
from segmentation.preprocess import mascara_circular
from segmentation.cluster_segmentation import segmentar_otsu_verde, limpiar_mascara
from segmentation.etiquetado import etiquetar, filtrar_y_extraer
from common.config import CROPS_DIR, MASKS_DIR



def procesar_imagen_base(imagen_rgb, patient_id, image_id, split):
    # 1. Obtener la máscara circular
    campo = mascara_circular(imagen_rgb)

    # 2. Segmentar mediante Otsu
    mascara_inicial, umbral = segmentar_otsu_verde(imagen_rgb, campo)

    # 3. Aplicar operaciones morfológicas
    mascara_limpia = limpiar_mascara(mascara_inicial)

    # 4. Etiquetar los objetos de la máscara limpia
    etiquetas = etiquetar(mascara_limpia)

    # 5. Filtrar y extraer células
    return filtrar_y_extraer(
        imagen_rgb, etiquetas, campo, patient_id, image_id, split
    )


def limpiar_resultados_anteriores(): #que se borren las máscaras anteriores si estaban mal y lo estoy corriendo de nuevo
    # Eliminar recortes RGB anteriores
    for archivo in CROPS_DIR.glob("*.png"): #busca los png de la carpeta
        archivo.unlink() #unlink: elimina cada archivo

    # Eliminar máscaras anteriores
    for archivo in MASKS_DIR.glob("*.png"): #busca los png de la carpeta
        archivo.unlink() #unlink: elimina cada archivo

    # Eliminar CSV anterior
    if CELLS_PRED_CSV.exists():
        CELLS_PRED_CSV.unlink()


#Objetivo: ejecutar la segmentación A1 sobre todas las imágenes seleccionadas del dataset y guardar la información de las células detectadas.
def segmentacion_base(split=None):

     # Borrar resultados de ejecuciones anteriores
    limpiar_resultados_anteriores()

    filas_total = [] #lista vacía en la que vamos a guardar la información de todas las células detectadas en la imagen

    for imagen in cargar_subset(split): #consulta el archivo subset.csv y devuelve las imágenes correspondientes al conjunto seleccionado.
        patient_id, image_id = imagen["patient_id"], imagen["image_id"]
        imagen_rgb = cargar_imagen(patient_id, image_id)

        #Acá procesamos la imagen
        filas, _ = procesar_imagen_base(imagen_rgb, patient_id, image_id, imagen["split"]) #Ejecuta las etapas de segmentación sobre esa imagen
        #Acá no necesitamos las etiquetas, sólo las filas. 
        filas_total.extend(filas) #Agrega a filas_total la información de todas las células detectadas y aceptadas en la imagen actual.
        print(f"{patient_id} / {image_id}: {len(filas)} células") #Muestra en la terminal cuantas células se aceptaron de esa imagen. 

    #Guardamos todas las células ya procesadas en un cvs
    CELLS_PRED_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(filas_total).to_csv(CELLS_PRED_CSV, index=False) #convertimos la lista de registros en una tabla
    #to_csv(...) guarda esa tabla en el archivo cells_pred.csv.

    print(f"Total: {len(filas_total)} células -> {CELLS_PRED_CSV}") #print(f"Total: {len(filas_total)} células -> {CELLS_PRED_CSV}")

#Si ejecuto directamente este archivo, llamar a la función segmentacion_base().
if __name__ == "__main__":
  segmentacion_base("train")