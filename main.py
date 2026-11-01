# Función del archivo: integra el pipeline completo (los tres bloques), cada bloque va agregando sus pasos.
# Ejecutar desde la carpeta principal del repo:  python main.py --split train

import numpy as np
from common.config import SEED
from common.io_utils import cargar_subset, cargar_imagen

# Cuando cada bloque tenga sus funciones, se importan acá, por ejemplo:
# from segmentation.preprocess import preprocesar                    # Bloque A
# from color_stain.wbc_detection import detectar_wbc                 # Bloque C
# from segmentation.cluster_seg import segmentar_eritrocitos         # Bloque A
# from segmentation.watershed_split import separar_celulas           # Bloque A
# from segmentation.label_filter import etiquetar_y_extraer          # Bloque A
# from color_stain.color_features import calcular_color              # Bloque C
# from texture_eval.glcm_features import calcular_textura            # Bloque B
# from texture_eval.auto_classifier import identificar_infectados    # Bloque B
# from color_stain.infected_seg import construir_mascara_infectados  # Bloque C
# from texture_eval.seg_metrics import evaluar                       # Bloque B (única parte que usa el GT)

#defino una función a la que entra una imagen y, mediante distintas funciones, se lleva a cabo todo el procesamiento.
def procesar_imagen(imagen_rgb): #imagen_rgb es el arreglo de numpy que devuelve cargar_imagenes
    
   # Orden previsto:
    #    1. imagen = preprocesar(imagen_rgb)
    #    2. wbc_mask, exclusion_mask = detectar_wbc(imagen)
    #    3. mascara = segmentar_eritrocitos(imagen, exclusion_mask)    # método elegido (Otsu / K-means / GMM)
    #    4. mascara = separar_celulas(mascara)                         # Watershed
    #    5. celulas = etiquetar_y_extraer(imagen, mascara)             # Merge Labeling + recortes
    #    6. color = calcular_color(celulas); textura = calcular_textura(celulas)
    #    7. infectadas = identificar_infectados(color, textura)
    #    8. return construir_mascara_infectados(celulas, infectadas)
    
    return None  # todavía no hay pasos implementados, esto hay que ir cambiandolo. 



#La función main va a encargarse de recorrer las imágenes del dataset y llamar a las funciones de procesamiento que desarrollen los tres bloques.
def main(split=None): #split: parámetro que indica qué imágenes querés procesar. 

#main(): se procesan todas las imágenes --> None está por default, si no le ingresamos ningún otro parámetro. 
#main("train"): se procesan sólo las imágenes del train
#main("test"): se procesan sólo las imágenes del test.


    np.random.seed(SEED)  # Fijar la semilla para reproducibilidad

    imagenes = cargar_subset(split)  # Obtener las imágenes, filtra según el valor de split, si corresponde.

    print(f"Imágenes a procesar: {len(imagenes)}")

    for imagen in imagenes: #recorre la lista de diccionarios
        # Cargar la imagen correspondiente
        imagen_rgb = cargar_imagen(imagen["patient_id"], imagen["image_id"]) #se guarda la imagen como un arreglo de numpy en formato RGB

        # Procesar la imagen con el pipeline
        resultado = procesar_imagen(imagen_rgb)

        # Mostrar la imagen procesada y su resultado
        print(f"{imagen['image_id']}: {resultado}")

#Cuando corremos el archivo main.py en vscode, se ejecuta directamente main("train")
if __name__ == "__main__":
    main("train")        