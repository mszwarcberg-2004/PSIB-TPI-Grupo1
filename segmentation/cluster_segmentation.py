# Función del archivo: segmentación semántica de eritrocitos (eritrocito vs fondo).
# Métodos: Otsu (referencia), K-means y GMM. Los tres devuelven (máscara binaria, información del método).

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from common.config import SEED

KERNEL_APERTURA = 5   # tamaño del kernel de la apertura (elimina ruido y uniones finas)
KERNEL_CIERRE = 7     # tamaño del kernel del cierre (une pedacitos de una misma célula)
HUECO_MAX = 4086      # área máxima (en píxeles) de un hueco que se rellena (centro claro del eritrocito) --> tiene un límite porque áreas más grandes pertenecen al fondo.
#Antes era 1500, elegido a mano. Ahora sale del GT de train: medimos los huecos que deja Otsu (después de apertura y cierre)
#y separamos los que caen DENTRO de un eritrocito del GT (son centros claros, hay que rellenarlos) de los que caen en el fondo.
#Los centros claros miden: mediana 773, percentil 95 = 2800 y percentil 99 = 4086 píxeles. Con 1500 quedaba sin rellenar el 27% de ellos.
#Elegimos el percentil 99. Probando 1500, 2800 y 4086 sobre train (validación dejando un paciente afuera), 4086 fue el mejor en todas las vueltas:
#F1 de detección 0.88 --> 0.94 e IoU global 0.802 --> 0.822.

K = 3                 # número de clusters: fondo, eritrocito y zonas oscuras
N_MUESTRA = 50000     # píxeles que se usan para ajustar K-means y GMM
LOTE = 2_000_000      # píxeles que se predicen por vez (para no llenar la memoria)


# MÉTODO: OTSU

#Otsu es el método de referencia. 
#Aplicamos otsu sobre el canal verde porque en imágenes de frotis sanguíneos teñidos con Giemsa, el canal verde puede presentar un contraste útil entre los eritrocitos rosados y el fondo claro.
def segmentar_otsu_verde(imagen_rgb, campo):

    # Extraer el canal verde de la imagen ya preprocesada
    verde = imagen_rgb[:, :, 1]

    # Calcular el umbral de Otsu dentro del campo circular
    umbral = threshold_otsu(verde[campo])

    # Generar la máscara binaria
    mascara = (verde < umbral) & campo

    return mascara, umbral


# LIMPIEZA MORFOLÓGICA (igual para todos los métodos)
#Estas tres funciones se habían borrado sin querer al agregar K-means y GMM. Las recuperamos del último commit, sin cambios,
#porque las usan a1_segmentacion_base.py, comparar_otsu.py y main.py (sin ellas aparecía un ImportError).

def _kernel(tamano): #Esta función crea un elemento estructurante elíptico del tamaño indicado --> para luego aplicar operaciones morfológicas sobre la segmentación hecha con Otsu
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tamano, tamano))

#Importante: en las imágenes teñidas, los eritrocitos muchas veces presentan una periferia más oscura y un centro más claro.
# El problema es que el centro claro sigue perteneciendo al eritrocito y al segmentarlos, ese centro podría quedar como un hueco negro.
def rellenar_huecos_chicos(mascara, area_max): #completa el interior de los eritrocitos.

    #Idea:
    # 1. busca huecos negros completamente encerrados por regiones blancas
    # 2. calcula cuántos píxeles tiene cada hueco.
    # 3. Rellena solamente los que tienen un área menor o igual a HUECO_MAX.
    # Devuelve la máscara con esos huecos rellenados.

    # identifica regiones negras completamente rodeadas de blanco y las convierte en blanco. Por ahora, las rellena todas, independientemente del tamaño.
    mascara_rellena = ndi.binary_fill_holes(mascara)

    # Identificar los píxeles que se rellenaron: los que están blancos en máscara_rellena y negros en máscara.
    huecos = mascara_rellena & ~mascara

    # Etiquetar los huecos y calcular sus áreas
    etiquetas, n = ndi.label(huecos)
    areas = ndi.sum(huecos, etiquetas, index=np.arange(1, n + 1))

    # Seleccionar los huecos que no superen el área máxima
    chicos = np.isin(etiquetas, np.where(areas <= area_max)[0] + 1)

    # Incorporar solamente los huecos pequeños a la máscara original
    return mascara | chicos


def limpiar_mascara(mascara): #Objetivo: mejorar la máscara binaria obtenida mediante Otsu, eliminando ruido mediante cierre y apertura
    m = mascara.astype(np.uint8) #Convierte una máscara booleana (True y False) a una matriz de enteros (1 y 0).
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, _kernel(KERNEL_APERTURA)) #Aplicamos apertura: elimina pequeñas regiones blancas aisladas que Otsu detectó erróneamente como eritrocitos. También puede romper conexiones muy finas entre células.
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, _kernel(KERNEL_CIERRE)) #Aplicamos cierre: une pequeños fragmentos blancos cercanos y cierra pequeñas discontinuidades dentro de las regiones segmentadas.
    return rellenar_huecos_chicos(m.astype(bool), HUECO_MAX)


# MÉTODOS: K-MEANS Y GMM (clustering por color)
#Pasamos a Lab porque separa el brillo (L) de la información cromática (a y b), y así el fondo claro se distingue fácil por L.

#Usamos SOLO el espacio Lab. En el análisis de color hecho en C, Lab fue el espacio que mejor agrupó los píxeles sin etiquetas

def _agrupar_por_color(imagen_rgb, campo, modelo):
    # Ajusta el modelo sobre una muestra de píxeles en el espacio Lab. Devuelve la máscara binaria de eritrocitos y los centros de los clusters.

    # Convertir la imagen al espacio Lab
    imagen_color = cv2.cvtColor(imagen_rgb, cv2.COLOR_RGB2LAB)

    # Convertir a float32 para aplicar clustering
    imagen_color = imagen_color.astype(np.float32)

    # Seleccionar solamente los píxeles dentro del campo circular
    datos = imagen_color[campo]

    # Muestra aleatoria reproducible para ajustar el modelo
    #usamos una muestra porque  no necesitamos utilizar los millones de píxeles de la imagen para ajustar el modelo. 
    # Seleccionar una muestra reduce considerablemente el costo computacional
    rng = np.random.default_rng(SEED)
    muestra = datos[
        rng.integers(0, len(datos), size=min(N_MUESTRA, len(datos)))
    ]

    # Entrena al modelo K-means o GMM sobre los píxeles en la muestra
    modelo.fit(muestra)

    # grupos: asigna cada píxel de la imagen a uno de los clusters que encontró K-means o GMM.
    grupos = np.concatenate([
        modelo.predict(datos[i:i + LOTE])
        for i in range(0, len(datos), LOTE)
    ])

    # 1. centros: obtiene los centroides/ el color representativo de cada cluster.
    centros = (
        modelo.cluster_centers_
        if hasattr(modelo, "cluster_centers_")
        else modelo.means_
    )

    # Identificar el fondo: en Lab, el canal L representa claridad.
    fondo = np.argmax(centros[:, 0]) #Suponemos que el cluster más claro es el fondo.

    # Construir máscara binaria: fondo = 0, resto de clusters = 1
    mascara = np.zeros(campo.shape, dtype=bool)
    mascara[campo] = grupos != fondo

    return mascara, centros


def segmentar_kmeans(imagen_rgb, campo, k=K):
    # K-means agrupa los píxeles según su color.
    modelo = KMeans(n_clusters=k, n_init=5, random_state=SEED)

    return _agrupar_por_color(imagen_rgb, campo, modelo)


def segmentar_gmm(imagen_rgb, campo, k=K):
    # GMM modela los grupos de píxeles mediante distribuciones gaussianas.
    modelo = GaussianMixture(
        n_components=k,
        covariance_type="full",
        random_state=SEED
    )

    return _agrupar_por_color(imagen_rgb, campo, modelo)


def segmentar(metodo, imagen_rgb, campo):
    # Llama al método de segmentación elegido.
    if metodo == "otsu":
        return segmentar_otsu_verde(imagen_rgb, campo)

    if metodo == "kmeans":
        return segmentar_kmeans(imagen_rgb, campo)

    if metodo == "gmm":
        return segmentar_gmm(imagen_rgb, campo)

    raise ValueError(f"Método desconocido: {metodo}")
