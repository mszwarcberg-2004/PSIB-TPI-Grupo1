# Función del archivo: contiene funciones compartidas para cargar imágenes y gt del dataset.

#Imports
from config import SELECTED_DIR, SUBSET_CSV 
#importo la variable SELECTED_DIR con la ruta hasta raw, donde están los datos
#importar la variable SUBSET_CVS con la ruta hasta el subset.csv, que tiene toda la info de las imágenes que procesamos
import cv2
import csv
import numpy as np

#Función para obtener las rutas y encontrar las imagenes y el gt de los distintos pacientes. 
def obtener_rutas(patient_id, image_id):
    # Buscar la carpeta del paciente seleccionado
    carpeta_paciente = SELECTED_DIR / patient_id

    # Construir las rutas de la imagen y su anotación GT
    ruta_imagen = carpeta_paciente / "Img" / image_id
    ruta_gt = carpeta_paciente / "GT" / f"{ruta_imagen.stem}.txt" #ruta_imagen.stem: permite obtener la ruta del archivo sin la parte .jpg

    # Comprobar que ambos archivos existan
    if not ruta_imagen.is_file():
        raise FileNotFoundError(f"No se encontró la imagen: {ruta_imagen}")

    if not ruta_gt.is_file():
        raise FileNotFoundError(f"No se encontró el GT: {ruta_gt}")

    return ruta_imagen, ruta_gt


# Cargar una imagen del dataset en formato RGB
def cargar_imagen(patient_id, image_id):

    # Obtener la ruta de la imagen
    ruta_imagen, _ = obtener_rutas(patient_id, image_id)
    #Llamamos a nuestra función anterior, que devuelve dos valores: la ruta de la imagen y la ruta del GT.
    # Como ahora solamente necesitamos la imagen, guardamos su ruta en ruta_imagen y utilizamos _ para indicar que no vamos a usar el segundo valor

    # Leer la imagen con OpenCV: abre la imagen y la convierte en una matriz de numpy: (alto, ancho,3) porque tiene 3 colores
    imagen = cv2.imread(str(ruta_imagen), cv2.IMREAD_COLOR)

    # Verificar que se haya cargado correctamente
    if imagen is None:
        raise ValueError(f"No se pudo cargar la imagen: {ruta_imagen}")

    # Convertir de BGR a RGB
    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

    return imagen


# Función para las anotaciones GT de una imagen
def cargar_gt(patient_id, image_id):

    # Obtener la ruta del archivo GT
    _, ruta_gt = obtener_rutas(patient_id, image_id) #misma idea que antes, sólo me importa acá que la función obtener_rutas devuelta la ruta de gt

    anotaciones = []

    # Abrir el archivo y leer las filas separadas por comas
    with open(ruta_gt, newline="", encoding="utf-8-sig") as archivo: #asigna el archivo abierto a la variable archivo
        lector = csv.reader(archivo) #csv.reader() permite leer cada fila como una lista de valores, utiliza la coma como separador por defecto

        # La primera fila contiene: CANTIDAD DE OBJETOS DETECTADOS EN LA IMAGEN,ANCHO DE LA IMAGEN EN PÍXELES, ALTO DE LA IMAGEN EN PÍXELES
        cabecera = next(lector) # Crea una lista con la primera linea del gt
        cantidad = int(cabecera[0]) #accede al primer valor de la lista y la convierte a un valor entero, la cantidad de objetos detectados
        ancho = int(cabecera[1]) #Convierte el segundo valor a entero: ancho de la imagen.
        alto = int(cabecera[2]) #Convierte el tercer valor a entero: ancho de la imagen.

        # Recorrer las anotaciones de las células
        for fila in lector:    #Recorre las filas restantes del GT, una por una
            if not fila: #si la fila está vacía, frena
                continue

            cell_id = fila[0].strip() #strip() elimina el espacio en blanco al principio y al final del texto
            label = fila[1].strip() #label: Parasitized, Uninfected o White_Blood_Cell
            tipo = fila[3].strip() #tipo de anotación: el contorno de los eritrocitos se representa con un polígono, glóbulos blancos dice "No comment" --> polígono: hacemos un contorno de la célula
            n_vertices = int(fila[4]) #cantidad de vertices del polígono 
            #El GT no nos da directamente una imagen recortada de cada célula, sino los puntos que indican dónde está su contorno.

            # Procesar solamente anotaciones de tipo polígono; si el tipo de anotación no es "polygon", salteamos esa fila.
            if tipo.lower() != "polygon":
                continue

            # Obtener las coordenadas de los vértices
            coordenadas = [float(valor) for valor in fila[5:]] #toma todos los valores desde la posición 5 hasta el final de la fila,
            #los convierte a numero decimal y los guarda en una lsita que se llama coordenadas

            if len(coordenadas) != 2 * n_vertices:
                raise ValueError(
                    f"Cantidad de coordenadas incorrecta en {cell_id}"
                )
            #Si la cantidad de coordenadas no es el doble de la cantidad de vértices (debería ser el doble porque hay [x,y] para cada vértice), tira error

            # Agrupar las coordenadas en pares (x, y)
            puntos = np.array(coordenadas).reshape(-1, 2) # Agrupa las coordenadas en pares (x, y); -1 calcula automáticamente la cantidad de filas, a partir de la cantidad de columnas y la cantidad de datos
            
            # Vamos guardando los datos de cada célula en un diccionario llamado anotaciones con tres claves: cell_id, label, y la matriz con las coordenadas de los puntos que forman el polígono
            anotaciones.append({
                "cell_id": cell_id,
                "label": label,
                "polygon": puntos
            })

    # Comprobar la cantidad de anotaciones leídas
    if len(anotaciones) != cantidad:
        print(
            f"Advertencia: se esperaban {cantidad} anotaciones "
            f"y se leyeron {len(anotaciones)}."
        )

    return anotaciones, (ancho, alto)


# Leer las imágenes seleccionadas y su división en entrenamiento y prueba en subset.csv
def cargar_subset(split=None):

#split es opcional: puedo correr esta funcion de tres maneras:

    #cargar_subset()             # Si no especificamos nada, devuelve todas las imágenes
    #cargar_subset("train")      # Solo imágenes de entrenamiento
    #cargar_subset("test")       # Solo imágenes de prueba

    # Abrir el archivo CSV con la lista de imágenes seleccionadas
    with open(SUBSET_CSV, newline="", encoding="utf-8-sig") as archivo:
        lector = csv.DictReader(archivo) #csv.DictReader() utiliza la primera fila del CSV como nombres de columnas: convierte cada fila en un diccionario, y la primera fila son los nombres de las claves. 

        #Ejemplo: la primera fila se representa así
        #{
        #"image_id": "IMG_01.jpg",
        #"patient_id": "Paciente1",
        #"split": "train"

        imagenes = list(lector) #lista que contiene a todos los diccionarios, cada uno correspondiente a una fila (es decir, a una imagen)

    # Filtrar por train o test solamente si se solicita
    if split is not None:
        filtradas = []

        for imagen in imagenes:
            if imagen["split"] == split:
                filtradas.append(imagen)

        imagenes = filtradas

    #Por ejemplo: si pusimos 'train' en la columna split, conserva sólo los diccionarios cuya columna split sea 'train'

    return imagenes
   
