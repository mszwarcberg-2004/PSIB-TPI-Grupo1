# Función del archivo: contiene funciones compartidas para cargar imágenes y gt del dataset.
# IMPORTANTE: el algoritmo solo usa cargar_imagen(). cargar_gt() se usa únicamente en la verificación.


#Imports
from common.config import SELECTED_DIR, SUBSET_CSV
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


# Función para las anotaciones GT de una imagen. Se conservan TODAS las filas, incluidos los glóbulos blancos (White_Blood_Cell).
#Si una fila no trae un polígono válido (por ejemplo, los WBC), "polygon" vale None.

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


            # Obtener las coordenadas de los vértices. Coordenadas (x1, y1, x2, y2, ...). Si no se pueden leer, no hay polígono.

            try: #try: intentar ejecutar el código. Si ocurre un error de tipo ValueError, se ejecuta el except 
                coordenadas = [float(v) for v in fila[5:] if v.strip() != ""]  #toma todos los valores de la fila desde la posición 5, que corresponden a las coordenadas y convierte a decimales.
                #guarda todos los números en la lista coordenadas
            except ValueError: #si no se puede ejecutar, se guarda una lista vacía
                coordenadas = []

            polygon = None #inicialmente indicamos que no tenemos polígono. Si después encontramos coordenadas válidas, reemplazamos None por la matriz de coordenadas.

            if n_vertices > 0 and len(coordenadas) == 2 * n_vertices: #haya exactamente dos coordenadas por vértice, una x y una y.
                polygon = np.array(coordenadas).reshape(-1, 2) #convierte la lista en un arreglo de numpy con 2 columnas y las filas necesarías para 2 columnas (por eso el -1)
            
            #Ejemplo: polygon = np.array([
            #     [100.5, 200.0],
            #     [150.0, 220.0],
            #     [130.0, 250.0]            Cada fila representa un vértice del polígono de la célula.
            # ])
            
            #Se evalúa elif cuando no se cumplió el if anterior
            elif tipo.lower() == "polygon":
                # Una fila declarada como polígono tiene que tener sus coordenadas completas
                raise ValueError(f"Cantidad de coordenadas incorrecta en {cell_id}")

            anotaciones.append({
                "cell_id": cell_id,
                "label": label,
                "tipo": tipo,
                "polygon": polygon,
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
   

#Función construir_cell_id sirve para crear un identificador único para cada célula del dataset, de manera que puedas guardar sus recortes, máscaras y características sin confundirlas con otras células.
  
def construir_cell_id(patient_id, image_id, indice): #indice: numero que le asignamos a esa célula dentro de la imagen. 
    # Identificador único de una célula: {paciente}_{imagen sin extensión}_{n con 4 dígitos}.
    # Ejemplo: 208C67P28N_ThinF_IMG_20150819_115417_0007
    # Lo usan los tres bloques para nombrar recortes ({cell_id}.png), máscaras y filas de los CSV.
    
    return f"{patient_id}_{Path(image_id).stem}_{indice:04d}"
#indice:04d: escribe el índice de la célula con cuatro dígitos, agregando ceros a la izquierda si es necesario.
