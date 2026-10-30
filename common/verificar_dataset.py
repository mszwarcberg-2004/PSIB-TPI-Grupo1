
# Función del archivo: verifica los archivos seleccionados y las anotaciones del dataset, que las funciones de io_utils.py funcionen bien

#Imports
from collections import Counter #permite contar cuantas veces aparece cada valor
from io_utils import cargar_subset, obtener_rutas, cargar_gt
from config import SUBSET_CSV

# Cargar la lista de imágenes seleccionadas
imagenes = cargar_subset()

# Guardar los pacientes que pertenecen a cada conjunto
pacientes_train = set() #set(): conjunto que almacena elementos sin repetirlos. Nos sirve porque cada paciente tiene dos imágenes, pero queremos contar cada paciente una sola vez.
pacientes_test = set()

# Contadores generales de células
total_parasitadas = 0
total_no_parasitadas = 0

for imagen in imagenes:

    patient_id = imagen["patient_id"]
    image_id = imagen["image_id"]
    split = imagen["split"]

    # Verificar que existan la imagen y su GT
    obtener_rutas(patient_id, image_id)

    # Leer las anotaciones del GT
    anotaciones, dimensiones = cargar_gt(patient_id, image_id)

    #Para recordar: en anotaciones, para cada célula guardamos un diccionario con: {    "cell_id": cell_id,    "label": clasificación parasitado o no,    "polygon": puntos})

    # Contar las células de cada clase: Counter recorre las etiquetas y cuenta cuántas veces aparece cada una.
    conteo = Counter(a["label"] for a in anotaciones) #recorremos las anotaciones de una clase y contamos cuántas células hay de cada clase

    #El resultado sería:
    #conteo = {
    #"Parasitized": 1,
   # "Uninfected": 2
   #}

    #Acá estamos accediendo a los valores del contador --> Importante: estas cantidades pertenecen a una sóla imagen, estamos dentro del for. 
    n_parasitadas = conteo["Parasitized"]
    n_no_parasitadas = conteo["Uninfected"]

    #+= es sumar un valor al que ya tenía la variable
    total_parasitadas += n_parasitadas
    total_no_parasitadas += n_no_parasitadas

    # Registrar los pacientes de cada conjunto: train o test
    if split == "train":
        pacientes_train.add(patient_id)
    elif split == "test":
        pacientes_test.add(patient_id)
    else:
        raise ValueError(f"Split no válido: {split}")
    
    #Imprimimos la información de cada imagen
    print(f"\nPaciente: {patient_id}")
    print(f"Imagen: {image_id}")
    print(f"Conjunto: {split}")
    print(f"Dimensiones GT: {dimensiones}")
    print(f"Parasitadas: {n_parasitadas}")
    print(f"No parasitadas: {n_no_parasitadas}")


# Verificar que ningún paciente esté en ambos conjuntos --> intersection busca los pacientes que están en ambos conjuntos
pacientes_repetidos = pacientes_train.intersection(pacientes_test)

if pacientes_repetidos:
    raise ValueError(
        f"Pacientes presentes en train y test: {pacientes_repetidos}"
    )


# Mostrar el resumen final del dataset
print("\nRESUMEN DEL DATASET")
print("Imágenes totales:", len(imagenes))
print("Pacientes de entrenamiento:", len(pacientes_train))
print("Pacientes de prueba:", len(pacientes_test))
print("Células parasitadas:", total_parasitadas)
print("Células no parasitadas:", total_no_parasitadas)
print("División por pacientes: correcta")
