#Función del archivo: centraliza las rutas del dataset y parámetros generales (semilla, escala, etc.).

from pathlib import Path

# ROOT: variable que guarda la ruta de la carpeta principal de tu repositorio
ROOT = Path(__file__).resolve().parents[1]

# Rutas de los datos
DATA_DIR = ROOT / "data" #ruta de la carpeta data del repo
SELECTED_DIR = DATA_DIR / "raw" #ruta de la carpeta raw dentro de data
SHARED_DIR = DATA_DIR / "shared" #ruta de la carpeta shared dentro de data
SUBSET_CSV = SHARED_DIR / "subset.csv" #ruta para entrar al subset.csv
GT_DIR = SHARED_DIR / "gt" #ruta para entrar a gt
PRED_DIR = SHARED_DIR / "pred" #ruta pra entrar 
FEATURES_DIR = SHARED_DIR / "features"

# Parametros generales
SEED = 42 #fija la semilla para que las selecciones aleatorias se hagan siempre igual, entonces sea reproducible.
SCALE = 1.0 #mantiene la resolución original de las imágenes.
TEST_SIZE = 0.25 #que el 25% de las imagenes que analizamos pertenezcan al test y el 75% al train. Proporción de pacientes que separamos para test

