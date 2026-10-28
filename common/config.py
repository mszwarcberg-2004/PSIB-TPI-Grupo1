#Función del archivo: centraliza las rutas del dataset y parámetros generales (semilla, escala, etc.).

from pathlib import Path

# Directorio principal del repositorio
ROOT = Path(__file__).resolve().parents[1]

# Rutas de los datos
DATA_DIR = ROOT / "data"
SELECTED_DIR = DATA_DIR / "raw"
SHARED_DIR = DATA_DIR / "shared"

SUBSET_CSV = SHARED_DIR / "subset.csv"
GT_DIR = SHARED_DIR / "gt"
PRED_DIR = SHARED_DIR / "pred"
FEATURES_DIR = SHARED_DIR / "features"

# Parametros generales
SEED = 42 #fija la semilla para que las selecciones aleatorias se hagan siempre igual, entonces sea reproducible.
SCALE = 1.0 #mantiene la resolución original de las imágenes.
TEST_SIZE = 0.25 #que el 25% de las imagenes que analizamos pertenezcan al test y el 75% al train. 