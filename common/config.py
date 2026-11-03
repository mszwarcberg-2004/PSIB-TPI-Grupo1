
# Función del archivo: centraliza las rutas del dataset
# y los parámetros generales del proyecto.

from pathlib import Path

# Carpeta principal del repositorio
ROOT = Path(__file__).resolve().parents[1]

# Rutas generales de los datos
DATA_DIR = ROOT / "data"
SELECTED_DIR = DATA_DIR / "raw"
SHARED_DIR = DATA_DIR / "shared"

SUBSET_CSV = SHARED_DIR / "subset.csv"
GT_DIR = SHARED_DIR / "gt"
FEATURES_DIR = SHARED_DIR / "features"
CHECK_DIR = SHARED_DIR / "gt_check"

# Carpeta general de resultados de segmentación
PRED_DIR = SHARED_DIR / "pred"


# Devuelve las rutas donde se guardarán los resultados según el método de segmentación utilizado. Recibe el nombre del método
def rutas_segmentacion(metodo):

    carpeta = PRED_DIR / metodo

    return {
        "crops": carpeta / "crops",
        "masks": carpeta / "masks",
        "csv": carpeta / "cells_pred.csv"
    }


# Parámetros generales
SEED = 42
SCALE = 1.0
TEST_SIZE = 0.25

# Nombres de las clases según el Ground Truth
LABEL_PARASITIZED = "Parasitized"
LABEL_UNINFECTED = "Uninfected"
LABEL_WBC = "White_Blood_Cell"
