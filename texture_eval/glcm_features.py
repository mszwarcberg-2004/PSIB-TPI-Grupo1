#características de textura GLCM por eritrocito
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops

sys.path.append(str(Path(__file__).resolve().parents[1]))
from common.config import ROOT

# ---- Parámetros -----
LEVELS = 32 # niveles de gris tras cuantizar
DISTANCES = [1, 2, 3]
ANGLES = [0, np.pi/4, np.pi/2, 3*np.pi/4]
CHANNEL = "green" # mejor contraste en el canal verde
PROPS = ["contrast", "homogeneity", "energy", "correlation"]


def _a_gris(crop_rgb):
    if CHANNEL == "green":
        return crop_rgb[:, :, 1]
    return cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)


def _cuantizar(g):
    #Reduce 256 niveles a LEVELS, con el mismo mapeo para todas las células.
    return (g.astype(np.uint16) * LEVELS // 256).astype(np.uint8)


def _entropia(glcm):
    #Entropía de la GLCM normalizada. Entrada (L, L, D, A), salida (D, A).
    p = glcm
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.where(p > 0, np.log2(p), 0.0)
    return -(p * logp).sum(axis=(0, 1))


def glcm_celula(crop_rgb, mask):
    #Devuelve dict de descriptores promediados sobre los ángulos, por distancia
    q = _cuantizar(_a_gris(crop_rgb))
    mask = mask > 0

    # El fondo recibe un nivel reservado (LEVELS) que luego se descarta,
    # así solo cuentan pares de píxeles que están ambos dentro de la célula
    q = np.where(mask, q, LEVELS).astype(np.uint8)

    # Calculo GLCM simétrica para todas las distancias y ángulos
    glcm = graycomatrix(q, DISTANCES, ANGLES, levels=LEVELS + 1, symmetric=True, normed=False)
    glcm = glcm[:LEVELS, :LEVELS].astype(np.float64)  # saco fila/columna del fondo

    # Normalización: cada matriz (distancia, ángulo) pasa a sumar 1
    totales = glcm.sum(axis=(0, 1), keepdims=True)
    n_pares = int(totales.min())
    if n_pares == 0: #None si no hay pares válidos
        return None
    glcm /= totales

    out = {"n_pixeles": int(mask.sum()), "n_pares_min": n_pares}
    
    # Propiedades de skimage y promediando los 4 ángulos para cada distancia
    for prop in PROPS:
        vals = graycoprops(glcm, prop)          # (D, A)
        for i, d in enumerate(DISTANCES):
            out[f"{prop}_d{d}"] = vals[i].mean()
   
    #Entropía y mismo promedio por ángulo
    ent = _entropia(glcm)
    for i, d in enumerate(DISTANCES):
        out[f"entropy_d{d}"] = ent[i].mean()
    return out

def extraer_features(cells_csv, out_csv):
    """Lee cells_gt.csv o cells_pred.csv, calcula la GLCM de cada célula y guarda
    cell_id + descriptores."""
    cells = pd.read_csv(cells_csv)
    filas = []
    for _, r in cells.iterrows():
        p_crop, p_mask = Path(r.crop_path), Path(r.mask_path)
        p_crop = p_crop if p_crop.is_absolute() else Path(ROOT) / p_crop
        p_mask = p_mask if p_mask.is_absolute() else Path(ROOT) / p_mask
        crop = cv2.cvtColor(cv2.imread(str(p_crop)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(p_mask), cv2.IMREAD_GRAYSCALE)
        feats = glcm_celula(crop, mask)
        if feats is None:
            print(f"[aviso] sin pares válidos: {r.cell_id}")
            continue
        filas.append({"cell_id": r.cell_id, **feats})
    out = pd.DataFrame(filas)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    return out