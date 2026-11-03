"""C4: características de color por célula.

Copia fiel de las funciones del notebook c4_caracteristicas_color.ipynb, con los parámetros de C3
(elegidos con train). Se pasan a un módulo para poder usarlas desde main.py con la segmentación propia.

Importante: la referencia de fondo (FONDO_REF) es un parámetro que se calcula UNA vez con las imágenes
de train y se reutiliza tal cual en test. Por eso se guarda en disco.
"""
import json
from pathlib import Path

import cv2
import numpy as np

# Parámetros de la máscara de manchas elegidos en C3 (solo con train)
PARAMS_C3 = {"erosion": 5, "umbral": -39, "area_min": 50}

RUTA_FONDO_REF = Path(__file__).resolve().parent / "results" / "fondo_ref_train.json"


# ---------- Normalización por fondo (C2) ----------

def mediana_fondo(imagen_rgb, campo):
    """Mediana R, G, B del fondo, estimado sin GT con Otsu sobre G."""
    verde = imagen_rgb[:, :, 1]
    umbral, _ = cv2.threshold(verde[campo].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return np.median(imagen_rgb[campo & (verde > umbral)].astype(float), axis=0)


def calcular_fondo_ref(fondos):
    """Referencia = promedio de los fondos de las imágenes de train. fondos: dict image_id -> (R, G, B)."""
    return np.mean(list(fondos.values()), axis=0)


def guardar_fondo_ref(fondo_ref, ruta=RUTA_FONDO_REF):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps([float(x) for x in fondo_ref]))


def cargar_fondo_ref(ruta=RUTA_FONDO_REF):
    if not ruta.exists():
        raise FileNotFoundError(f"Falta {ruta}: se genera al correr main('train'). Test no la recalcula.")
    return np.array(json.loads(ruta.read_text()))


def normalizar_fondo(recorte_rgb, fondo_imagen, fondo_ref):
    return np.clip(recorte_rgb * (fondo_ref / fondo_imagen), 0, 255).astype(np.uint8)


# ---------- Máscara de manchas oscuras (C3) ----------

def kernel_eliptico(radio):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radio + 1, 2 * radio + 1))


def manchas_c3(verde, mascara, erosion, umbral, area_min):
    """Píxeles con G muy por debajo de la mediana de la célula, solo en manchas grandes.
    Devuelve la máscara final y el área de cada mancha."""
    dentro = cv2.erode(mascara, kernel_eliptico(erosion)) > 0 if erosion > 0 else mascara > 0
    if not dentro.any():
        return np.zeros_like(dentro), np.array([], dtype=int)
    relativo = verde.astype(np.int16) - int(np.median(verde[dentro]))
    marcados = ((relativo < umbral) & dentro).astype(np.uint8)
    n, etiquetas, stats, _ = cv2.connectedComponentsWithStats(marcados, connectivity=8)
    grandes = [k for k in range(1, n) if stats[k, cv2.CC_STAT_AREA] >= area_min]
    return np.isin(etiquetas, grandes), stats[grandes, cv2.CC_STAT_AREA]


# ---------- Características ----------

def caracteristicas_color(recorte_rgb, mascara):
    """17 características de color + n_pixeles de una célula (recorte ya normalizado por fondo)."""
    mascara = (mascara > 0).astype(np.uint8)          # 0/1 uint8, para que cv2.erode funcione con cualquier formato
    dentro = mascara > 0
    hsv = cv2.cvtColor(recorte_rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(recorte_rgb, cv2.COLOR_RGB2LAB)
    canales = {"G": recorte_rgb[:, :, 1], "L": lab[:, :, 0], "a": lab[:, :, 1], "b": lab[:, :, 2], "S": hsv[:, :, 1]}
    valores = {nombre: canal[dentro].astype(float) for nombre, canal in canales.items()}

    f = {"n_pixeles": int(dentro.sum())}

    # a) Color de la célula entera
    for nombre, v in valores.items():
        f[f"{nombre}_mediana"] = np.median(v)

    # b) Píxeles extremos (absolutos y relativos a la mediana de la célula)
    for nombre in ["G", "L", "b"]:
        f[f"{nombre}_p5"] = np.percentile(valores[nombre], 5)
    for nombre in ["a", "S"]:
        f[f"{nombre}_p95"] = np.percentile(valores[nombre], 95)
    for nombre in ["G", "b"]:
        f[f"{nombre}_rel_p1"] = np.percentile(valores[nombre], 1) - f[f"{nombre}_mediana"]

    # c) Manchas de la máscara C3
    manchas, areas = manchas_c3(canales["G"], mascara, **PARAMS_C3)
    f["n_manchas"] = len(areas)
    f["area_mancha_max"] = int(areas.max()) if len(areas) else 0
    f["frac_manchas"] = manchas.sum() / dentro.sum()
    if manchas.any():
        f["contraste_G_mancha"] = f["G_mediana"] - canales["G"][manchas].mean()
        f["contraste_b_mancha"] = f["b_mediana"] - canales["b"][manchas].mean()
    else:
        f["contraste_G_mancha"] = 0.0
        f["contraste_b_mancha"] = 0.0
    return f


def color_celula(recorte_rgb, mascara, fondo_imagen, fondo_ref):
    """Normaliza el recorte por el fondo de su imagen y calcula las características de color."""
    return caracteristicas_color(normalizar_fondo(recorte_rgb, fondo_imagen, fondo_ref), mascara)
