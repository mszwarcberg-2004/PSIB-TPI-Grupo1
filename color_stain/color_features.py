# Función del archivo: características de color de cada eritrocito (Bloque C).
# Las funciones vienen del notebook C4 (características cromáticas por célula), sin cambios en la lógica.
# La usa main.py (paso 7) para calcular el color de cada célula segmentada, igual que glcm_celula para la textura.
#
# Uso en el pipeline (para cada imagen):
#     fondo = mediana_fondo(imagen_rgb, campo)                        # una vez por imagen (de normalizacion.py)
#     atributos = caracteristicas_celula(crop, mask, fondo)           # para cada célula, con el recorte SIN normalizar

import json

import cv2
import numpy as np

from common.config import ROOT, LABEL_UNINFECTED, LABEL_WBC
from color_stain.normalizacion import normalizar_fondo, cargar_referencia

# Archivo con los rangos de color calculados con train (los guarda el notebook C4)
RUTA_RANGOS = ROOT / "color_stain" / "results" / "rangos_color.json"

# Parámetros de la máscara de manchas oscuras por célula, elegidos en C3 (solo con train)
PARAMS_C3 = {"erosion": 5, "umbral": -39, "area_min": 50}

N_BINS_HIST = 8            # cantidad de intervalos de los histogramas por célula
CANALES_HIST = ["G", "b"]  # canales con histograma: G (oscuro) y b de Lab (violeta)
PERCENTIL_RANGO = 1        # "oscuro"/"violeta" = por debajo del percentil 1 de los píxeles de eritrocitos sanos de train


def kernel_eliptico(radio):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radio + 1, 2 * radio + 1))


def canales_color(recorte_rgb):
    # Los canales que usamos: G (RGB), L, a, b (Lab) y S (HSV)
    hsv = cv2.cvtColor(recorte_rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(recorte_rgb, cv2.COLOR_RGB2LAB)
    return {"G": recorte_rgb[:, :, 1], "L": lab[:, :, 0], "a": lab[:, :, 1], "b": lab[:, :, 2], "S": hsv[:, :, 1]}


def manchas_c3(verde, mascara, erosion, umbral, area_min):
    # Máscara de C3: píxeles con G muy por debajo de la mediana de la célula, solo en manchas grandes.
    # Devuelve la máscara final y el área de cada mancha.
    dentro = cv2.erode(mascara, kernel_eliptico(erosion)) > 0 if erosion > 0 else mascara > 0
    if not dentro.any():
        return np.zeros_like(dentro), np.array([], dtype=int)
    relativo = verde.astype(np.int16) - int(np.median(verde[dentro]))
    marcados = ((relativo < umbral) & dentro).astype(np.uint8)
    n, etiquetas, stats, _ = cv2.connectedComponentsWithStats(marcados, connectivity=8)
    grandes = [k for k in range(1, n) if stats[k, cv2.CC_STAT_AREA] >= area_min]
    return np.isin(etiquetas, grandes), stats[grandes, cv2.CC_STAT_AREA]


#Función calcular_rangos: fija, SOLO con células de train, el rango "oscuro"/"violeta" y los intervalos de los histogramas.
def calcular_rangos(celulas):
    # celulas: lista de (label, recorte normalizado, máscara) de train
    pix_todos = {c: [] for c in CANALES_HIST}
    pix_sanos = {c: [] for c in CANALES_HIST}
    for label, recorte, mascara in celulas:
        if label == LABEL_WBC:
            continue
        canales = canales_color(recorte)
        for c in CANALES_HIST:
            valores = canales[c][mascara > 0]
            pix_todos[c].append(valores)
            if label == LABEL_UNINFECTED:
                pix_sanos[c].append(valores)
    pix_todos = {c: np.concatenate(v) for c, v in pix_todos.items()}
    pix_sanos = {c: np.concatenate(v) for c, v in pix_sanos.items()}

    return {
        # Oscuro: G por debajo del percentil 1 de los eritrocitos sanos; violeta: lo mismo con b
        "umbral_oscuro": float(np.percentile(pix_sanos["G"], PERCENTIL_RANGO)),
        "umbral_violeta": float(np.percentile(pix_sanos["b"], PERCENTIL_RANGO)),
        # Histogramas: intervalos iguales entre el percentil 0.5 y el 99.5 de todos los eritrocitos (sin mirar etiquetas)
        "bordes_hist": {c: np.linspace(*np.percentile(pix_todos[c], [0.5, 99.5]), N_BINS_HIST + 1).tolist()
                        for c in CANALES_HIST},
        "calculado_con": "train",
    }


def guardar_rangos(rangos):
    RUTA_RANGOS.parent.mkdir(parents=True, exist_ok=True)
    with open(RUTA_RANGOS, "w") as archivo:
        json.dump(rangos, archivo, indent=2)


def cargar_rangos():
    if not RUTA_RANGOS.exists():
        raise FileNotFoundError(f"No está {RUTA_RANGOS}: hay que ejecutar antes el notebook C4, que lo calcula con train.")
    with open(RUTA_RANGOS) as archivo:
        return json.load(archivo)


#Función principal: recibe el recorte de una célula YA normalizado por fondo y su máscara, y devuelve un diccionario con las características
def caracteristicas_color(recorte_rgb, mascara, rangos=None):
    dentro = mascara > 0
    if not dentro.any():  #máscara vacía (puede pasar con la segmentación automática): devolvemos None, igual que glcm_celula, y main.py la saltea
        return None
    if rangos is None:
        rangos = cargar_rangos()
    canales = canales_color(recorte_rgb)
    valores = {nombre: canal[dentro].astype(float) for nombre, canal in canales.items()}

    f = {"n_pixeles": int(dentro.sum())}

    # a) Estadísticas de color de la célula entera
    for nombre, v in valores.items():
        f[f"{nombre}_media"] = v.mean()
        f[f"{nombre}_desvio"] = v.std()
        f[f"{nombre}_mediana"] = np.median(v)

    # b) Píxeles extremos (absolutos y relativos a la mediana de la célula)
    for nombre in ["G", "L", "b"]:
        f[f"{nombre}_p5"] = np.percentile(valores[nombre], 5)
    for nombre in ["a", "S"]:
        f[f"{nombre}_p95"] = np.percentile(valores[nombre], 95)
    for nombre in ["G", "b"]:
        f[f"{nombre}_rel_p1"] = np.percentile(valores[nombre], 1) - f[f"{nombre}_mediana"]

    # c) Proporción de tonos oscuros/violetas y área cromática integrada
    oscuro = valores["G"] < rangos["umbral_oscuro"]
    violeta = valores["b"] < rangos["umbral_violeta"]
    f["frac_oscura"] = oscuro.mean()
    f["frac_violeta"] = violeta.mean()
    f["area_cromatica"] = int((oscuro & violeta).sum())
    f["frac_cromatica"] = (oscuro & violeta).mean()

    # d) Manchas de la máscara C3
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

    # e) Histogramas normalizados de G y b (los valores fuera del rango van al primer o último intervalo)
    for nombre in CANALES_HIST:
        bordes = np.array(rangos["bordes_hist"][nombre])
        v = np.clip(valores[nombre], bordes[0], bordes[-1])
        hist, _ = np.histogram(v, bins=bordes)
        for k, h in enumerate(hist / hist.sum()):
            f[f"{nombre}_hist_{k}"] = h
    return f


#Atajo para el pipeline: recibe el recorte SIN normalizar y el fondo de su imagen (mediana_fondo), lo normaliza con la referencia de train y calcula todo
def caracteristicas_celula(recorte_rgb, mascara, fondo_imagen, rangos=None):
    recorte_norm = normalizar_fondo(recorte_rgb, fondo_imagen, cargar_referencia())
    return caracteristicas_color(recorte_norm, mascara, rangos)
