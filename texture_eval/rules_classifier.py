"""B4 y B5: reglas y clasificador para identificar eritrocitos parasitados
(textura, color y color + textura). Se ajusta solo con train."""
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from texture_eval.thresholds import umbral_f1, sentido_de
from texture_eval.seg_metrics import resumen


# ---------- Piezas básicas (fit = datos de ajuste, val = datos a predecir) ----------

def regla(fit, val, f):
    """Regla de un atributo: sentido y umbral (F1) calculados solo con 'fit'."""
    s = sentido_de(roc_auc_score(fit["y"], fit[f]))
    t = umbral_f1(s * fit[f].values, fit["y"].values)
    return s * val[f].values >= t


def logistica(fit, val, feats):
    """Regresión logística sobre feats; umbral de probabilidad por F1 sobre 'fit'."""
    mod = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    mod.fit(fit[feats], fit["y"])
    t = umbral_f1(mod.predict_proba(fit[feats])[:, 1], fit["y"].values)
    return mod.predict_proba(val[feats])[:, 1] >= t


def definir_estrategias(sel_color, sel_tex):
    """Diccionario nombre -> función(fit, val) que devuelve las predicciones (bool)."""
    f_col, f_tex = sel_color[0], sel_tex[0]       # mejor atributo de cada tipo
    return {
        f"Color ({f_col})":        lambda fit, val: regla(fit, val, f_col),
        f"Textura ({f_tex})":      lambda fit, val: regla(fit, val, f_tex),
        "Color AND textura":       lambda fit, val: regla(fit, val, f_col) & regla(fit, val, f_tex),
        "Color OR textura":        lambda fit, val: regla(fit, val, f_col) | regla(fit, val, f_tex),
        "Logística color":         lambda fit, val: logistica(fit, val, sel_color),
        "Logística textura":       lambda fit, val: logistica(fit, val, sel_tex),
        "Logística color+textura": lambda fit, val: logistica(fit, val, sel_color + sel_tex),
    }


def evaluar_lopo(m, estrategias):
    """Leave-one-patient-out de todas las estrategias. m debe tener la columna 'y' (1 = parasitada).
    Devuelve (loo, res): conteos por estrategia y paciente, y el resumen agrupado por estrategia."""
    filas = []
    for nombre, fn in estrategias.items():
        for pac in m["patient_id"].unique():
            fit, val = m[m.patient_id != pac], m[m.patient_id == pac]
            pred, real = fn(fit, val), val["y"].values.astype(bool)
            filas.append({"estrategia": nombre, "paciente": pac,
                          "TP": (pred & real).sum(), "FP": (pred & ~real).sum(),
                          "FN": (~pred & real).sum(), "TN": (~pred & ~real).sum()})
    loo = pd.DataFrame(filas)
    loo["F1"] = 2 * loo.TP / (2 * loo.TP + loo.FP + loo.FN).replace(0, np.nan)
    res = resumen(loo, "estrategia").loc[list(estrategias)]
    return loo, res


# ---------- Modelo final (para B6) ----------

def ajustar_final(m, feats):
    """Logística + umbral de F1 ajustados con TODAS las células de train."""
    mod = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    mod.fit(m[feats], m["y"])
    t = umbral_f1(mod.predict_proba(m[feats])[:, 1], m["y"].values)
    return {"modelo": mod, "umbral": t, "features": feats}


def marcar(tabla, M):
    """Set de cell_id que el método M marca como parasitadas."""
    p = M["modelo"].predict_proba(tabla[M["features"]])[:, 1]
    return set(tabla.loc[p >= M["umbral"], "cell_id"])


def guardar_metodos(metodos, ruta):
    joblib.dump(metodos, ruta)       # no se sube a Git (results/ está ignorado)


def cargar_metodos(ruta):
    return joblib.load(ruta)