"""B2 y B4: ranking de atributos, selección y umbrales.
Todas las funciones reciben una tabla que debe contener SOLO células de train."""
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_curve, precision_recall_curve

POS, NEG = "Parasitized", "Uninfected"


def auc_mw(pos, neg):
    """AUC = U / (n1·n2), equivalente al área bajo la ROC, y p-valor de Mann-Whitney."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[~np.isnan(pos)], neg[~np.isnan(neg)]
    U, p = mannwhitneyu(pos, neg, alternative="two-sided")
    return U / (len(pos) * len(neg)), p


def sentido_de(auc):
    """+1 si el atributo es mayor en parasitadas, -1 si es menor."""
    return 1 if auc >= 0.5 else -1


def ranking_atributos(tabla, feat_cols):
    """Ranking de atributos por poder de separación (parasitadas vs no infectadas)."""
    filas = []
    for f in feat_cols:
        auc, p = auc_mw(tabla.loc[tabla.label == POS, f], tabla.loc[tabla.label == NEG, f])
        filas.append({"feature": f, "auc": auc,
                      "separacion": max(auc, 1 - auc),            # 0.5 = nada, 1 = perfecto
                      "sentido": "mayor en parasitadas" if auc > 0.5 else "menor en parasitadas",
                      "p_value": p})
    return (pd.DataFrame(filas).sort_values("separacion", ascending=False)
            .reset_index(drop=True))


def seleccionar_atributos(tabla, ranking, sep_min=0.6, rho_max=0.9):
    """De mejor a peor AUC: corta si separa menos que sep_min y descarta los que son
    casi copia (|Spearman| >= rho_max) de uno ya elegido. Devuelve (seleccion, corr)."""
    corr = tabla[ranking["feature"].tolist()].corr(method="spearman")
    sep = ranking.set_index("feature")["separacion"]
    seleccion = []
    for f in ranking["feature"]:
        if sep[f] < sep_min:
            break
        if all(abs(corr.loc[f, s]) < rho_max for s in seleccion):
            seleccion.append(f)
    return seleccion, corr


def auc_por_paciente(tabla, features, min_n=5):
    """AUC de cada atributo dentro de cada paciente (NaN si hay < min_n células en una clase)."""
    filas = {}
    for pac, g in tabla.groupby("patient_id"):
        pos, neg = g[g.label == POS], g[g.label == NEG]
        ok = len(pos) >= min_n and len(neg) >= min_n
        filas[pac] = {f: (auc_mw(pos[f], neg[f])[0] if ok else np.nan) for f in features}
    return pd.DataFrame(filas).T


# ---------- Umbrales ----------

def umbral_f1(score, y):
    """Umbral (score >= t) que maximiza F1 sobre los datos de ajuste."""
    p, r, t = precision_recall_curve(y, score)
    f1 = 2 * p * r / np.clip(p + r, 1e-12, None)
    return t[f1[:-1].argmax()]                 # f1[:-1] alinea con t


def umbral_youden(score, y):
    """Umbral que maximiza sensibilidad - (1 - especificidad)."""
    fpr, tpr, thr = roc_curve(y, score)
    return thr[(tpr - fpr).argmax()]


def umbral_criterio(score, y, criterio="F1"):
    return umbral_f1(score, y) if criterio == "F1" else umbral_youden(score, y)


def tabla_umbrales(tabla, ranking, seleccion):
    """Umbral óptimo de cada atributo seleccionado según Youden y según F1 (con todo train)."""
    auc = ranking.set_index("feature")["auc"]
    y = (tabla.label == POS).astype(int).values
    P, N = y.sum(), (1 - y).sum()
    filas = []
    for f in seleccion:
        s = sentido_de(auc[f])
        fpr, tpr, thr = roc_curve(y, s * tabla[f].values)
        tp, fp = tpr * P, fpr * N                      # de tasas a conteos, para la precisión
        prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
        f1 = np.divide(2 * prec * tpr, prec + tpr, out=np.zeros_like(tpr), where=(prec + tpr) > 0)
        for nombre, i in [("Youden", (tpr - fpr).argmax()), ("F1", f1.argmax())]:
            filas.append({"feature": f, "criterio": nombre,
                          "regla": f"{f} {'>=' if s == 1 else '<='} {s * thr[i]:.4f}",
                          "umbral": s * thr[i], "sentido": s,    # umbral en unidades originales
                          "sensibilidad": tpr[i], "especificidad": 1 - fpr[i],
                          "precision": prec[i], "F1": f1[i]})
    return pd.DataFrame(filas)


def lopo_umbral(tabla, ranking, seleccion, criterio="F1"):
    """Leave-one-patient-out del umbral de cada atributo: se ajusta con los demás pacientes
    y se prueba en el que quedó afuera. Devuelve TP/FP/FN/TN por atributo y paciente."""
    auc = ranking.set_index("feature")["auc"]
    filas = []
    for f in seleccion:
        s = sentido_de(auc[f])
        for pac in tabla["patient_id"].unique():
            fit, val = tabla[tabla.patient_id != pac], tabla[tabla.patient_id == pac]
            t = umbral_criterio(s * fit[f].values, (fit.label == POS).astype(int).values, criterio)
            pred = s * val[f].values >= t
            real = (val.label == POS).values
            filas.append({"feature": f, "paciente_fuera": pac, "umbral": s * t,
                          "TP": (pred & real).sum(), "FP": (pred & ~real).sum(),
                          "FN": (~pred & real).sum(), "TN": (~pred & ~real).sum()})
    return pd.DataFrame(filas)