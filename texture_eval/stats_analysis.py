"""Gráficos y comparaciones de B2, B4, B5 y B6."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

from texture_eval.thresholds import POS, NEG, sentido_de, auc_mw


def boxplots_top(tabla, ranking, n=6):
    """Boxplots por clase de los n mejores atributos."""
    top = ranking["feature"].head(n).tolist()
    auc = ranking.set_index("feature")["auc"]
    fig, axs = plt.subplots(2, (n + 1) // 2, figsize=(4 * ((n + 1) // 2), 7))
    for ax, f in zip(axs.ravel(), top):
        datos = [tabla.loc[tabla.label == c, f].dropna() for c in (POS, NEG)]
        ax.boxplot(datos, tick_labels=[POS, NEG], showfliers=False)
        ax.set_title(f"{f}\nAUC={auc[f]:.2f}")
    plt.tight_layout()
    return fig


def heatmap_correlacion(corr):
    """Mapa de calor de correlaciones: bloques intensos = atributos redundantes."""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.index, fontsize=7)
    fig.colorbar(im, ax=ax)
    return fig


def curvas_roc(tabla, ranking, seleccion):
    """Curvas ROC de los atributos seleccionados."""
    auc = ranking.set_index("feature")["auc"]
    y = (tabla.label == POS).astype(int).values
    fig, ax = plt.subplots(figsize=(6, 6))
    for f in seleccion:
        fpr, tpr, _ = roc_curve(y, sentido_de(auc[f]) * tabla[f].values)
        ax.plot(fpr, tpr, label=f"{f} (AUC={max(auc[f], 1 - auc[f]):.2f})")
    ax.plot([0, 1], [0, 1], "k--", label="azar")
    ax.set_xlabel("FPR (1 - especificidad)"); ax.set_ylabel("TPR (sensibilidad)")
    ax.legend(fontsize=8)
    return fig


def ic_auc_bootstrap(tabla, f, n=1000, seed=42):
    """Intervalo de confianza 95 % del AUC remuestreando PACIENTES (no células)."""
    rng = np.random.default_rng(seed)
    grupos = {p: g for p, g in tabla.groupby("patient_id")}
    pacientes = list(grupos)
    aucs = []
    for _ in range(n):
        g = pd.concat([grupos[p] for p in rng.choice(pacientes, len(pacientes), replace=True)])
        pos, neg = g.loc[g.label == POS, f], g.loc[g.label == NEG, f]
        if len(pos) and len(neg):
            aucs.append(auc_mw(pos, neg)[0])
    return np.percentile(aucs, [2.5, 97.5])


def barras_estrategias(res, columnas=("precision", "sensibilidad", "F1")):
    """Barras comparando precisión, sensibilidad y F1 de cada estrategia o método."""
    ax = res[list(columnas)].plot.bar(figsize=(9, 4), rot=30)
    ax.set_ylim(0, 1); ax.set_ylabel("valor"); ax.legend(loc="lower right")
    plt.tight_layout()
    return ax.figure