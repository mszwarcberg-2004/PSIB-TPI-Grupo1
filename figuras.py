# Función del archivo: figura de resultados, con una fila por imagen y 4 columnas:
#   original | máscara de parasitadas | solo las células parasitadas | contornos (rojo = parasitada, verde = sana)

# Ejecutar desde la carpeta principal del repo:
#     python figuras.py train                         (una imagen por paciente de train)
#     python figuras.py test                          (las 4 imágenes de test, DESPUÉS de correr main("test"))
#     python figuras.py incognita ruta/imagen.jpg     (imágenes ya procesadas con predecir.py; también acepta una carpeta)

#La figura se guarda como PNG en la carpeta de resultados de cada modo y además se muestra en pantalla.

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.config import PRED_DIR
from common.io_utils import cargar_subset, cargar_imagen
from segmentation.preprocess import mascara_circular
from predecir import listar_imagenes, dibujar_contornos, CARPETA_SALIDA as CARPETA_INCOGNITA
#dibujar_contornos (un contorno por célula: rojo = parasitada, verde = sana) está en predecir.py, así no se repite el código

ESCALA = 0.2            # las imágenes se achican al 20 % para la figura (son de 5312 × 2988)


#Arma la lista de imágenes a mostrar, con todo lo necesario para cada fila (sale de los resultados que guardó cada script)
def filas_train_test(split):
    if split == "train":
        base, carpeta_par = PRED_DIR / "otsu", PRED_DIR / "parasitadas"
        registros = pd.DataFrame(cargar_subset("train")).groupby("patient_id").head(1)  #una imagen por paciente
    else:
        base = PRED_DIR / "resultados_test"
        carpeta_par = base / "parasitadas"
        registros = pd.DataFrame(cargar_subset("test"))

    if not (base / f"etiquetas_{split}.csv").exists():
        comando = "python main.py" if split == "train" else 'main("test") (una única vez, ver README)'
        raise FileNotFoundError(f"No están los resultados de {split}. Primero hay que correr {comando}.")

    celulas = pd.read_csv(base / "cells_pred.csv").merge(
        pd.read_csv(base / f"etiquetas_{split}.csv")[["cell_id", "pred_final"]], on="cell_id")
    celulas = celulas.rename(columns={"pred_final": "parasitada"})
    metricas = pd.read_csv(base / f"metricas_{split}.csv").set_index("image_id")

    filas = []
    for _, r in registros.iterrows():
        nombre = Path(r["image_id"]).stem + ".png"
        m = metricas.loc[r["image_id"]]
        filas.append({
            "titulo": f"{r['patient_id']}\n{r['image_id']}\nparasitemia {m['parasitemia_pred']:.1f}% (GT {m['parasitemia_gt']:.1f}%)",
            "original": cargar_imagen(r["patient_id"], r["image_id"]),
            "mascara": carpeta_par / "mascaras" / nombre,
            "enmascarada": carpeta_par / "imagenes" / nombre,
            "celulas": celulas[celulas["image_id"] == r["image_id"]],
            "masks": base / "masks",
        })
    return filas, carpeta_par / f"figura_{split}.png"


def filas_incognita(rutas):
    resumen = pd.read_csv(CARPETA_INCOGNITA / "resumen.csv").set_index("imagen") if (CARPETA_INCOGNITA / "resumen.csv").exists() else None
    filas = []
    for ruta in listar_imagenes(rutas):
        carpeta = CARPETA_INCOGNITA / ruta.stem
        if not (carpeta / "celulas.csv").exists():
            print(f"Se saltea {ruta.name}: primero hay que correr python predecir.py {ruta}")
            continue
        imagen = cv2.cvtColor(cv2.imread(str(ruta), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        celulas = pd.read_csv(carpeta / "celulas.csv")
        parasitemia = 100 * celulas["parasitada"].mean() if len(celulas) else 0.0
        filas.append({
            "titulo": f"{ruta.name}\nparasitemia {parasitemia:.1f}%",
            "original": imagen,
            "mascara": carpeta / "mascara_parasitadas.png",
            "enmascarada": carpeta / "imagen_parasitadas.png",
            "celulas": celulas,
            "masks": carpeta / "masks",
        })
    return filas, CARPETA_INCOGNITA / "figura_incognita.png"


def achicar(imagen):
    return cv2.resize(imagen, None, fx=ESCALA, fy=ESCALA, interpolation=cv2.INTER_AREA)


def armar_figura(filas, salida, mostrar=True):
    if not filas:
        raise ValueError("No hay imágenes para mostrar.")

    columnas = ["Original", "Máscara de parasitadas", "Solo células parasitadas", "Contornos: rojo = parasitada, verde = sana"]
    fig, ejes = plt.subplots(len(filas), 4, figsize=(16, 4.2 * len(filas)), squeeze=False)

    for i, f in enumerate(filas):
        for clave in ("mascara", "enmascarada"):
            if not Path(f[clave]).exists():
                raise FileNotFoundError(f"Falta {f[clave]}: hay que volver a correr el script que genera estos resultados.")
        mascara = cv2.imread(str(f["mascara"]), cv2.IMREAD_GRAYSCALE)
        enmascarada = cv2.cvtColor(cv2.imread(str(f["enmascarada"]), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        contornos = dibujar_contornos(f["original"], f["celulas"], f["masks"])

        # Recortamos los 4 paneles al cuadrado que contiene el círculo del microscopio, así las células se ven más grandes
        filas_c, cols_c = np.where(mascara_circular(f["original"]))
        r0, r1, c0, c1 = filas_c.min(), filas_c.max() + 1, cols_c.min(), cols_c.max() + 1
        paneles = [achicar(p[r0:r1, c0:c1]) for p in (f["original"], mascara, enmascarada, contornos)]

        for j, (eje, panel) in enumerate(zip(ejes[i], paneles)):
            eje.imshow(panel, cmap="gray" if panel.ndim == 2 else None)
            eje.set_xticks([])
            eje.set_yticks([])
            if i == 0:
                eje.set_title(columnas[j], fontsize=11)
        ejes[i][0].set_ylabel(f["titulo"], fontsize=8, rotation=0, ha="right", va="center")

    plt.tight_layout()
    Path(salida).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(salida, dpi=150, bbox_inches="tight")
    print(f"Figura guardada en: {salida}")
    if mostrar:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Figura de resultados (original, máscara, parasitadas y contornos).")
    parser.add_argument("modo", choices=["train", "test", "incognita"])
    parser.add_argument("imagenes", nargs="*", help="solo para incognita: imágenes o carpetas ya procesadas con predecir.py")
    parser.add_argument("--no-mostrar", action="store_true", help="solo guarda la figura, sin abrir la ventana")
    args = parser.parse_args()

    if args.modo == "incognita":
        if not args.imagenes:
            parser.error("en modo incognita hay que indicar las imágenes (las mismas que se pasaron a predecir.py)")
        filas, salida = filas_incognita(args.imagenes)
    else:
        filas, salida = filas_train_test(args.modo)
    armar_figura(filas, salida, mostrar=not args.no_mostrar)
