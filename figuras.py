# Función del archivo: figuras de resultados, con una fila por imagen y 4 columnas:
#   original | máscara de parasitadas | solo las células parasitadas | contornos (rojo = parasitada, verde = sana)
# Se arma UNA FIGURA POR PACIENTE (sus 2 imágenes). Al cerrar la ventana aparece la del paciente siguiente.

# Ejecutar desde la carpeta principal del repo:
#     python figuras.py train                         (los 6 pacientes de train)
#     python figuras.py test                          (los 2 pacientes de test, DESPUÉS de correr main("test"))
#     python figuras.py incognita ruta/imagen.jpg     (imágenes ya procesadas con predecir.py; también acepta una carpeta)
#                                                     (como no hay paciente, se agrupan de a 2 imágenes)

#Cada figura se guarda como PNG en la carpeta de resultados de cada modo y además se muestra en pantalla.

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

ESCALA = 0.2              # las imágenes se achican al 20 % para la figura (son de 5312 × 2988)
IMAGENES_POR_FIGURA = 2   # en incógnita no hay paciente: se agrupan de a 2 imágenes


#Arma la lista de imágenes a mostrar, con todo lo necesario para cada fila (sale de los resultados que guardó cada script).
#"grupo" es el paciente: las filas del mismo grupo van en la misma figura.
#La imagen original no se carga acá sino al dibujar cada figura ("cargar"), así no quedan todas en memoria a la vez.
def filas_train_test(split):
    if split == "train":
        base, carpeta_par = PRED_DIR / "otsu", PRED_DIR / "parasitadas"
    else:
        base = PRED_DIR / "resultados_test"
        carpeta_par = base / "parasitadas"
    registros = cargar_subset(split)

    if not (base / f"etiquetas_{split}.csv").exists():
        comando = "python main.py" if split == "train" else 'main("test") (una única vez, ver README)'
        raise FileNotFoundError(f"No están los resultados de {split}. Primero hay que correr {comando}.")

    celulas = pd.read_csv(base / "cells_pred.csv").merge(
        pd.read_csv(base / f"etiquetas_{split}.csv")[["cell_id", "pred_final"]], on="cell_id")
    celulas = celulas.rename(columns={"pred_final": "parasitada"})
    metricas = pd.read_csv(base / f"metricas_{split}.csv").set_index("image_id")

    filas = []
    for r in registros:
        nombre = Path(r["image_id"]).stem + ".png"
        m = metricas.loc[r["image_id"]]
        filas.append({
            "grupo": r["patient_id"],
            "titulo": f"{Path(r['image_id']).stem}\nparasitemia {m['parasitemia_pred']:.1f}% (GT {m['parasitemia_gt']:.1f}%)",
            "cargar": lambda r=r: cargar_imagen(r["patient_id"], r["image_id"]),
            "mascara": carpeta_par / "mascaras" / nombre,
            "enmascarada": carpeta_par / "imagenes" / nombre,
            "celulas": celulas[celulas["image_id"] == r["image_id"]],
            "masks": base / "masks",
        })
    return filas, carpeta_par


def filas_incognita(rutas):
    filas = []
    for ruta in listar_imagenes(rutas):
        carpeta = CARPETA_INCOGNITA / ruta.stem
        if not (carpeta / "celulas.csv").exists():
            print(f"Se saltea {ruta.name}: primero hay que correr python predecir.py {ruta}")
            continue
        celulas = pd.read_csv(carpeta / "celulas.csv")
        parasitemia = 100 * celulas["parasitada"].mean() if len(celulas) else 0.0
        filas.append({
            "grupo": f"imagenes_{len(filas) // IMAGENES_POR_FIGURA + 1}",  #de a 2 imágenes por figura
            "titulo": f"{ruta.stem}\nparasitemia {parasitemia:.1f}%",
            "cargar": lambda ruta=ruta: cv2.cvtColor(cv2.imread(str(ruta), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB),
            "mascara": carpeta / "mascara_parasitadas.png",
            "enmascarada": carpeta / "imagen_parasitadas.png",
            "celulas": celulas,
            "masks": carpeta / "masks",
        })
    return filas, CARPETA_INCOGNITA


def achicar(imagen):
    return cv2.resize(imagen, None, fx=ESCALA, fy=ESCALA, interpolation=cv2.INTER_AREA)


#Arma UNA figura con las filas que recibe (las imágenes de un paciente), la guarda y, si se pide, la muestra.
#plt.show() espera a que se cierre la ventana: así, al cerrarla, aparece la figura siguiente.
def armar_figura(filas, salida, titulo="", mostrar=True):
    if not filas:
        raise ValueError("No hay imágenes para mostrar.")

    columnas = ["Original", "Máscara de parasitadas", "Solo células parasitadas", "Contornos\n(rojo = parasitada, verde = sana)"]

    # Tamaño: cada panel es cuadrado y la figura entera entra en una pantalla (como mucho 8 pulgadas de alto)
    alto_fila = min(3.0, 8 / len(filas))
    fig, ejes = plt.subplots(len(filas), 4, figsize=(4 * alto_fila + 0.6, alto_fila * len(filas) + 0.8), squeeze=False)
    if titulo:
        fig.suptitle(titulo, fontsize=11)

    for i, f in enumerate(filas):
        for clave in ("mascara", "enmascarada"):
            if not Path(f[clave]).exists():
                raise FileNotFoundError(f"Falta {f[clave]}: hay que volver a correr el script que genera estos resultados.")
        original = f["cargar"]()
        mascara = cv2.imread(str(f["mascara"]), cv2.IMREAD_GRAYSCALE)
        enmascarada = cv2.cvtColor(cv2.imread(str(f["enmascarada"]), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        contornos = dibujar_contornos(original, f["celulas"], f["masks"])

        # Recortamos los 4 paneles al cuadrado que contiene el círculo del microscopio, así las células se ven más grandes
        filas_c, cols_c = np.where(mascara_circular(original))
        r0, r1, c0, c1 = filas_c.min(), filas_c.max() + 1, cols_c.min(), cols_c.max() + 1
        paneles = [achicar(p[r0:r1, c0:c1]) for p in (original, mascara, enmascarada, contornos)]

        for j, (eje, panel) in enumerate(zip(ejes[i], paneles)):
            eje.imshow(panel, cmap="gray" if panel.ndim == 2 else None)
            eje.set_xticks([])
            eje.set_yticks([])
            if i == 0:
                eje.set_title(columnas[j], fontsize=9)
        ejes[i][0].set_ylabel(f["titulo"], fontsize=8)  #nombre de la imagen y parasitemia, en vertical al costado de la fila

    plt.tight_layout(pad=0.4, w_pad=0.3, h_pad=0.6)
    Path(salida).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(salida, dpi=250, bbox_inches="tight")  #más resolución en el archivo, porque la figura en pantalla es más chica
    print(f"Figura guardada en: {salida}")
    if mostrar:
        plt.show()
    plt.close(fig)


#Agrupa las filas por paciente (o de a 2 en incógnita) y arma una figura por grupo, una detrás de otra
def figuras_por_grupo(filas, carpeta, modo, mostrar=True):
    grupos = list(dict.fromkeys(f["grupo"] for f in filas))  #grupos en el orden en que aparecen
    for n, grupo in enumerate(grupos, start=1):
        filas_grupo = [f for f in filas if f["grupo"] == grupo]
        titulo = f"{modo} - {grupo} ({n} de {len(grupos)})"
        armar_figura(filas_grupo, Path(carpeta) / f"figura_{modo}_{grupo}.png", titulo, mostrar)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Figuras de resultados (original, máscara, parasitadas y contornos), una por paciente.")
    parser.add_argument("modo", choices=["train", "test", "incognita"])
    parser.add_argument("imagenes", nargs="*", help="solo para incognita: imágenes o carpetas ya procesadas con predecir.py")
    parser.add_argument("--no-mostrar", action="store_true", help="solo guarda las figuras, sin abrir las ventanas")
    args = parser.parse_args()

    if args.modo == "incognita":
        if not args.imagenes:
            parser.error("en modo incognita hay que indicar las imágenes (las mismas que se pasaron a predecir.py)")
        filas, carpeta = filas_incognita(args.imagenes)
    else:
        filas, carpeta = filas_train_test(args.modo)
    figuras_por_grupo(filas, carpeta, args.modo, mostrar=not args.no_mostrar)
