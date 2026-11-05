# Función del archivo: aplicar el pipeline completo a imágenes nuevas, SIN GT (para los casos incógnita de la cátedra).
# Ejecutar desde la carpeta principal del repo:
#     python predecir.py ruta/a/imagen.jpg                 (una imagen)
#     python predecir.py ruta/a/carpeta/                   (todas las imágenes de una carpeta)
#     python predecir.py ruta/a/carpeta/ --salida otra/carpeta

#Usa procesar_imagen de main.py con los mismos parámetros fijos (elegidos con train). No usa el GT ni subset.csv: solo necesita la imagen.
#Los resultados quedan aparte, en data/shared/pred/incognita/ (una carpeta por imagen y un resumen general).

import argparse
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from common.config import PRED_DIR
from segmentation.a1_segmentacion_base import limpiar_resultados_anteriores
from texture_eval import clasificacion_pred as clf
import main

EXTENSIONES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
CARPETA_SALIDA = PRED_DIR / "incognita"
PACIENTE = "incognita"  #no conocemos el paciente: se usa este nombre para armar el cell_id de cada célula


#Arma la lista de imágenes a procesar: acepta archivos sueltos o carpetas (toma todas las imágenes de adentro)
def listar_imagenes(rutas):
    imagenes = []
    for ruta in map(Path, rutas):
        if not ruta.exists():
            print(f"No existe: {ruta}")
        elif ruta.is_dir():
            imagenes += sorted(p for p in ruta.iterdir() if p.suffix.lower() in EXTENSIONES)
        elif ruta.suffix.lower() in EXTENSIONES:
            imagenes.append(ruta)
        else:
            print(f"Se saltea (no es una imagen ni una carpeta): {ruta}")
    return imagenes


ROJO, VERDE = (255, 0, 0), (0, 200, 0)  #colores de los contornos, en RGB
GROSOR_CONTORNO = 8                     #grosor del contorno, en píxeles de la imagen original


#Dibuja el contorno de CADA célula detectada, una por una: rojo si se clasificó como parasitada, verde si no.
#celulas: tabla con cell_id, bbox_x, bbox_y y la columna "parasitada" (True/False).
#carpeta_masks: carpeta con la máscara de cada célula (masks/{cell_id}.png), que se ubica en la imagen con su bbox.
#Como cada célula usa su propia máscara, dos células pegadas quedan con dos contornos separados.
#La usan predecir.py (contornos.jpg) y figuras.py (última columna de la figura). Devuelve la imagen en RGB.
def dibujar_contornos(imagen_rgb, celulas, carpeta_masks):
    salida = imagen_rgb.copy()
    for _, c in celulas.iterrows():
        mascara = cv2.imread(str(Path(carpeta_masks) / f"{c['cell_id']}.png"), cv2.IMREAD_GRAYSCALE)
        if mascara is None:
            continue
        contornos, _ = cv2.findContours((mascara > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color = ROJO if c["parasitada"] else VERDE
        cv2.drawContours(salida, contornos, -1, color, GROSOR_CONTORNO, offset=(int(c["bbox_x"]), int(c["bbox_y"])))
    return salida


def predecir(rutas_entrada, carpeta_salida=CARPETA_SALIDA):

    imagenes = listar_imagenes(rutas_entrada)
    if not imagenes:
        raise ValueError("No se encontró ninguna imagen para procesar.")

    decision = clf.cargar_decision()  #método de clasificación elegido con train (no se reajusta)
    print(f"Imágenes a procesar: {len(imagenes)} | método: {decision['metodo']} | resultados en: {carpeta_salida}")

    resumen = []
    for ruta in imagenes:
        imagen_bgr = cv2.imread(str(ruta), cv2.IMREAD_COLOR)
        if imagen_bgr is None:
            print(f"No se pudo leer: {ruta}")
            continue
        imagen_rgb = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB)  #paso a rgb

        # Carpeta propia de esta imagen (si ya existía, se borran solo sus recortes y máscaras anteriores)
        carpeta = Path(carpeta_salida) / ruta.stem
        rutas = {"crops": carpeta / "crops", "masks": carpeta / "masks", "csv": carpeta / "celulas.csv"}
        limpiar_resultados_anteriores(rutas)

        # Pipeline completo (el mismo de main.py), sin GT
        resultado = main.procesar_imagen(imagen_rgb, PACIENTE, ruta.name, "incognita", rutas, decision)

        # Una fila por célula: datos de la célula y decisión (color solo y método final)
        celulas = pd.DataFrame(resultado["celulas"])
        if len(celulas):
            celulas["pred_color"] = celulas["cell_id"].map(resultado["pred_color"])
            celulas["parasitada"] = celulas["cell_id"].map(resultado["pred_final"])
        celulas.to_csv(rutas["csv"], index=False)

        # Imágenes de salida
        mascara_par = resultado["mascara_infectados"]
        cv2.imwrite(str(carpeta / "mascara_parasitadas.png"), mascara_par.astype(np.uint8) * 255)
        cv2.imwrite(str(carpeta / "imagen_parasitadas.png"),
                    cv2.cvtColor(imagen_rgb * mascara_par[..., None].astype(np.uint8), cv2.COLOR_RGB2BGR))
        contornos = dibujar_contornos(imagen_rgb, celulas, rutas["masks"]) if len(celulas) else imagen_rgb
        cv2.imwrite(str(carpeta / "contornos.jpg"), cv2.cvtColor(contornos, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])

        n_par = int(celulas["parasitada"].sum()) if len(celulas) else 0
        resumen.append({"imagen": ruta.name, "eritrocitos": len(celulas), "parasitados": n_par,
                        "parasitemia_%": round(resultado["parasitemia"], 2)})
        print(f"{ruta.name}: {len(celulas)} eritrocitos | {n_par} parasitados | parasitemia {resultado['parasitemia']:.2f}%")

    if not resumen:  #no se pudo leer ninguna imagen: no se escribe un resumen vacío
        raise ValueError("No se pudo procesar ninguna imagen.")

    tabla = pd.DataFrame(resumen)
    Path(carpeta_salida).mkdir(parents=True, exist_ok=True)
    tabla.to_csv(Path(carpeta_salida) / "resumen.csv", index=False)
    print(f"\nResumen -> {Path(carpeta_salida) / 'resumen.csv'}")
    return tabla


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aplica el pipeline a imágenes nuevas, sin GT.")
    parser.add_argument("entradas", nargs="+", help="imágenes o carpetas con imágenes")
    parser.add_argument("--salida", default=str(CARPETA_SALIDA), help="carpeta de resultados")
    args = parser.parse_args()
    predecir(args.entradas, args.salida)
