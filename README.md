# PSIB-TPI-Grupo1

**Detección de eritrocitos parasitados por malaria en frotis sanguíneos: segmentación de eritrocitos, características de textura y color, clasificación y cálculo de la parasitemia.**

El algoritmo recibe una imagen de microscopio de un frotis fino teñido con Giemsa y:
1. segmenta todos los eritrocitos (Otsu, limpieza morfológica y Watershed), excluyendo los glóbulos blancos;
2. calcula atributos de **textura** (GLCM) y de **color** de cada eritrocito;
3. clasifica cada eritrocito como **parasitado o sano**;
4. calcula la **parasitemia** (% de eritrocitos infectados) y la **máscara de eritrocitos parasitados**.

Los resultados se comparan con las anotaciones del Ground Truth (GT) del dataset.

**Criterios y decisiones de cada paso:** [`report.md`](report.md). El registro completo de pruebas está en [`docs/report_detallado.md`](docs/report_detallado.md).

| Bloque | Carpeta | Contenido |
|---|---|---|
| A - Segmentación | `segmentation/` | Preprocesamiento, Otsu (y K-means/GMM para comparar), limpieza morfológica, Watershed, filtrado y extracción de células. |
| B - Textura y evaluación | `texture_eval/` | GLCM, selección de atributos, umbrales, elección del clasificador y métricas de evaluación. |
| C - Color | `color_stain/` | Normalización de color, glóbulos blancos, atributos de color, criterio de color, parasitemia, máscara de infectados y compresión JPEG. |

## 1. Dataset

**NIH-NLM-ThinBloodSmearsPf, Polygon Set.** JPG RGB de 5312 × 2988 píxeles. Cada imagen tiene un archivo GT (`.txt`) con el polígono y la clase de cada célula: `Parasitized`, `Uninfected` o `White_Blood_Cell`.

Se seleccionaron **16 imágenes de 8 pacientes**. La división es **por paciente**, así ningún paciente está en train y test a la vez:

| Conjunto | Pacientes | Imágenes |
|---|---:|---:|
| Train | 6 | 12 |
| Test | 2 | 4 |

La división está en `data/shared/subset.csv`. Las imágenes y los GT están en `data/raw/` y **nunca se modifican**.

**Test no se usa para ajustar nada:** se corre una única vez al final, con todo fijo.

## 2. Instalación

Se desarrolló con **Python 3.14**. Desde la carpeta principal del repositorio:

```bash
python3 -m venv .venv
source .venv/bin/activate          # En Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

`requirements.txt` tiene las versiones exactas con las que se obtuvieron los resultados: numpy, opencv-python, scikit-image, scipy, Pillow, pandas, scikit-learn, joblib, matplotlib, jupyter e ipykernel.

## 3. Cómo ejecutar

**Todo se ejecuta desde la carpeta principal del repositorio.** Los scripts que están dentro de una carpeta se corren como módulo (`python -m carpeta.archivo`, sin `.py`). Si no, aparece `ModuleNotFoundError: No module named 'common'`.

### 3.1. Imágenes nuevas, sin GT (casos incógnita)

```bash
python predecir.py ruta/a/imagen.jpg          # una imagen
python predecir.py ruta/a/carpeta/            # todas las imágenes de una carpeta (.jpg, .png, .tif, .bmp)
python predecir.py ruta/a/carpeta/ --salida otra/carpeta
```

- Aplica a cada imagen **el mismo pipeline de `main.py`**, con los parámetros fijos elegidos con train. **No necesita GT** ni que la imagen esté en `subset.csv`.
- Si se corre sin indicar ninguna imagen, muestra cómo usarlo y no hace nada.
- Resultados en `data/shared/pred/incognita/`:
  - `resumen.csv`: una fila por imagen, con eritrocitos, parasitados y parasitemia %;
  - una carpeta por imagen con:
    - `contornos.jpg`: la imagen con cada célula marcada, **rojo = parasitada**, **verde = sana**;
    - `mascara_parasitadas.png` e `imagen_parasitadas.png`;
    - `celulas.csv`: decisión de cada célula;
    - `crops/` y `masks/`.

### 3.2. Pipeline completo con evaluación (train)

```bash
python main.py
```

Corre todo sobre las 12 imágenes de **train** (unos 1,5 minutos): segmentación, textura, color, clasificación, parasitemia y **evaluación contra el GT**. Al final imprime un resumen con todas las métricas.

**Test** se corre una sola vez, al final del trabajo. Hay que poner `EVALUACION_FINAL = True` en `main.py` y ejecutar `main("test")`. Los resultados de test quedan aparte, en `data/shared/pred/resultados_test/`.

### 3.3. Otros scripts (análisis y comparaciones, solo train)

| Comando | Qué hace |
|---|---|
| `python -m common.verificar_dataset` | Verifica imágenes y GT, cuenta células y comprueba la división por paciente. |
| `python -m common.visualizar_gt` | Dibuja los polígonos del GT sobre cada imagen (`data/shared/gt_check/`). |
| `python -m segmentation.comparar_otsu` | Compara 12 configuraciones de preprocesamiento y Otsu (IoU y Dice). |
| `python -m segmentation.sensibilidad_kernels` | Tabla de sensibilidad de los kernels de apertura, cierre y gaussiano (unos 8 minutos). |
| `python -m segmentation.a1_segmentacion_base` | Solo la segmentación (Bloque A) con su evaluación. Permite comparar Otsu, K-means y GMM. |
| `python -m color_stain.normalizacion` | Recalcula el fondo de referencia de train. |

Los notebooks (tabla de la sección 6) se abren desde su propia carpeta y usan **solo train**.

## 4. Flujo del pipeline

```text
cargar la imagen (RGB original)
 [C] detectar_wbc ............................ núcleos de glóbulos blancos (violeta y oscuro)
 [A] preprocesar_imagen ...................... campo circular + CLAHE en G + gaussiano
     segmentar_otsu_verde .................... máscara binaria
     limpiar_mascara ......................... apertura 5, cierre 7, relleno de huecos ≤ 4.086 px
     separar_celulas ......................... Watershed (distancia mínima 0,7 × radio típico)
     excluir_wbc ............................. saca los objetos con ≥ 10 % de núcleo de glóbulo blanco
     filtrar_y_extraer ....................... área 0,6–2,0 × típica, sin borde --> crops/, masks/
 [B] glcm_celula ............................. textura de cada célula
 [C] caracteristicas_celula .................. color de cada célula (normalizado por el fondo)
 [C] clasificar_color ........................ ¿tiene al menos una mancha oscura?
 [B] decidir (clasificacion_pred) ............ parasitada = mancha Y contrast_d3 ≥ 0,709
 [C] parasitemia, mascara_infectados ......... % de parasitadas y máscara final
     evaluar_contra_gt (main.py) ............. solo en main.py: IoU/Dice, detección, clasificación y parasitemia contra el GT
```

## 5. Resultados que se generan

Los resultados **no se suben a GitHub** (están en el `.gitignore`): se generan al correr los scripts.

```text
data/shared/
├── pred/
│   ├── otsu/                       ← main.py, train
│   │   ├── crops/, masks/          recorte RGB y máscara de cada célula
│   │   ├── cells_pred.csv          datos de cada célula detectada
│   │   ├── etiquetas_train.csv     clase real (GT) y clase predicha de cada célula
│   │   └── metricas_train.csv      métricas de cada imagen
│   ├── parasitadas/                ← main.py, train: salida final
│   │   ├── mascaras/               máscara de eritrocitos parasitados de cada imagen
│   │   └── imagenes/               cada imagen mostrando solo las células parasitadas
│   ├── resultados_test/            ← main("test"): todo lo anterior junto, más la evaluación final
│   └── incognita/                  ← predecir.py: imágenes nuevas sin GT
└── features/                       ← main.py
    ├── features_texture_otsu_train.csv   15 atributos GLCM de cada célula
    └── features_color_otsu_train.csv     47 atributos de color de cada célula
```

**Columnas de los CSV** (se unen por `cell_id`):
- **`cells_pred.csv`:** `cell_id, image_id, patient_id, split, bbox_x, bbox_y, bbox_w, bbox_h, area, circularidad`.
- **`etiquetas_<split>.csv`:**
  - `cell_id, image_id, patient_id`;
  - `clase_gt`: `Parasitized`, `Uninfected` o `sin_pareja` si no coincide con ningún eritrocito del GT;
  - `iou_gt`;
  - `pred_color`: decisión de la regla de color sola;
  - `pred_final`: decisión del método elegido.
- **`metricas_<split>.csv`** (una fila por imagen):
  - `IoU, Dice`;
  - detección: `TP, FP, FN`;
  - clasificación: `TP_cls, FP_cls, FN_cls, TN_cls` y, con la regla de color sola, `TP_c5, FP_c5, FN_c5`;
  - `parasitemia_pred, parasitemia_gt`.
- **`cell_id`:** `{paciente}_{imagen}_{número de 4 dígitos}`, por ejemplo `208C67P28N_ThinF_IMG_20150819_115417_0007`.

## 6. Qué hay en cada carpeta y en cada archivo

### Raíz del repositorio

| Archivo / carpeta | Qué es |
|---|---|
| `main.py` | **Pipeline completo.** `procesar_imagen` aplica todos los pasos a una imagen **sin usar el GT**; `evaluar_contra_gt` mide el resultado contra el GT; `main(split)` recorre las imágenes de train (o de test, al final), guarda los resultados e imprime el resumen. |
| `predecir.py` | **Casos incógnita:** aplica `procesar_imagen` de `main.py` a imágenes nuevas, sin GT, y guarda los resultados aparte. |
| `README.md` | Este archivo: instalación, ejecución y organización del repositorio. |
| `report.md` | **Criterios y decisiones** de cada paso del pipeline, con su justificación y resultados. |
| `requirements.txt` | Librerías y versiones exactas. |
| `.gitignore` | Archivos que no se suben: entorno virtual, cachés y resultados generados (`data/shared/pred/`, `data/shared/features/`). |
| `.vscode/settings.json` | Configuración de VS Code para ejecutar `common.verificar_dataset` desde el editor. |
| `docs/report_detallado.md` | Registro completo, en orden cronológico, de todas las pruebas del Bloque A (versión anterior del report). |

### `common/`: funciones compartidas por los tres bloques

| Archivo | Qué contiene |
|---|---|
| `config.py` | Rutas del proyecto (`ROOT`, `PRED_DIR`, `FEATURES_DIR`…), semilla, nombres de las clases y `rutas_segmentacion`. |
| `io_utils.py` | Carga de imágenes, GT y subset (`cargar_imagen`, `cargar_gt`, `cargar_subset`), nombre de cada célula (`construir_cell_id`) y carga de cada célula (`cargar_celula`). |
| `generar_gt_utils.py` | Recortes de las células desde los polígonos del GT (`celulas_desde_gt`) y mapa de etiquetas del GT (`mapa_gt`). |
| `visualizar_gt.py` | Dibuja los polígonos del GT sobre las imágenes y arma la máscara del GT. |
| `verificar_dataset.py` | Verifica archivos, conteos de células y la división por paciente. |

### `segmentation/`: Bloque A

| Archivo | Qué contiene |
|---|---|
| `preprocess.py` | Campo circular del microscopio, CLAHE, filtro gaussiano y corrección de iluminación (probada y descartada). |
| `cluster_segmentation.py` | Otsu sobre el canal verde, K-means y GMM en Lab, y limpieza morfológica (apertura, cierre, relleno de huecos). |
| `watershed_split.py` | Separación de células pegadas con Watershed y estimación automática del radio típico. |
| `etiquetado.py` | Área típica de cada imagen, filtro de área y borde, y extracción de recortes y máscaras. |
| `a1_segmentacion_base.py` | Segmentación de una imagen de punta a punta (`procesar_imagen`), exclusión de glóbulos blancos (`excluir_wbc`) y GT sin las células del borde (`gt_sin_borde`). Ejecutado solo, evalúa la segmentación en train. |
| `comparar_otsu.py` | Comparación de configuraciones de preprocesamiento y Otsu. |
| `sensibilidad_kernels.py` | Tabla de sensibilidad de los kernels de apertura, cierre y gaussiano. |
| `results/comparacion_otsu.csv` | IoU y Dice de cada configuración y cada imagen (la genera `comparar_otsu.py`). |
| `results/sensibilidad_kernels.csv` | F1, IoU y Dice para cada valor de los kernels (la genera `sensibilidad_kernels.py`). |
| `notebooks/` | Vacía (reservada). |

### `texture_eval/`: Bloque B

| Archivo | Qué contiene |
|---|---|
| `glcm_features.py` | Atributos GLCM de una célula (`glcm_celula`): contraste, homogeneidad, energía, correlación y entropía a 1, 2 y 3 píxeles. |
| `thresholds.py` | Ranking de atributos (AUC con Mann-Whitney), selección sin redundancia y umbrales (Youden y máximo F1). |
| `rules_classifier.py` | Reglas y regresión logística para color, textura y su combinación, con validación dejando un paciente afuera. |
| `stats_analysis.py` | Gráficos y estadística: boxplots, correlaciones, curvas ROC e intervalo de confianza del AUC (bootstrap). |
| `clasificacion_pred.py` | Aplica el método elegido (`b_metodo_elegido.json`) a cada célula (`decidir`) y hace la evaluación final de test (`evaluar_test`). |
| `seg_metrics.py` | IoU y Dice, y emparejamiento célula a célula con el GT (algoritmo húngaro, IoU ≥ 0,5). |
| `notebooks/b_textura_evaluacion2.ipynb` | Análisis de textura: GLCM, ranking y selección, umbrales con ROC, estrategias color + textura y **elección del método final** (sección 8, sobre las células segmentadas). |
| `results/b_metodo_elegido.json` | **Método final y umbral de textura** (`contrast_d3` ≥ 0,709). Lo lee `main.py`. |
| `results/b2_ranking_atributos.csv` / `_seg.csv` | AUC y p-valor de cada atributo de textura, sobre las células del GT / sobre las segmentadas. |
| `results/b2_atributos_seleccionados.csv` / `_seg.csv` | Atributos de textura elegidos (GT / segmentadas). |
| `results/b4_umbrales_textura.csv` | Umbrales de textura con Youden y máximo F1, sobre las células del GT. |
| `results/b5_estrategias_resumen.csv`, `b5_estrategias_por_paciente.csv`, `b5_f1_por_paciente.csv` | Comparación de estrategias color + textura con validación por paciente (sobre las células del GT). |
| `results/b6_total_train.csv`, `b6_por_paciente_train.csv` | Evaluación de las estrategias contra el GT en train. |
| `results/features_texture_gt_provisorio.csv` | Atributos GLCM de las células recortadas con el GT (análisis exploratorio). |

### `color_stain/`: Bloque C

| Archivo | Qué contiene |
|---|---|
| `normalizacion.py` | Normalización de color por el fondo de referencia de train (balance de blancos). |
| `wbc_detection.py` | Detección de núcleos de glóbulos blancos (`detectar_wbc`). |
| `color_features.py` | 47 atributos de color de cada célula y máscara de manchas oscuras de C3 (`caracteristicas_celula`). |
| `color_classifier.py` | Criterio de color "tiene mancha" (`clasificar_color`), parasitemia y máscara de eritrocitos parasitados. |
| `color_candidates.py` | Regiones candidatas a parásito sobre la imagen completa (estrategia alternativa, no se usa en `main.py`). |
| `c1_analisis_tincion.ipynb` | Tinción y espacios de color: qué canal separa mejor células, fondo, parásitos y glóbulos blancos. |
| `c2_normalizacion_tincion.ipynb` | Normalización de color (balance de blancos, Reinhard) y filtros de bordes (Sobel, Prewitt, Laplaciano). |
| `c3_mascara_oscura_violeta.ipynb` | Máscara de manchas oscuras (parásitos) y de núcleos de glóbulos blancos. |
| `c4_caracteristicas_color.ipynb` | Cálculo de los 47 atributos de color por célula, ranking y selección. |
| `c5_criterios_color.ipynb` | Elección del criterio de color con validación por paciente ("tiene mancha" contra reglas con umbrales). |
| `c6_compresion_jpeg.ipynb` | **Compresión y descompresión JPEG:** efecto sobre la imagen, los atributos de color y la clasificación. |
| `c7_celulas_segmentadas.ipynb` | El criterio de color aplicado sobre las células que segmenta el pipeline, parasitemia y máscara final. |
| `results/referencia_fondo.json` | Color de fondo de referencia (promedio de train). Lo usan `normalizacion.py` y el pipeline. |
| `results/rangos_color.json` | Rangos de color (oscuro, violeta, histogramas) calculados con train. Lo usa `color_features.py`. |
| `results/c5_criterio_color.json` | **Criterio de color final** y parámetros de la máscara de manchas. Lo usa `color_classifier.py`. |
| `results/c4_ranking_color.csv` | AUC de cada atributo de color y cuáles se seleccionaron. |
| `results/c5_resultados_lopo.csv` | Resultados de las reglas de color con validación por paciente. |
| `results/c6_resultados_jpeg.csv` | Efecto de cada calidad JPEG (tamaño, PSNR, SSIM, ΔE, F1). |
| `results/c7_clasificacion_segmentadas.csv`, `c7_parasitemia_paciente.csv` | Criterio de color y parasitemia sobre las células segmentadas. |
| `results/features_color_gt.csv` | Atributos de color de las células recortadas con el GT. |
| `results/.gitignore` | Evita subir las imágenes intermedias que genera C3 (`c3_candidatas/`). |
| `notebooks/` | Vacía (reservada). |

### `data/`

| Carpeta / archivo | Qué es |
|---|---|
| `raw/<paciente>/Img/*.jpg` y `raw/<paciente>/GT/*.txt` | Imágenes originales y anotaciones del GT (no se modifican). En los pacientes 208 y 209 la carpeta se llama `Gt`; el código contempla los dos nombres. |
| `shared/subset.csv` | Las 16 imágenes seleccionadas y su división train/test. |
| `shared/gt_check/` | Imágenes con los polígonos del GT dibujados (las genera `visualizar_gt.py`). |
| `shared/pred/` y `shared/features/` | Resultados generados (sección 5); no se suben a GitHub. |

## 7. Organización del trabajo

- Cada integrante trabajó en la carpeta de su bloque, en ramas propias (`feat/segmentation`, `feat/texture-eval`, `feat/color-stain`), integradas a `main` con merges y pull requests.
- Los archivos de `common/` y los formatos de `data/shared/` son compartidos.
- Antes de trabajar, traer lo último de `main` con `git pull`.

## Pendientes

- Evaluación final sobre test, una única vez.
- Figuras para el informe.
