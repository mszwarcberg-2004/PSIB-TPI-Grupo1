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

La división está en `data/shared/subset.csv`. Las imágenes y los GT están en `data/raw/<paciente>/Img/` y `data/raw/<paciente>/GT/` y **nunca se modifican**.

**Test no se usa para ajustar nada:** se corre una única vez al final, con todo fijo.

## 2. Instalación

Se desarrolló con **Python 3.14**. Desde la carpeta principal del repositorio:

```bash
python3 -m venv .venv
source .venv/bin/activate          # En Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

Dependencias (`requirements.txt`, con las versiones exactas con las que se obtuvieron los resultados): numpy, opencv-python, scikit-image, scipy, Pillow, pandas, scikit-learn, joblib, matplotlib, jupyter e ipykernel.

## 3. Cómo ejecutar

**Todo se ejecuta desde la carpeta principal del repositorio.** Los scripts que están dentro de una carpeta se corren como módulo (`python -m carpeta.archivo`, sin `.py`). Si no, aparece `ModuleNotFoundError: No module named 'common'`.

### Pipeline completo

```bash
python main.py
```

Corre todo sobre las 12 imágenes de **train** (unos 1,5 minutos): segmentación, textura, color, clasificación, parasitemia y evaluación contra el GT. Al final imprime un resumen con todas las métricas.

**Test** se corre una sola vez, al final del trabajo. Hay que poner `EVALUACION_FINAL = True` en `main.py` y ejecutar `main("test")`. Mientras esté en `False`, `main.py` no deja correr test.

### Otros scripts (análisis y comparaciones, solo train)

| Comando | Qué hace | Qué genera |
|---|---|---|
| `python -m common.verificar_dataset` | Verifica imágenes y GT, cuenta células y comprueba la división por paciente. | Solo imprime en la terminal. |
| `python -m common.visualizar_gt` | Dibuja los polígonos del GT sobre cada imagen. | `data/shared/gt_check/` |
| `python -m segmentation.comparar_otsu` | Compara 12 configuraciones de preprocesamiento y Otsu (IoU y Dice). | `segmentation/results/comparacion_otsu.csv` |
| `python -m segmentation.sensibilidad_kernels` | Tabla de sensibilidad de los kernels de apertura, cierre y gaussiano (unos 8 minutos). | `segmentation/results/sensibilidad_kernels.csv` |
| `python -m segmentation.a1_segmentacion_base` | Solo la segmentación (Bloque A) con su evaluación. Permite comparar Otsu, K-means y GMM. | `data/shared/pred/<método>/` |
| `python -m color_stain.normalizacion` | Recalcula el fondo de referencia de train. | `color_stain/results/referencia_fondo.json` |

### Notebooks (análisis con el que se eligieron los parámetros)

Se abren desde su propia carpeta. Todos usan **solo train**.

| Notebook | Contenido |
|---|---|
| `color_stain/c1_analisis_tincion.ipynb` | Tinción y espacios de color: qué canal separa mejor células, fondo, parásitos y glóbulos blancos. |
| `color_stain/c2_normalizacion_tincion.ipynb` | Normalización de color (balance de blancos) y filtros de bordes (Sobel, Prewitt, Laplaciano). |
| `color_stain/c3_mascara_oscura_violeta.ipynb` | Máscara de manchas oscuras (parásitos) y de núcleos de glóbulos blancos. |
| `color_stain/c4_caracteristicas_color.ipynb` | 47 atributos de color por célula y su selección. |
| `color_stain/c5_criterios_color.ipynb` | Criterio de color para decidir si una célula está parasitada (validación por paciente). |
| `color_stain/c6_compresion_jpeg.ipynb` | Efecto de la compresión y descompresión JPEG. |
| `color_stain/c7_celulas_segmentadas.ipynb` | El criterio de color aplicado sobre las células que segmenta el pipeline. |
| `texture_eval/notebooks/b_textura_evaluacion2.ipynb` | Textura GLCM, selección de atributos, umbrales y elección del clasificador final (sección 8). |

## 4. Resultados que genera `main.py`

Los resultados **no se suben a GitHub** (están en el `.gitignore`): se generan al correr `main.py`.

```text
data/shared/
├── pred/
│   ├── otsu/                          ← train
│   │   ├── crops/                     recorte RGB de cada célula (imagen original, fondo negro)
│   │   ├── masks/                     máscara de cada célula
│   │   ├── cells_pred.csv             datos de cada célula detectada
│   │   ├── etiquetas_train.csv        clase real (GT) y clase predicha de cada célula
│   │   └── metricas_train.csv         métricas de cada imagen
│   ├── parasitadas/                   ← train, salida final
│   │   ├── mascaras/                  máscara de eritrocitos parasitados de cada imagen
│   │   └── imagenes/                  cada imagen mostrando solo las células parasitadas
│   └── resultados_test/               ← test: todo lo anterior junto, más la evaluación final
└── features/
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

## 5. Flujo del pipeline

```text
cargar_imagen (RGB original)
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
     evaluar_contra_gt (main.py) ............. IoU/Dice, detección, clasificación y parasitemia contra el GT
```

## 6. Qué contiene cada archivo de código

| Archivo | Contenido |
|---|---|
| `main.py` | Pipeline completo. `procesar_imagen` aplica todo a una imagen **sin usar el GT**; `evaluar_contra_gt` mide contra el GT; `main(split)` recorre las imágenes, guarda los resultados e imprime el resumen. |
| **common/** | |
| `config.py` | Rutas del proyecto, semilla y nombres de las clases. |
| `io_utils.py` | Carga de imágenes, GT, subset (`cargar_subset`) y de cada célula (`cargar_celula`). |
| `generar_gt_utils.py` | Recortes desde los polígonos del GT y mapa de etiquetas del GT (`mapa_gt`). |
| `visualizar_gt.py` | Dibuja los polígonos del GT sobre las imágenes. |
| `verificar_dataset.py` | Verifica archivos, conteos y la división por paciente. |
| **segmentation/** (A) | |
| `preprocess.py` | Campo circular, CLAHE, filtro gaussiano y corrección de iluminación (descartada). |
| `cluster_segmentation.py` | Otsu sobre G, K-means y GMM en Lab, y limpieza morfológica con relleno de huecos. |
| `watershed_split.py` | Separación de células pegadas con Watershed y estimación automática del radio. |
| `etiquetado.py` | Área típica de cada imagen, filtro de área y borde, y extracción de recortes y máscaras. |
| `a1_segmentacion_base.py` | Segmentación de una imagen de punta a punta (`procesar_imagen`), exclusión de glóbulos blancos (`excluir_wbc`) y GT sin las células del borde (`gt_sin_borde`). |
| `comparar_otsu.py` | Comparación de configuraciones de preprocesamiento y Otsu. |
| `sensibilidad_kernels.py` | Tabla de sensibilidad de los kernels de apertura, cierre y gaussiano. |
| **texture_eval/** (B) | |
| `glcm_features.py` | Atributos GLCM de una célula (`glcm_celula`): 5 propiedades × 3 distancias. |
| `thresholds.py` | Ranking de atributos (AUC con Mann-Whitney), selección sin redundancia y umbrales (Youden y máximo F1). |
| `rules_classifier.py` | Reglas y regresión logística para color, textura y su combinación, con validación por paciente. |
| `stats_analysis.py` | Gráficos y estadística: boxplots, correlaciones, curvas ROC e intervalo de confianza del AUC (bootstrap). |
| `clasificacion_pred.py` | Aplica el método elegido (`b_metodo_elegido.json`) a cada célula y hace la evaluación final de test. |
| `seg_metrics.py` | IoU, Dice y emparejamiento célula a célula con el GT (algoritmo húngaro, IoU ≥ 0,5). |
| **color_stain/** (C) | |
| `normalizacion.py` | Normalización de color por el fondo de referencia de train (balance de blancos). |
| `wbc_detection.py` | Detección de núcleos de glóbulos blancos (`detectar_wbc`). |
| `color_features.py` | 47 atributos de color de cada célula y manchas oscuras de C3 (`caracteristicas_celula`). |
| `color_classifier.py` | Criterio de color (`clasificar_color`), parasitemia y máscara de eritrocitos parasitados. |
| `color_candidates.py` | Regiones candidatas a parásito sobre la imagen completa (estrategia alternativa, no se usa en `main.py`). |

**Parámetros elegidos con train** (archivos que `main.py` lee y no modifica):
- `color_stain/results/referencia_fondo.json`: fondo de referencia;
- `rangos_color.json`: rangos de color;
- `c5_criterio_color.json`: criterio de color;
- `texture_eval/results/b_metodo_elegido.json`: método final y umbral de textura.

## 7. Organización del trabajo

- Cada integrante trabajó en la carpeta de su bloque, en ramas propias (`feat/segmentation`, `feat/texture-eval`, `feat/color-stain`), integradas a `main` con merges y pull requests.
- Los archivos de `common/` y los formatos de `data/shared/` son compartidos.
- Antes de trabajar, traer lo último de `main` con `git pull`.

## Pendientes

- Poder correr el pipeline sobre imágenes nuevas sin GT ("casos incógnita" de la cátedra).
- Evaluación final sobre test, una única vez.
- Figuras para el informe.
