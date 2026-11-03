# PSIB-TPI-Grupo1

## Descripción del proyecto

**Segmentación de eritrocitos mediante clustering por color y clasificación de células parasitadas por malaria mediante características cromáticas y de textura.**

El objetivo del trabajo es desarrollar un algoritmo de procesamiento de imágenes microscópicas de frotis sanguíneos que permita identificar eritrocitos individuales y clasificarlos como parasitados o no parasitados.

Para ello, se implementan técnicas de preprocesamiento, segmentación mediante Otsu, K-means y GMM, separación de células mediante Watershed y extracción de características de color y textura (GLCM).

Finalmente, se evalúan los resultados comparándolos con las anotaciones Ground Truth (GT) del dataset.

El trabajo está dividido en tres bloques:

| Bloque | Carpeta | Contenido |
|---|---|---|
| A - Segmentación | `segmentation/` | Preprocesamiento, Otsu, K-means, GMM, limpieza morfológica, etiquetado, filtrado, extracción de células y Watershed. |
| B - Textura y evaluación | `texture_eval/` | Características GLCM, selección de atributos, métricas de segmentación (IoU, Dice, matching) y clasificación. |
| C - Color | `color_stain/` | Espacios de color, características cromáticas, detección de glóbulos blancos y máscara final de eritrocitos infectados. |

## 1. Dataset

Se utiliza el dataset **NIH-NLM-ThinBloodSmearsPf**, específicamente el subconjunto **Polygon Set**.

Las imágenes son JPG RGB de 5312 × 2988 píxeles. Cada imagen tiene un archivo GT (`.txt`) con:

- Identificador de cada célula.
- Clasificación: `Parasitized`, `Uninfected` o `White_Blood_Cell`.
- Coordenadas de los polígonos que delimitan sus contornos.

Se seleccionaron **16 imágenes de 8 pacientes** (dos imágenes por paciente). La división se hizo **por paciente**, para que imágenes de un mismo paciente no aparezcan en train y test a la vez (fuga de información):

| Conjunto | Pacientes | Imágenes |
|----------|-----------|----------|
| Train | 6 | 12 |
| Test | 2 | 4 |
| Total | 8 | 16 |

Las imágenes seleccionadas contienen **240 eritrocitos parasitados**, **3.111 no parasitados** y 13 glóbulos blancos, según el GT.

La división está en `data/shared/subset.csv`. **Test no se usa para ajustar nada**: solo se va a evaluar una vez, al final, con todos los parámetros fijos.

## 2. Estructura del repositorio

```text
PSIB-TPI-Grupo1/
│
├── main.py                      # Pipeline completo: integra los tres bloques
├── requirements.txt
├── README.md
├── report.md                    # Decisiones tomadas y resultados (base para el informe)
│
├── common/                      # Funciones compartidas por los tres bloques
│   ├── config.py                # Rutas y parámetros generales (SEED, nombres de clases, rutas_segmentacion)
│   ├── io_utils.py              # Carga de imágenes, GT, subset y células (cargar_celula)
│   ├── generar_gt_utils.py      # GT --> recortes por célula y mapas de etiquetas (mapa_gt)
│   ├── visualizar_gt.py         # Dibuja los polígonos del GT sobre las imágenes (data/shared/gt_check/)
│   └── verificar_dataset.py     # Verifica archivos, conteos y división por paciente
│
├── segmentation/                # Bloque A
│   ├── preprocess.py            # Máscara circular, CLAHE, filtro gaussiano, corrección de iluminación
│   ├── cluster_segmentation.py  # Otsu, K-means, GMM y limpieza morfológica
│   ├── etiquetado.py            # Etiquetado, filtrado por área/borde y extracción de recortes y máscaras
│   ├── watershed_split.py       # Separación de células pegadas mediante Watershed
│   ├── a1_segmentacion_base.py  # Segmentación de punta a punta + evaluación sobre train
│   ├── comparar_otsu.py         # Comparación de configuraciones de Otsu
│   ├── notebooks/
│   └── results/                 # comparacion_otsu.csv
│
├── texture_eval/                # Bloque B
│   ├── glcm_features.py         # Características GLCM de una célula (glcm_celula)
│   ├── seg_metrics.py           # IoU y Dice globales, matching célula a célula (evaluar_imagen)
│   ├── notebooks/               # glcm_gt_provisorio.ipynb: análisis de textura sobre el GT
│   └── results/                 # Ranking de atributos, atributos seleccionados, umbrales
│
├── color_stain/                 # Bloque C
│   ├── c1_analisis_tincion.ipynb       # Análisis de la tinción y de los espacios de color
│   ├── c2_normalizacion_tincion.ipynb  # Normalización de color y filtros de bordes
│   ├── c3_mascara_oscura_violeta.ipynb # Manchas oscuras (parásitos) y núcleos de glóbulos blancos
│   ├── c4_caracteristicas_color.ipynb  # Características de color por célula y selección
│   ├── normalizacion.py         # Balance de blancos con el fondo de referencia de train
│   ├── wbc_detection.py         # Detección de núcleos de glóbulos blancos (detectar_wbc)
│   └── results/                 # referencia_fondo.json, características y máscaras de C1-C4
│
└── data/
    ├── raw/                     # Imágenes originales: <paciente>/Img/*.jpg y <paciente>/GT/*.txt
    └── shared/
        ├── subset.csv           # Imágenes seleccionadas y split train/test
        ├── gt_check/            # Imágenes con los polígonos del GT dibujados
        ├── pred/<método>/       # Resultados de la segmentación (NO se sube a GitHub)
        │   ├── crops/           # Recorte RGB de cada célula (fondo negro)
        │   ├── masks/           # Máscara individual de cada célula
        │   ├── cells_pred.csv   # Una fila por célula detectada
        │   └── etiquetas_train.csv  # Clase del GT de cada célula detectada de train (la genera main.py)
        └── features/            # CSV de características (los genera main.py)
```

Las imágenes originales de `data/raw/` **nunca se modifican**: todo el procesamiento trabaja sobre copias.

> Nota: los pacientes 208 y 209 tienen la carpeta del GT escrita `Gt` (no `GT`). `obtener_rutas()` contempla los dos nombres, así que funciona también en Linux o Colab.

## 3. Configuración del entorno

Se desarrolló con **Python 3.14**. Desde la carpeta principal del repositorio:

```bash
python3 -m venv .venv
source .venv/bin/activate          # En Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

El entorno virtual `.venv/` no se sube a GitHub; cada integrante lo crea localmente.

## 4. Cómo ejecutar

**Todos los scripts se ejecutan desde la carpeta principal del repositorio**, y los que están dentro de una carpeta se corren como módulo (`python -m carpeta.archivo`, sin `.py`). Si se ejecutan de otra forma aparece `ModuleNotFoundError: No module named 'common'`.

| Comando | Qué hace | Qué genera |
|---|---|---|
| `python -m common.verificar_dataset` | Verifica que estén todas las imágenes y GT, cuenta células y comprueba que ningún paciente esté en train y test. | Solo imprime en la terminal. |
| `python -m common.visualizar_gt` | Dibuja los polígonos del GT sobre cada imagen (reducida al 25%). | `data/shared/gt_check/*.png` |
| `python -m segmentation.comparar_otsu` | Compara 12 configuraciones de Otsu sobre train con IoU y Dice. | `segmentation/results/comparacion_otsu.csv` |
| `python -m segmentation.a1_segmentacion_base` | Segmenta las imágenes de train (incluye Watershed), extrae las células y evalúa: IoU/Dice global y detección célula a célula (TP, FP, FN, precisión, sensibilidad, F1). | `data/shared/pred/<método>/` |
| `python main.py` | Pipeline completo sobre train: segmentación con Watershed (Bloque A) + textura GLCM de cada célula (Bloque B). | `data/shared/pred/<método>/` y `data/shared/features/features_texture_<método>.csv` |

- El método de segmentación se elige con `METODO` en `main.py` y con el segundo argumento de `segmentacion_base(...)` al final de `a1_segmentacion_base.py` (`"otsu"`, `"kmeans"` o `"gmm"`).
- Watershed se activa o desactiva con `APLICAR_WATERSHED` al principio de `a1_segmentacion_base.py` (por defecto `True`). Con `False` se usa el etiquetado simple, útil para comparar.
- `a1_segmentacion_base.py` y `main.py` **borran los resultados anteriores** de `data/shared/pred/<método>/` antes de volver a generarlos.
- Tiempos aproximados sobre las 12 imágenes de train: `a1` con Otsu ~1 min, `main.py` ~1 min, `comparar_otsu` ~1,5 min. K-means y GMM tardan algo más.
- El notebook `texture_eval/notebooks/glcm_gt_provisorio.ipynb` se abre desde su propia carpeta (calcula la raíz del repo a partir de ahí).

## 5. Flujo del pipeline y formatos compartidos

```text
subset.csv --> cargar_imagen (RGB original)
                  |
   [A] preprocesar_imagen --> campo circular + copia con CLAHE (canal verde) y gaussiano
       segmentar (otsu | kmeans | gmm) --> máscara binaria
       limpiar_mascara (apertura, cierre, relleno de huecos chicos)
       separar_celulas (Watershed: separa células pegadas y etiqueta cada una)
       excluir_wbc (elimina los objetos que contienen un núcleo de glóbulo blanco de [C] detectar_wbc)
       filtrar_y_extraer (área relativa a la célula típica de la imagen y borde) --> crops/, masks/, cells_pred.csv
                  |
   [B] cargar_celula --> glcm_celula --> features_texture_<método>.csv
       seg_metrics: evaluar_segmentacion (global) y evaluar_imagen (célula a célula, contra el GT)
       con los pares de evaluar_imagen, main.py guarda la clase del GT de cada célula detectada --> etiquetas_train.csv
                  |
   [C] detectar_wbc (núcleos violetas y oscuros, usado por [A])
       (pendiente) características de color, clasificación, parasitemia y máscara final de infectados
```

Formatos que comparten los tres bloques:

- **`cell_id`**: `{paciente}_{imagen sin extensión}_{n con 4 dígitos}`, por ejemplo `208C67P28N_ThinF_IMG_20150819_115417_0007`. Lo genera `construir_cell_id()` y se usa para nombrar recortes, máscaras y filas de los CSV. Sirve para unir las características de textura y de color de una misma célula.
- **Recorte (`crops/{cell_id}.png`)**: recorte RGB de la imagen **original** (sin CLAHE) dentro del bounding box, con el fondo en negro.
- **Máscara (`masks/{cell_id}.png`)**: misma forma que el recorte, 255 = célula, 0 = fondo.
- **`cells_pred.csv`**: `cell_id, image_id, patient_id, split, bbox_x, bbox_y, bbox_w, bbox_h, area, circularidad`.
- **`etiquetas_train.csv`**: `cell_id, image_id, patient_id, clase_gt, iou_gt`. `clase_gt` es `Parasitized` o `Uninfected` si la célula detectada tiene pareja en el GT (IoU ≥ 0,5), o `sin_pareja` si no. Sirve para elegir los umbrales del clasificador, solo con train.
- **`mascara_wbc`** (la entrega `detectar_wbc` del Bloque C): matriz booleana del tamaño de la imagen, con `True` en los núcleos de los glóbulos blancos. Se pasa a `procesar_imagen()`, que después de Watershed elimina los objetos enteros que contienen un núcleo.

## 6. Organización del trabajo

Cada integrante desarrolla sus funciones principalmente dentro de la carpeta de su bloque.

Los archivos de `common/` y los formatos de `data/shared/` son compartidos: cualquier cambio que afecte a otros bloques se comunica al grupo.

Se trabaja con ramas individuales (`feat/...`) y los cambios se integran a `main` mediante Git. Antes de empezar a trabajar, traer lo último de `main` (`git pull origin main`).

Las decisiones y resultados de cada etapa se registran en `report.md`.

---

## Pendientes

**Bloque A**
- Inspección visual de la segmentación (contornos de TP, FP y FN sobre algunas imágenes de train) para el informe.
- Unificar el recorrido de imágenes de `main.py` y `a1_segmentacion_base.py`.

**Bloque B**
- Recalcular las características GLCM sobre las células segmentadas (no solo sobre los polígonos del GT) y volver a elegir los umbrales.
- Implementar la clasificación de células infectadas e integrarla en `main.py`.

**Bloque C**
- Características cromáticas por célula (pasar `caracteristicas_color` de C4 a un `.py` y conectarla en `main.py`).
- Clasificación, parasitemia y construcción de la máscara final de eritrocitos infectados.
- Experimento de compresión y descompresión (obligatorio por la consigna).

**General**
- Evaluación final sobre test, una única vez, con todos los parámetros fijos.
- Fijar las versiones exactas de las librerías en `requirements.txt`.
- Borrar archivos viejos que quedaron en la carpeta local `data/shared/pred/` (`cells_pred.csv` suelto y `otsu/metricas_otsu_train.csv`).
