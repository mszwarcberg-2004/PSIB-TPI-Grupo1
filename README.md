# PSIB-TPI-Grupo1

## Descripción del proyecto

**Segmentación de eritrocitos mediante clustering por color y clasificación de células parasitadas por malaria mediante características cromáticas y de textura.**

El objetivo del trabajo es desarrollar un algoritmo de procesamiento de imágenes microscópicas de frotis sanguíneos que permita identificar eritrocitos individuales y clasificarlos como parasitados o no parasitados.

Para ello, se implementarán técnicas de preprocesamiento, segmentación mediante Otsu, K-means y GMM, separación de células mediante Watershed y extracción de características de color y textura (GLCM).

Finalmente, se evaluarán los resultados comparándolos con las anotaciones Ground Truth del dataset.

## 1. Dataset

Se utiliza el dataset **NIH-NLM-ThinBloodSmearsPf**, específicamente el subconjunto **Polygon Set**.

Las imágenes microscópicas están almacenadas en formato JPG y cuentan con archivos Ground Truth (`.txt`) que contienen:

- Identificador de cada célula.
- Clasificación: `Parasitized`, `Uninfected` o `White_Blood_Cell`.
- Coordenadas de los polígonos que delimitan sus contornos.

Para el desarrollo del TP se seleccionaron **16 imágenes de 8 pacientes**, con dos imágenes por paciente.

La división del dataset se realizó por paciente para evitar fuga de información:

| Conjunto | Pacientes | Imágenes |
|----------|-----------|----------|
| Train | 6 | 12 |
| Test | 2 | 4 |
| Total | 8 | 16 |

Las imágenes seleccionadas contienen:

- 240 eritrocitos parasitados.
- 3.111 eritrocitos no parasitados.

Estas cantidades se obtuvieron a partir de las anotaciones GT.

## 2. Estructura del repositorio

```text
PSIB-TPI-Grupo1/
│
├── common/
│   ├── __init__.py
│   ├── config.py
│   ├── io_utils.py
│   └── verificar_dataset.py
│
├── data/
│   ├── raw/
│   ├── selected/
│   │   └── carpetas de pacientes/
│   │       ├── Img/
│   │       └── GT/
│   │
│   └── shared/
│       ├── subset.csv
│       ├── gt/
│       ├── pred/
│       └── features/
│
├── segmentation/
│   ├── __init__.py
│   ├── notebooks/
│   └── results/
│
├── texture_eval/
│   ├── __init__.py
│   ├── notebooks/
│   └── results/
│
├── color_stain/
│   ├── __init__.py
│   ├── notebooks/
│   └── results/
│
├── report/
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

### Carpetas principales

**`common/`**

Contiene las funciones y configuraciones compartidas por los tres bloques. Permite utilizar las mismas imágenes y anotaciones sin duplicar código.

**`data/`**

Contiene las imágenes seleccionadas y los archivos compartidos.

- `raw/`: reservado para datos originales que no se suben al repositorio.
- `selected/`: imágenes seleccionadas, organizadas por paciente. Cada paciente tiene una carpeta `Img/` con las imágenes JPG y una carpeta `GT/` con sus anotaciones.
- `shared/subset.csv`: imágenes seleccionadas y división en `train` y `test`.
- `shared/gt/`: recortes y máscaras generados a partir del Ground Truth.
- `shared/pred/`: recortes y máscaras generados mediante la segmentación propia.
- `shared/features/`: archivos CSV con características de color y textura.

**`segmentation/` - Bloque A**

Preprocesamiento, segmentación de eritrocitos, separación de células y generación de recortes y máscaras.

**`texture_eval/` - Bloque B**

Extracción de características GLCM, evaluación de segmentación, clasificación y análisis de resultados.

**`color_stain/` - Bloque C**

Análisis de tinción y espacios de color, detección de bordes, extracción de características cromáticas y experimentos de compresión.

**`main.py`**

Integración final del pipeline desarrollado por los tres bloques.

## 3. Funciones compartidas (`common/`)

### 3.1. `config.py`

Centraliza las rutas del proyecto y los parámetros generales.

Variables principales:

- `ROOT`: ruta de la carpeta principal del repositorio.
- `DATA_DIR`: carpeta de datos.
- `SELECTED_DIR`: carpeta donde se encuentran las imágenes seleccionadas.
- `SHARED_DIR`: carpeta de archivos compartidos.
- `SUBSET_CSV`: ruta del CSV con la selección de imágenes.
- `GT_DIR`: carpeta de recortes y máscaras obtenidos desde el GT.
- `PRED_DIR`: carpeta de recortes y máscaras obtenidos mediante la segmentación propia.
- `FEATURES_DIR`: carpeta de características extraídas.
- `SEED = 42`: semilla para reproducibilidad.
- `SCALE = 1.0`: procesamiento a resolución original.
- `TEST_SIZE = 0.25`: proporción de pacientes destinada a prueba.

La división actual está definida manualmente en `subset.csv`, por lo que `TEST_SIZE` no se utiliza para generar una nueva partición.

Las rutas se construyen a partir de `ROOT`, permitiendo ejecutar el código en diferentes computadoras sin modificar las rutas absolutas.

### 3.2. `io_utils.py`

Contiene las funciones compartidas para cargar imágenes y anotaciones.

#### `obtener_rutas(patient_id, image_id)`

Recibe el identificador del paciente y el nombre de la imagen.

Devuelve las rutas de:

- La imagen original JPG.
- Su archivo Ground Truth TXT.

También verifica que ambos archivos existan.

#### `cargar_imagen(patient_id, image_id)`

Carga una imagen mediante OpenCV, la convierte de BGR a RGB y devuelve una matriz NumPy con los valores de sus píxeles.

#### `cargar_gt(patient_id, image_id)`

Lee el archivo Ground Truth de una imagen.

Devuelve:

1. Una lista de diccionarios con las anotaciones.
2. Las dimensiones de la imagen indicadas en el GT.

Cada anotación contiene:

```python
{
    "cell_id": cell_id,
    "label": label,
    "polygon": puntos
}
```

- `cell_id`: identificador de la célula dentro del GT.
- `label`: clasificación de la célula.
- `polygon`: matriz NumPy con las coordenadas `(x, y)` de los vértices del contorno.

Los polígonos se utilizarán posteriormente para generar las máscaras individuales de las células.

#### `cargar_subset(split=None)`

Lee `data/shared/subset.csv` y devuelve una lista de diccionarios con la información de las imágenes seleccionadas.

Cada diccionario contiene:

```python
{
    "image_id": "nombre_imagen.jpg",
    "patient_id": "identificador_paciente",
    "split": "train"
}
```

La función puede utilizarse de tres formas:

```python
cargar_subset()          # Todas las imágenes seleccionadas
cargar_subset("train")   # Imágenes de entrenamiento
cargar_subset("test")    # Imágenes de prueba
```

Esta función devuelve la información de los archivos, no las matrices de las imágenes.

### 3.3. `verificar_dataset.py`

Script utilizado para comprobar la correcta carga del dataset.

Realiza las siguientes verificaciones:

- Existencia de cada imagen y su archivo GT.
- Lectura de las anotaciones.
- Conteo de células parasitadas y no parasitadas.
- Cantidad de imágenes y pacientes seleccionados.
- Ausencia de pacientes compartidos entre entrenamiento y prueba.

Resultados de la última ejecución:

```text
RESUMEN DEL DATASET
Imágenes totales: 16
Pacientes de entrenamiento: 6
Pacientes de prueba: 2
Células parasitadas: 240
Células no parasitadas: 3111
División por pacientes: correcta
```

## 4. Configuración del entorno

Se utiliza un entorno virtual de Python para instalar las dependencias del proyecto.

Desde la carpeta principal del repositorio:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

El entorno virtual `.venv/` no se sube a GitHub.

Cada integrante debe crearlo localmente.

## 5. Estado actual del proyecto

### Paso 0 - Configuración inicial compartida

Implementado:

- Estructura inicial del repositorio.
- Selección de imágenes y pacientes.
- División en entrenamiento y prueba mediante `subset.csv`.
- Configuración de rutas y parámetros generales.
- Funciones para cargar imágenes y anotaciones GT.
- Lectura del CSV de imágenes seleccionadas.
- Verificación de archivos y conteo de células.
- Comprobación de que no existen pacientes compartidos entre train y test.

Pendiente:

- Verificación visual de los polígonos GT sobre algunas imágenes.
- Definición final de identificadores únicos y formatos de recortes y máscaras.
- Unificación de imports para la integración del proyecto.

### Próxima etapa

Comenzar el desarrollo de los bloques A, B y C.

El Bloque A generará inicialmente los recortes y máscaras individuales desde las anotaciones GT, permitiendo que los bloques B y C comiencen a trabajar con las características de textura y color.

Posteriormente, se desarrollará la segmentación automática y se evaluará el pipeline completo.

## 6. Organización del trabajo

Cada integrante desarrollará sus funciones principalmente dentro de la carpeta correspondiente a su bloque.

Los archivos de `common/` y los formatos de `data/shared/` son compartidos. Cualquier modificación que afecte a los otros bloques deberá comunicarse al grupo.

Se trabajará con ramas individuales y los cambios se integrarán a `main` mediante Git.

Los resultados generados durante el procesamiento se mantendrán separados de las imágenes originales.

