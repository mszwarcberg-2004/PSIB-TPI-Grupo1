# Decisiones y avances del proyecto

## 1. Selección y organización del dataset

Se seleccionaron **16 imágenes del Polygon Set de NIH-NLM-ThinBloodSmearsPf**, correspondientes a 8 pacientes (2 imágenes por paciente).

**Decisiones y justificación:**
- **Selección de imágenes:** se buscaron imágenes con eritrocitos parasitados y no parasitados, y variedad en la tinción y distribución celular, para evaluar el algoritmo en distintas condiciones.
- **División por paciente:** se asignaron 6 pacientes a entrenamiento (12 imágenes) y 2 a prueba (4 imágenes). Esto evita que imágenes de un mismo paciente aparezcan en ambos conjuntos y reduce el riesgo de fuga de información.
- **Resolución original:** se decidió trabajar inicialmente sin reescalar las imágenes para conservar los detalles celulares, importantes para el análisis posterior de color y textura.
- **Conservación de originales:** todas las operaciones de preprocesamiento y segmentación se realizan sobre copias o variables derivadas. Las imágenes originales no se modifican ni sobrescriben.

El subconjunto contiene **240 eritrocitos parasitados y 3.111 no parasitados**, según el Ground Truth (GT). Este desbalance deberá considerarse especialmente al evaluar la clasificación de células infectadas.

## 2. Uso del Ground Truth y metodología de evaluación

Se implementaron funciones compartidas para cargar las imágenes RGB, leer las anotaciones del GT y seleccionar las imágenes según su conjunto.

También se visualizaron los polígonos del GT sobre las imágenes originales para comprobar visualmente su correspondencia con las células.

**Decisiones y justificación:**
- **GT como referencia:** se utiliza para comparar las máscaras predichas con las anotaciones reales, pero nunca para detectar automáticamente los eritrocitos.
- **Entrenamiento:** se utilizan los GT de las 12 imágenes de train para comparar configuraciones y seleccionar parámetros.
- **Prueba:** las 4 imágenes de test se reservan para evaluar el algoritmo una vez definido, sin ajustar sus parámetros según esos resultados.
- **Métricas:** se utilizan IoU y Dice para medir la superposición entre la máscara segmentada y el GT.

Para esta evaluación se construye una máscara binaria del GT que incluye los polígonos de eritrocitos parasitados y no parasitados, excluyendo los glóbulos blancos.

**Importante:** las métricas obtenidas en train son resultados de desarrollo, no una estimación independiente del rendimiento. Además, IoU y Dice miden superposición de píxeles, no la correcta separación de cada célula individual.

## 3. Exclusión del borde negro del microscopio

Las imágenes presentan un campo de visión circular rodeado por un borde negro, cuya intensidad podría confundirse con estructuras celulares oscuras.

**Decisión:** generar una máscara circular que delimite la región válida antes de aplicar la segmentación.

**Implementación:**
1. Umbralizar la imagen para distinguir el campo de visión del borde negro.
2. Seleccionar la región conectada más grande.
3. Rellenar sus huecos.
4. Aplicar una pequeña erosión para excluir posibles artefactos del borde.

**Justificación:** evitar que los píxeles negros externos influyan en el cálculo de umbrales y generen falsas detecciones.

La máscara circular se utiliza durante el procesamiento sin modificar la imagen RGB original.

## 4. Segmentación de eritrocitos mediante Otsu

Se eligió **Otsu como primer método de segmentación**, para establecer una referencia inicial (*baseline*) que permita comparar posteriormente con K-means y GMM.

### 4.1. Selección del canal verde

**Decisión:** aplicar Otsu al canal verde de la imagen RGB.

**Justificación:** se buscó aprovechar el contraste de intensidad entre los eritrocitos teñidos y el fondo claro. La elección también se consideró a partir de bibliografía sobre segmentación de células sanguíneas teñidas con Giemsa.

Otsu determina automáticamente un umbral que separa dos grupos de intensidades. En nuestro caso, los píxeles más oscuros que el umbral se consideran posibles regiones celulares.

El umbral se calcula utilizando únicamente los píxeles del campo circular.

**Limitación:** esta segmentación detecta posibles eritrocitos, pero todavía no distingue entre parasitados y no parasitados.

### 4.2. Selección del preprocesamiento

Se compararon distintas configuraciones de Otsu sobre las **12 imágenes de entrenamiento**, utilizando IoU y Dice respecto del GT.

Se evaluaron:
- Otsu con y sin filtro gaussiano.
- Apertura y cierre morfológicos.
- Relleno de huecos.
- Corrección de iluminación.
- Mejora de contraste local mediante CLAHE.
- Combinaciones de estas operaciones.

**Decisiones y justificación:**

**Filtro gaussiano:** se incorporó para reducir pequeñas variaciones de intensidad antes de calcular el umbral. Su mejora individual fue pequeña, pero se mantuvo en la configuración seleccionada.

**CLAHE:** se incorporó porque mejora el contraste local y permitió distinguir mejor los eritrocitos del fondo. Fue el preprocesamiento que produjo la mejora más importante en las métricas.

**Corrección de iluminación:** se probó para compensar diferencias de iluminación, pero empeoró las métricas con los parámetros utilizados. Por lo tanto, no se incorporó al pipeline actual.

**Conservación de colores originales:** CLAHE se aplica solamente a una copia del canal verde para realizar la segmentación. Los recortes celulares se extraen de la imagen RGB original, para preservar la información cromática necesaria en las etapas posteriores.

### 4.3. Selección de operaciones morfológicas

Se observó que algunos eritrocitos presentan un centro más claro que su periferia. Como consecuencia, Otsu puede segmentar solamente el contorno y dejar huecos negros en el interior.

**Decisión:** aplicar operaciones morfológicas después de Otsu.

- **Apertura:** elimina pequeñas regiones aisladas de ruido.
- **Cierre:** ayuda a cerrar pequeñas discontinuidades.
- **Relleno de huecos:** recupera regiones interiores que deberían formar parte de los eritrocitos.

El relleno se limita a huecos pequeños para evitar completar regiones excesivamente grandes.

**Resultado:** el relleno de huecos fue la operación que más contribuyó a mejorar la superposición con el GT. La apertura y el cierre aportaron una mejora adicional pequeña.

### 4.4. Resultados y configuración seleccionada

La comparación de las principales configuraciones sobre train produjo:

| Configuración | IoU promedio | Dice promedio |
|---|---:|---:|
| Otsu sin filtro | 0.7615 | 0.8636 |
| Otsu + gaussiano + limpieza completa | 0.7779 | 0.8743 |
| CLAHE + gaussiano + Otsu, sin limpieza | 0.7862 | 0.8796 |
| CLAHE + gaussiano + Otsu + solo relleno | 0.8006 | 0.8888 |
| **CLAHE + gaussiano + Otsu + limpieza completa** | **0.8016** | **0.8894** |

**Configuración seleccionada:**

**Canal verde → CLAHE → filtro gaussiano → Otsu → apertura → cierre → relleno de huecos.**

**Justificación:** fue la configuración con mayor IoU y Dice promedio sobre las imágenes de entrenamiento.

Se conservó la limpieza completa porque presentó el mejor resultado, aunque la diferencia respecto de utilizar solamente relleno de huecos fue mínima.

Esta configuración se considera el **baseline de Otsu**, pendiente de su evaluación independiente sobre test.

## 5. Etiquetado, filtrado y extracción de células

Una vez obtenida la máscara binaria, se implementó el procesamiento de objetos individuales.

### 5.1. Etiquetado de componentes conectadas

Se aplica etiquetado de componentes conectadas con **conectividad 8**, asignando un número diferente a cada región blanca conectada.

**Objetivo:** identificar objetos candidatos a eritrocitos para analizarlos individualmente.

**Limitación:** dos eritrocitos en contacto pueden formar una única región y recibir la misma etiqueta.

### 5.2. Filtrado de objetos

Se utilizan propiedades geométricas de cada región etiquetada.

**Criterios actuales:**
- Área mínima: 800 píxeles.
- Área máxima: 12.000 píxeles.
- Exclusión de objetos que tocan la franja de seguridad próxima al borde circular.

**Justificación:**
- Eliminar pequeñas manchas de ruido.
- Descartar regiones demasiado grandes que podrían corresponder a células agrupadas u otros objetos.
- Evitar recortes de eritrocitos incompletos en los bordes.

Estos umbrales son provisionales y no garantizan que todos los objetos aceptados sean eritrocitos correctamente segmentados.

### 5.3. Extracción de recortes y máscaras

Para cada objeto aceptado se obtiene su *bounding box* y se generan:

- **Crop RGB:** recorte de la imagen original, conservando únicamente los píxeles de la región celular y dejando el fondo negro.
- **Máscara individual:** imagen binaria que representa la región detectada.
- **Registro en `cells_pred.csv`:** identificador, paciente, imagen, conjunto, coordenadas, área y circularidad.

**Decisión:** guardar por separado los resultados de cada método de segmentación, para evitar que futuras ejecuciones sobrescriban resultados anteriores.

La estructura adoptada es:

```text
data/shared/pred/
    otsu/
        crops/
        masks/
        cells_pred.csv
    kmeans/
    gmm/
```

Las rutas se generan mediante la función compartida `rutas_segmentacion(metodo)`.

### 5.4. Resultados y limitaciones observadas

Al ejecutar el pipeline de Otsu sobre las 12 imágenes de entrenamiento se obtuvieron **618 objetos aceptados** después del filtrado.

La inspección visual mostró resultados mixtos:
- Algunos eritrocitos se segmentaron y recortaron correctamente.
- Otros quedaron incompletos o presentaron formas irregulares.
- Algunas regiones pueden agrupar células que están en contacto.

**Decisión:** conservar Otsu como método de referencia y avanzar con otros métodos, en lugar de continuar ajustándolo indefinidamente.

**Justificación:** disponer de un baseline permite comparar objetivamente si K-means y GMM mejoran la segmentación.

El número de objetos aceptados no constituye por sí mismo una métrica de calidad. La evaluación debe complementarse con IoU, Dice e inspección visual.

Aclaración sobre los resultados: el IoU de 0.8016 y el Dice de 0.8894 corresponden a la máscara de Otsu después de CLAHE y limpieza morfológica, pero antes del etiquetado y filtrado de objetos. Por otro lado, los 618 objetos corresponden a las regiones aceptadas después del filtrado por área y borde. Por lo tanto, las métricas evalúan la superposición de la máscara de segmentación con el GT, mientras que el número de objetos indica cuántas regiones fueron aceptadas para la extracción de células.

## 6. Estrategia general del proyecto

Se definieron dos estrategias posibles para obtener finalmente **máscaras de eritrocitos parasitados completos**.

### Estrategia A: segmentación dirigida por color

Identificar regiones cromáticas compatibles con la presencia de parásitos y, a partir de ellas, delimitar los eritrocitos completos que las contienen.

**Justificación:** aprovechar las diferencias de color asociadas a la infección para orientar la detección.

### Estrategia B: segmentación general y posterior clasificación

1. Segmentar todos los eritrocitos mediante Otsu, K-means o GMM.
2. Separar y extraer los objetos individuales.
3. Calcular características de color y textura.
4. Identificar cuáles corresponden a eritrocitos parasitados.

**Justificación:** separar el problema de localizar las células del problema de determinar si están infectadas.

También se contempla incorporar detección y exclusión de glóbulos blancos mediante características cromáticas y geométricas.

Estas etapas todavía no están implementadas en el pipeline actual.

## 7. Evaluación y próximos pasos

**Estado actual:** se completó una primera implementación de Otsu con CLAHE, limpieza morfológica, etiquetado, filtrado y extracción de recortes, utilizando train.

**Próximos pasos:**
1. Implementar la segmentación mediante **K-means**, utilizando características cromáticas en espacios como Lab o HSV.
2. Implementar y evaluar **GMM** como método alternativo de agrupamiento.
3. Comparar los métodos con las mismas métricas y el mismo conjunto de entrenamiento.
4. Evaluar **Watershed** para separar eritrocitos en contacto.
5. Incorporar la detección y exclusión de glóbulos blancos.
6. Desarrollar la identificación de eritrocitos parasitados mediante características de color y textura.
7. Evaluar los métodos definidos sobre el **conjunto de test**, manteniendo sus parámetros fijos.

La evaluación final deberá distinguir entre la calidad de segmentación de píxeles, la detección de células individuales y la identificación de eritrocitos parasitados.

## 8. Bibliografía de referencia

Walliander, M., et al. (2013). *Automated segmentation of blood cells in Giemsa stained digitized thin blood films*. Diagnostic Pathology, 8(Suppl. 1), S37.

https://doi.org/10.1186/1746-1596-8-S1-S37

*Referencia consignada previamente; verificar los datos bibliográficos antes de la entrega final.*