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

**Respaldo del análisis de color (Bloque C, notebook C1):** se comparó qué tan bien cada canal de RGB, HSV y Lab separa los píxeles de fondo de los de células (AUC, solo train). El mejor fue **G** (AUC 0,969), prácticamente igual a L de Lab (0,967). Esto confirma la elección del canal verde.

**Normalización de color (Bloque C, notebook C2):** se evaluó normalizar el color de cada imagen (balance de blancos y Reinhard). Para separar células del fondo no mejoró, porque Otsu ya calcula el umbral en cada imagen. Por eso no se incorporó a la segmentación.

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
- Área mínima: 6.000 píxeles.
- Área máxima: 22.000 píxeles.
- Exclusión de objetos que tocan la franja de seguridad próxima al borde circular (3 píxeles).

Inicialmente se usaron 800 y 12.000 píxeles. Estos valores se corrigieron después de medir las áreas reales del GT (ver sección 5.5).

**Justificación:**
- Eliminar pequeñas manchas de ruido.
- Descartar regiones demasiado grandes que podrían corresponder a células agrupadas u otros objetos.
- Evitar recortes de eritrocitos incompletos en los bordes.

Estos umbrales no garantizan que todos los objetos aceptados sean eritrocitos correctamente segmentados.

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

*Nota: estos 618 objetos corresponden al filtro de área original (800 a 12.000 píxeles). Con el filtro corregido se aceptan 1.175 (sección 5.5).*

Aclaración sobre los resultados: el IoU de 0.8016 y el Dice de 0.8894 corresponden a la máscara de Otsu después de CLAHE y limpieza morfológica, pero antes del etiquetado y filtrado de objetos. Por otro lado, los 618 objetos corresponden a las regiones aceptadas después del filtrado por área y borde. Por lo tanto, las métricas evalúan la superposición de la máscara de segmentación con el GT, mientras que el número de objetos indica cuántas regiones fueron aceptadas para la extracción de células.

### 5.5. Corrección del filtro de área

Al comparar el filtro con las áreas de los eritrocitos del GT de **train** se observó que el área máxima de 12.000 píxeles era demasiado baja:

| Área de los eritrocitos del GT (train) | Píxeles |
|---|---:|
| Percentil 1 | 8.175 |
| Percentil 5 | 9.872 |
| Mediana | 13.929 |
| Percentil 95 | 17.832 |
| Percentil 99 | 19.536 |

El 81% de los eritrocitos del GT superaba los 12.000 píxeles, por lo que el filtro descartaba la mayoría de las células individuales y no solo las células pegadas.

**Decisión:** usar un área mínima de **6.000** y una máxima de **22.000** píxeles.

**Justificación:**
- Los límites se apoyan en los percentiles 1 y 99 del GT de train, con margen hacia abajo porque nuestras máscaras resultan algo más chicas que los polígonos del GT.
- Se probaron varias combinaciones sobre train con la evaluación célula a célula (sección 5.6), y esta fue de las que mejor equilibró precisión y sensibilidad:

| Área mín. - máx. | TP | FP | FN | Precisión | Sensibilidad | F1 |
|---|---:|---:|---:|---:|---:|---:|
| 800 - 12.000 (original) | 527 | 91 | 1.780 | 0.853 | 0.228 | 0.360 |
| 800 - 20.000 | 1.110 | 103 | 1.197 | 0.915 | 0.481 | 0.631 |
| 5.000 - 20.000 | 1.110 | 56 | 1.197 | 0.952 | 0.481 | 0.639 |
| **6.000 - 22.000** | **1.119** | **56** | **1.188** | **0.952** | **0.485** | **0.643** |
| 8.000 - 22.000 | 1.099 | 44 | 1.208 | 0.962 | 0.476 | 0.637 |
| 6.000 - 25.000 | 1.138 | 78 | 1.169 | 0.936 | 0.493 | 0.646 |

Se eligió 6.000 - 22.000 en lugar de 6.000 - 25.000 porque, con una mejora mínima de F1, el segundo acepta más falsos positivos (objetos grandes que suelen ser dos células pegadas).

**Resultado:** con Otsu, la cantidad de células aceptadas en train pasó de 618 a **1.175**.

**Limitación:** la mayoría de los eritrocitos que siguen sin detectarse corresponden a células en contacto, que forman un único objeto demasiado grande o con IoU bajo respecto de cada célula del GT. Esto se resolvió después con Watershed (sección 8).

### 5.6. Evaluación célula a célula (matching)

IoU y Dice globales miden superposición de píxeles, pero no indican si cada eritrocito se detectó por separado. Por eso se agregó una evaluación individual.

**Método:**
1. Se construye un mapa de etiquetas del GT (cada eritrocito con un número distinto, sin glóbulos blancos).
2. Se calcula el IoU entre cada célula del GT y cada célula predicha aceptada.
3. Se emparejan las células uno a uno (algoritmo húngaro) y un par se considera **verdadero positivo (TP)** si su IoU es al menos **0,5**.
4. Las células predichas sin pareja son **falsos positivos (FP)** y las del GT sin pareja son **falsos negativos (FN)**.
5. Se calculan precisión, sensibilidad y F1 sumando TP, FP y FN de todas las imágenes (así cada célula pesa lo mismo, independientemente de la imagen).

**Decisiones y justificación:**
- **Umbral de IoU 0,5:** es el criterio habitual en detección de objetos. Además, con IoU mayor a 0,5 cada célula solo puede emparejarse con una única célula de la otra máscara.
- **Exclusión de las células del GT cortadas por el borde:** nuestro filtrado descarta a propósito las células que tocan el borde del campo circular. Si se dejaran en el GT, se contarían como FN células que se decidió no analizar. Se usa la misma franja de 3 píxeles en ambos casos.
- **Solo células aceptadas:** se evalúan las células que pasaron el filtrado, porque son las que se usan en las etapas siguientes.

Resultado con Otsu sobre train:

| TP | FP | FN | Precisión | Sensibilidad | F1 |
|---:|---:|---:|---:|---:|---:|
| 1.119 | 56 | 1.188 | 0.952 | 0.485 | 0.643 |

**Interpretación:** casi todo lo que se acepta es un eritrocito bien segmentado (precisión alta), pero todavía se pierde aproximadamente la mitad de las células, principalmente por contacto entre células.

*Con Watershed (sección 8), el F1 de Otsu pasó a 0,880.*

## 6. Segmentación mediante K-means y GMM

Se implementaron dos métodos de agrupamiento por color, como alternativa a Otsu.

**Implementación:**
1. Se usa la misma imagen preprocesada que para Otsu (CLAHE en el canal verde y filtro gaussiano) y el mismo campo circular.
2. La imagen se convierte al espacio **Lab**, que separa la luminosidad (L) de la información de color (a y b).
3. Se agrupan los píxeles en **K = 3** grupos: fondo, eritrocitos y zonas más oscuras (por ejemplo, parásitos o núcleos).
4. El modelo se ajusta sobre una muestra aleatoria de **50.000 píxeles** del campo (con semilla fija, para que sea reproducible) y después se clasifican todos los píxeles por partes.
5. El grupo con mayor luminosidad (L) se considera fondo; el resto forma la máscara de eritrocitos.
6. Después se aplica la misma limpieza morfológica, etiquetado y filtrado que con Otsu.

**Decisiones y justificación:**
- **Muestra de píxeles:** cada imagen tiene unos 7 millones de píxeles dentro del campo. Ajustar el modelo con una muestra da resultados equivalentes y reduce mucho el tiempo de cálculo.
- **Ajuste por imagen:** el modelo se ajusta en cada imagen, sin usar etiquetas. No hay entrenamiento con el GT ni uso de test.
- **Espacio Lab:** en el análisis de color del Bloque C (notebook C1) se agruparon píxeles de train sin etiquetas en RGB, HSV y Lab, y se midió cuánto coincidían los grupos con las clases reales (ARI). Lab fue el mejor (K-means 0,38; GMM 0,41), y en HSV el tono es circular (el rosa está cerca de 0 y de 179 a la vez), lo que complica agruparlo. Por eso la segmentación por agrupamiento usa **solo Lab**.
- **GMM con covarianza completa:** permite grupos de forma elíptica, a diferencia de K-means, que supone grupos esféricos.
- **Se mantiene CLAHE en el preprocesamiento:** se probó aplicar K-means y GMM sin CLAHE, porque CLAHE modifica solo el canal verde y altera el color. Sin embargo, sin CLAHE los resultados empeoraron en train (IoU de K-means 0,801 → 0,765; GMM 0,785 → 0,683), por lo que se mantuvo.

## 7. Comparación preliminar de los métodos de segmentación

Resultados sobre las 12 imágenes de train, con el mismo preprocesamiento, limpieza y filtrado, **antes de incorporar Watershed** (la comparación con Watershed está en la sección 8.5):

| Método | IoU global | Dice global | Células aceptadas | TP | FP | FN | Precisión | Sensibilidad | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Otsu** | **0.8016** | **0.8894** | 1.175 | 1.119 | 56 | 1.188 | 0.952 | **0.485** | **0.643** |
| K-means | 0.8009 | 0.8868 | 776 | 758 | 18 | 1.549 | 0.977 | 0.329 | 0.492 |
| GMM | 0.7847 | 0.8779 | 638 | 621 | 17 | 1.686 | 0.973 | 0.269 | 0.422 |

**Observaciones:**
- A nivel de píxeles, Otsu y K-means son prácticamente equivalentes; GMM es algo peor.
- A nivel de células individuales, Otsu detecta bastantes más células. K-means y GMM tienen mayor precisión pero aceptan menos objetos, probablemente porque generan más uniones entre células vecinas, que quedan fuera del filtro de área.
- El filtro de área se calibró con Otsu; todavía no se ajustó para K-means y GMM.

**Estado:** comparación preliminar. La elección final del método se hará después de incorporar Watershed, porque la separación de células pegadas puede cambiar el resultado de K-means y GMM.

**Actualización:** con Watershed, Otsu siguió siendo el mejor método (sección 8.5). Se propone mantenerlo como método final (es el que usa `main.py`).

## 8. Separación de células en contacto mediante Watershed

### 8.1. Problema

Después de la limpieza morfológica, los eritrocitos que se tocan quedan unidos en un único objeto blanco. El etiquetado de componentes conectadas les asigna la misma etiqueta, y entonces:
- el objeto resulta demasiado grande y el filtro de área lo descarta, o
- se acepta, pero su IoU con cada célula del GT es bajo y no cuenta como detección.

Sin separar células, la sensibilidad de detección con Otsu era de 0,485: se perdía aproximadamente la mitad de los eritrocitos, principalmente por contacto entre células.

### 8.2. Método

Watershed ("divisoria de aguas") separa objetos unidos interpretando la imagen como un relieve topográfico.

1. **Transformada de distancia:** a cada píxel de la máscara se le asigna su distancia al píxel de fondo más cercano. El valor es máximo en el centro de cada eritrocito y disminuye en la zona angosta donde se tocan dos células.
2. **Marcadores:** se buscan los máximos locales de la distancia. Cada máximo representa el centro de una célula.
3. **Inundación:** se invierte la distancia (los centros pasan a ser "pozos") y se inunda el relieve desde cada marcador. Donde se encuentran las aguas de dos marcadores distintos se traza el límite entre las dos células.
4. El resultado es una imagen de etiquetas con cada célula separada, que reemplaza al etiquetado simple. Después se aplican el mismo filtrado por área y borde y la misma extracción de recortes.

Watershed no agrega ni elimina píxeles de la máscara: solo divide objetos. Por eso el IoU y el Dice globales no cambian.

### 8.3. Parámetro automático: distancia mínima entre centros

Si dos máximos están muy cerca, probablemente pertenecen a la misma célula; si se aceptaran ambos, una célula se cortaría en pedazos (sobresegmentación). Por eso se exige una distancia mínima entre centros.

**Decisión:** calcular esa distancia automáticamente en cada imagen como `0,7 × radio típico de un eritrocito`.

**Estimación del radio (sin GT):** para cada objeto de la máscara, el máximo de la transformada de distancia es aproximadamente el radio de una célula (incluso si el objeto son dos células pegadas). El radio típico de la imagen es la **mediana** de esos valores; se usa la mediana para que manchas pequeñas o grupos grandes no la desplacen. Con Otsu se obtiene un radio de unos 47 píxeles.

**Justificación:** el parámetro se adapta a cada imagen (si las células se ven más grandes o más chicas, la distancia cambia con ellas) y no depende del GT.

### 8.4. Selección de la variante

Se compararon sobre train dos formas de obtener los marcadores, con distintos valores:
- **Máximos locales con distancia mínima** (`peak_local_max`), con distancia mínima de 0,5; 0,7 y 0,9 radios.
- **H-máximos:** máximos que sobresalen al menos una altura h de su entorno, con h de 0,1 y 0,2 radios.

| Variante | Otsu: precisión | Otsu: sensibilidad | Otsu: F1 | K-means: F1 |
|---|---:|---:|---:|---:|
| Sin Watershed | 0.952 | 0.485 | 0.643 | 0.492 |
| Máximos locales, 0,5 × radio | 0.934 | 0.834 | 0.881 | 0.816 |
| **Máximos locales, 0,7 × radio** | **0.938** | **0.828** | **0.880** | **0.839** |
| Máximos locales, 0,9 × radio | 0.936 | 0.821 | 0.875 | 0.832 |
| H-máximos, h = 0,1 × radio | 0.895 | 0.814 | 0.853 | 0.845 |
| H-máximos, h = 0,2 × radio | 0.910 | 0.804 | 0.854 | 0.834 |

**Decisión:** máximos locales con distancia mínima de **0,7 × radio**.

**Justificación:**
- Con Otsu, 0,5 y 0,7 son prácticamente iguales, pero 0,7 es claramente mejor con K-means, por lo que es más estable entre métodos.
- Los h-máximos generaron más falsos positivos con Otsu (más células cortadas en pedazos) y tardaron aproximadamente el doble.

### 8.5. Resultados

Detección célula a célula sobre train, con Watershed incorporado:

| Método | TP | FP | FN | Precisión | Sensibilidad | F1 |
|---|---:|---:|---:|---:|---:|---:|
| **Otsu** | 1.911 | 127 | 396 | **0.938** | 0.828 | **0.880** |
| K-means | 1.875 | 286 | 432 | 0.868 | 0.813 | 0.839 |
| GMM | 1.911 | 343 | 396 | 0.848 | 0.828 | 0.838 |

- Con Otsu, las células detectadas pasaron de 1.119 a **1.911** (de 2.307 eritrocitos del GT, sin contar los del borde), y el F1 de **0,643 a 0,880**.
- Watershed mejoró mucho más a K-means y GMM (que unían más células), pero Otsu sigue siendo el mejor.

**Limitaciones:**
- Los falsos positivos aumentaron (de 56 a 127 con Otsu): algunas células alargadas o con formas irregulares se cortan en dos.
- El filtro de área (6.000 - 22.000 píxeles) se calibró antes de incorporar Watershed; conviene revisarlo.
- Grupos de células muy superpuestas pueden seguir sin separarse, porque no se forma una "cintura" clara entre ellas.

**Implementación:** `segmentation/watershed_split.py` (`separar_celulas` y `estimar_radio`). Se activa con `APLICAR_WATERSHED = True` en `a1_segmentacion_base.py`, y con `False` se puede volver al etiquetado simple para comparar.

### 8.6. Uso del gradiente (Sobel) en Watershed

En el Bloque C (notebook C2) se compararon filtros pasa altos para resaltar el contorno de los eritrocitos: **Sobel y Prewitt con suavizado gaussiano previo** fueron los mejores (AUC borde vs fondo 0,95, contra 0,88 sin suavizar), y el Laplaciano el peor. Se propuso usar el gradiente como relieve de Watershed, para que el límite entre dos células caiga sobre su borde.

Se probó sobre train con Otsu, manteniendo los mismos marcadores (centros) y cambiando solo el relieve que se inunda:

| Relieve | Precisión | Sensibilidad | F1 |
|---|---:|---:|---:|
| **Distancia al fondo (elegido)** | 0.938 | 0.828 | **0.880** |
| Sobel de G suavizado | 0.931 | 0.680 | 0.786 |
| Distancia + 0,5 × Sobel | 0.938 | 0.825 | 0.878 |
| Distancia + 1 × Sobel | 0.941 | 0.821 | 0.877 |

**Decisión:** mantener la distancia al fondo. Sobel y Prewitt con gaussiano son los mejores detectores de bordes (C2), pero usar el gradiente en Watershed no mejoró la detección, porque los bordes entre células en contacto son débiles y los parásitos generan bordes internos (la respuesta del filtro dentro de las células parasitadas es 5 a 7 veces mayor que en las sanas).

## 9. Características de textura (GLCM) - Bloque B

Para cada eritrocito se calculan características de textura a partir de la matriz de co-ocurrencia de niveles de gris (GLCM).

**Parámetros:**
- **Canal verde:** es el que mejor contrasta el parásito con el citoplasma del eritrocito en las imágenes teñidas.
- **32 niveles de gris** (en lugar de 256): simplifica la matriz y reduce el costo de cálculo. Se usa el mismo mapeo para todas las células.
- **Distancias 1, 2 y 3 píxeles** y **ángulos 0°, 45°, 90° y 135°**: se promedian los cuatro ángulos para cada distancia, para que las características no dependan de la orientación de la célula.
- **Solo píxeles de la célula:** los píxeles del fondo se asignan a un nivel reservado que después se descarta, para que solo cuenten los pares de píxeles que están ambos dentro de la célula.
- **Características:** contraste, homogeneidad, energía, correlación y entropía (15 en total: 5 propiedades × 3 distancias).

### 9.1. Selección de atributos

El análisis se hizo sobre los recortes del GT de **train** (217 parasitadas y 2.135 no parasitadas).

1. **Capacidad de separación:** para cada característica se calculó el área bajo la curva ROC (AUC) mediante el estadístico U de Mann-Whitney. Un AUC de 0,5 indica que no separa las clases y 1 (o 0) que las separa perfectamente.
2. **Redundancia:** se calculó la correlación de Spearman entre características. Recorriendo el ranking de mejor a peor, se agregó cada característica solo si su correlación con las ya elegidas era menor a 0,9, y se descartaron las de separación menor a 0,6.

| Característica | AUC | Sentido |
|---|---:|---|
| contrast_d3 | 0.991 | mayor en parasitadas |
| contrast_d2 | 0.987 | mayor en parasitadas |
| homogeneity_d2 | 0.056 | menor en parasitadas |
| entropy_d2 | 0.919 | mayor en parasitadas |
| correlation_d1 | 0.890 | mayor en parasitadas |
| energy_d1 | 0.137 | menor en parasitadas |

**Atributos seleccionados:** `contrast_d3`, `correlation_d1` y `energy_d1`.

Como referencia, el tamaño de la célula (cantidad de píxeles) tiene AUC 0,518: no distingue parasitadas de no parasitadas, por lo que la separación se debe a la textura y no al tamaño.

### 9.2. Umbrales de clasificación

Para cada atributo seleccionado se eligió un umbral sobre la curva ROC de train, con dos criterios:
- **Youden:** maximiza sensibilidad + especificidad - 1.
- **F1:** maximiza el equilibrio entre precisión y sensibilidad, más adecuado con clases desbalanceadas (aproximadamente 1 parasitada cada 10 no parasitadas).

| Regla | Sensibilidad | Especificidad | Precisión | F1 |
|---|---:|---:|---:|---:|
| contrast_d3 ≥ 0,596 (Youden) | 0.940 | 0.981 | 0.833 | 0.883 |
| contrast_d3 ≥ 0,698 (F1) | 0.894 | 0.990 | 0.898 | 0.896 |
| correlation_d1 ≥ 0,944 (Youden) | 0.903 | 0.797 | 0.311 | 0.463 |
| energy_d1 ≤ 0,418 (Youden) | 0.788 | 0.821 | 0.309 | 0.444 |

Para estimar si el umbral generaliza a pacientes nuevos se hizo una **validación dejando un paciente afuera**: el umbral se calcula con 5 pacientes y se evalúa en el sexto, rotando. Con `contrast_d3` se obtuvo sensibilidad 0,940, precisión 0,533 y F1 0,680: la precisión baja bastante respecto de train, lo que indica que el umbral depende del paciente (diferencias de tinción).

**Limitaciones:**
- Los atributos y umbrales se obtuvieron con los **polígonos del GT**. Nuestras máscaras segmentadas tienen contornos distintos, por lo que hay que recalcular las características sobre las células segmentadas antes de aplicar los umbrales.
- Las parasitadas están concentradas en un paciente (209C68P29N tiene 145 de las 217 de train).
- La selección de atributos se hizo con todo train antes de la validación por paciente, por lo que esa validación puede ser algo optimista.

## 10. Integración de los bloques

El archivo `main.py` integra el pipeline completo. Por ahora:
1. Segmenta cada imagen con el método elegido (Bloque A), separa las células pegadas con Watershed y guarda recortes y máscaras de cada célula.
2. Calcula las características GLCM de cada célula detectada (Bloque B) y las guarda en `data/shared/features/features_texture_<método>.csv`.

**Decisiones sobre los formatos compartidos:**
- **Identificador único de célula (`cell_id`):** `{paciente}_{imagen}_{número de 4 dígitos}`. Permite unir las características de textura y de color de una misma célula.
- **Recortes de la imagen original:** los recortes se extraen de la imagen RGB original, no de la preprocesada, para que los Bloques B y C trabajen con los colores e intensidades reales.
- **Resultados separados por método:** cada método guarda en `data/shared/pred/<método>/`, para poder comparar sin sobrescribir.
- **Máscara de glóbulos blancos:** el Bloque C entregará una máscara booleana del tamaño de la imagen con `True` en los glóbulos blancos; esos píxeles se eliminan de la segmentación antes de la limpieza morfológica.

## 11. Estrategia general del proyecto

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

## 12. Evaluación y próximos pasos

**Estado actual:**
- Otsu, K-means y GMM implementados, con limpieza morfológica, separación de células mediante Watershed, filtrado y extracción de células.
- Evaluación global (IoU, Dice) y célula a célula (TP, FP, FN, precisión, sensibilidad, F1) sobre train.
- Características de textura GLCM calculadas y analizadas sobre el GT de train.
- Pipeline de `main.py` funcionando para segmentación y textura.

**Próximos pasos:**
1. Confirmar el método de segmentación final (propuesta: Otsu) y revisar el filtro de área después de Watershed.
2. Inspeccionar visualmente la segmentación sobre imágenes de train.
3. Incorporar la detección y exclusión de glóbulos blancos (Bloque C).
4. Recalcular la textura sobre las células segmentadas y desarrollar la identificación de eritrocitos parasitados con características de color y textura.
5. Evaluar el algoritmo final sobre el **conjunto de test**, una sola vez, manteniendo los parámetros fijos.

La evaluación final deberá distinguir entre la calidad de segmentación de píxeles, la detección de células individuales y la identificación de eritrocitos parasitados.

## 13. Bibliografía de referencia

Walliander, M., et al. (2013). *Automated segmentation of blood cells in Giemsa stained digitized thin blood films*. Diagnostic Pathology, 8(Suppl. 1), S37.

https://doi.org/10.1186/1746-1596-8-S1-S37

*Referencia consignada previamente; verificar los datos bibliográficos antes de la entrega final.*