# Criterios y decisiones del TP

Resumen de **qué hace cada paso del pipeline, con qué valor y por qué**. Sirve para justificar el código en el informe y en la defensa oral.
El registro completo de todo lo que se probó en el Bloque A, en orden cronológico, está en [`docs/report_detallado.md`](docs/report_detallado.md).

## 0. Reglas que respetamos en todo el trabajo

1. **Test no se usa para ninguna decisión.** Métodos, parámetros, umbrales y atributos se eligen solo con train. Test se corre **una única vez** al final, con todo fijo (`EVALUACION_FINAL = True` en `main.py`).
2. **División por paciente:** ningún paciente está a la vez en train y en test.
3. **Cada parámetro tiene un criterio explícito:** un percentil de train, una comparación en train o una regla automática por imagen.
4. **Parámetros relativos a la escala** cuando se puede (radio y área típicos estimados en cada imagen, sin GT).
5. **Contra el sobreajuste:** validación dejando un paciente afuera (LOPO) o tabla de sensibilidad alrededor del valor elegido.
6. **El GT se usa para medir y para elegir parámetros en train**, nunca para decidir en una imagen nueva. `procesar_imagen` de `main.py` no usa el GT.

## 1. Dataset

- **NIH-NLM-ThinBloodSmearsPf, Polygon Set.** Frotis finos teñidos con Giemsa, *P. falciparum*. JPG RGB de 5312 × 2988 píxeles, con un campo circular de microscopio.
- **GT:** polígonos de cada célula, con la clase `Parasitized`, `Uninfected` o `White_Blood_Cell`.
- **Selección:** 16 imágenes de 8 pacientes (2 por paciente), con variedad de tinción y de parasitemia.

| Conjunto | Pacientes | Imágenes |
|---|---:|---:|
| Train | 6 | 12 |
| Test | 2 | 4 |

- En total hay 240 parasitados, 3.111 sanos y 13 glóbulos blancos. **Las clases están muy desbalanceadas** (alrededor de 1 parasitada cada 13 sanas), por eso se usa F1 y no exactitud.
- En train, las parasitadas están concentradas: el paciente 209 tiene 145 de las 217, y tres pacientes tienen solo 4 o 5.

## 2. Pipeline

```
Imagen RGB
 → [A] campo circular → CLAHE (canal G) + gaussiano → Otsu sobre G → apertura, cierre y relleno de huecos
 → [A] Watershed (separa células pegadas) → [A+C] exclusión de glóbulos blancos → [A] filtro de área y borde
 → recorte y máscara de cada célula (de la imagen ORIGINAL, sin CLAHE)
 → [B] textura GLCM   +   [C] color (con normalización por el fondo)
 → [B+C] clasificación: parasitada si (tiene mancha oscura) Y (contrast_d3 ≥ 0,709)
 → [C] parasitemia y máscara de eritrocitos parasitados
 → evaluación contra el GT (solo para medir)
```

**Estrategia elegida:** primero segmentar **todos** los eritrocitos y después clasificar cada uno. Así se separa el problema de *encontrar* las células del de *decidir si están infectadas*, y la parasitemia se calcula directamente (parasitadas / total).

## 3. Preprocesamiento (Bloque A y C)

| Paso | Valor | Criterio | Archivo |
|---|---|---|---|
| Campo circular | gris > 20; región conexa más grande; erosión de 5 px | Excluir el borde negro, que Otsu confundiría con células. Los 5 px evitan artefactos JPEG del borde. | `segmentation/preprocess.py` |
| Canal verde (G) | — | Es el que mejor separa célula de fondo: AUC 0,969 (C1, train), igual a L de Lab (0,967). | `cluster_segmentation.py` |
| CLAHE en G | clip 2,0; grilla 8×8 | Fue la mejora más grande en la comparación de configuraciones (IoU 0,803 → 0,822). | `preprocess.py` |
| Filtro gaussiano | 5×5 | Reduce ruido antes de Otsu. Casi no influye en el resultado (ver tabla de sensibilidad, sección 5). | `preprocess.py` |
| Corrección de iluminación | **descartada** | Empeoró (IoU 0,803 → 0,775). | — |
| Normalización de color por el fondo | fondo de referencia de train: RGB (220, 181, 166) | Cada canal se multiplica para que el fondo quede igual a la referencia. **No se usa para segmentar** (Otsu ya calcula el umbral en cada imagen, C2), **sí para color y glóbulos blancos**, que usan umbrales fijos. | `color_stain/normalizacion.py` |

**Los recortes se toman de la imagen original**, no de la preprocesada, para que la textura y el color usen las intensidades reales.

**Comparación de configuraciones** (IoU promedio en train, `segmentation/comparar_otsu.py`):

| Configuración | IoU | Dice |
|---|---:|---:|
| Otsu sin filtro | 0,762 | 0,864 |
| Otsu + gaussiano + limpieza | 0,803 | 0,890 |
| CLAHE + gaussiano + Otsu, sin limpieza | 0,786 | 0,880 |
| **CLAHE + gaussiano + Otsu + limpieza (elegida)** | **0,822** | **0,902** |

## 4. Segmentación: Otsu sobre G (Bloque A)

- **Otsu** elige automáticamente, en cada imagen, el umbral que mejor separa dos grupos de intensidad. Los píxeles más oscuros que el umbral son célula.
- Se calcula solo con los píxeles del campo circular.

**Por qué Otsu y no K-means o GMM** (agrupamiento por color en Lab, K = 3, ajustado en cada imagen sin etiquetas). Con el pipeline completo, en train:

| Método | IoU | F1 de detección |
|---|---:|---:|
| **Otsu (G)** | **0,822** | **0,946** |
| K-means (Lab) | 0,806 | 0,879 |
| GMM (Lab) | 0,786 | 0,864 |

Otsu es el mejor y además el más simple. K-means y GMM unen más células vecinas.
- **Lab** para el agrupamiento: fue el espacio que mejor agrupó los píxeles sin etiquetas (C1, ARI). En HSV el tono es circular.
- **K = 3:** fondo, eritrocito y zonas oscuras.

## 5. Limpieza morfológica (Bloque A)

| Operación | Valor | Para qué |
|---|---|---|
| Apertura | elipse 5×5 | Borra puntos de ruido y uniones finas entre células. |
| Cierre | elipse 7×7 | Une pedazos de una misma célula. |
| Relleno de huecos | huecos ≤ **4.086 px** | Los eritrocitos tienen el **centro más claro** y Otsu los deja como "donas". |

**Criterio del relleno de huecos (desde el GT de train):** se midieron los huecos que caen **dentro de un eritrocito del GT** (centros claros). Su mediana es 773 px, el p95 2.800 y el **p99 4.086**. Se eligió el **p99**, que rellena casi todos los centros sin rellenar el fondo entre células. Con el valor anterior (1.500, elegido a mano) quedaba sin rellenar el 27 % de los centros y el F1 de detección era 0,88 (ahora 0,946).

**Sensibilidad de los kernels** (se cambia un parámetro por vez, train, `segmentation/sensibilidad_kernels.py`):

| Parámetro | F1 de detección según el valor |
|---|---|
| Apertura | 3 → 0,908 · **5 → 0,946** · 7 → 0,946 · 9 → 0,946 |
| Cierre | 5 → 0,948 · **7 → 0,946** · 9 → 0,938 · 11 → 0,935 |
| Gaussiano | sin → 0,946 · 3 → 0,946 · **5 → 0,946** · 7 → 0,946 |

Los tres están en una zona estable. Solo empeora la apertura 3, que deja pasar ruido (152 objetos de más contra 59), y el cierre de 9 o más, que pega células. El IoU queda entre 0,821 y 0,823 en todos los casos.

## 6. Separación de células pegadas: Watershed (Bloque A)

- **Problema:** dos eritrocitos que se tocan quedan como un solo objeto. Sin Watershed se perdía alrededor de la mitad de las células (sensibilidad 0,49).
- **Método:**
  1. Transformada de distancia: es máxima en el centro de cada célula.
  2. Máximos locales: son los marcadores, uno por célula.
  3. Inundación desde cada marcador: el límite entre dos células queda en la "cintura" donde se tocan.
- **Parámetro automático:** la distancia mínima entre centros es **0,7 × radio típico** de la imagen. El radio es la mediana del máximo de la distancia de cada objeto, sin GT (unos 47 px).
- **Criterio:** se compararon en train 0,5, 0,7 y 0,9 radios y la variante con h-máximos. 0,7 fue el mejor para K-means y casi igual al mejor para Otsu, o sea el más estable. Los h-máximos cortaban más células y tardaban el doble.
- **Gradiente (Sobel) como relieve: descartado.** Sobel y Prewitt con gaussiano son los mejores detectores de borde (C2), pero en Watershed bajaron el F1 (0,880 → 0,786). Los bordes entre células pegadas son débiles y los parásitos generan bordes internos.

## 7. Exclusión de glóbulos blancos (Bloque A + C)

- **Problema:** Otsu segmenta los glóbulos blancos y Watershed los corta en pedazos. En train, 3 pedazos pasaban como eritrocitos.
- **Idea:** el **núcleo** del glóbulo blanco es grande, violeta y muy oscuro; el citoplasma no se distingue por color. Por eso se detecta el núcleo y se elimina el **objeto entero** que lo contiene.

| Criterio | Valor | Origen (solo train) |
|---|---|---|
| Violeta | b (Lab) < 127 | Umbral de Youden píxel a píxel, glóbulo blanco contra eritrocito (C1, AUC 0,92). |
| Oscuro | G < 100 | Núcleo con G ≈ 52; el 95 % de los píxeles de parasitadas tiene G > 124. |
| Tamaño del núcleo | ≥ 6.000 px | 3.000, 6.000 y 12.000 dan igual; se eligió el central. |
| Fracción de núcleo en el objeto | ≥ 10 % | Los pedazos de glóbulo blanco tienen entre 68 y 96 %; los eritrocitos, 0 %. |

- **Resultado:** 0 pedazos de glóbulo blanco aceptados y **0 eritrocitos perdidos**.
- Con "solo violeta", la versión original de C3, se perdían 3 parasitadas del paciente 209, el de tinción más oscura. Por eso se agregó la condición de oscuro.
- **Limitación:** en train hay solo 10 glóbulos blancos.

## 8. Filtro de área y borde (Bloque A)

- **Área relativa a cada imagen, sin GT:** se aceptan los objetos entre **0,6 y 2,0 veces el área típica** de la imagen.
  - El área típica es la **mediana ponderada por área**: el objeto en el que se completa la mitad de los píxeles de célula. Da entre 10.000 y 14.600 px en train.
  - La mediana simple no sirve: con muchos fragmentos chicos, como en el paciente 209, daba unos 2.000 px.
- **Criterio:** grilla en train (mínimo de 0,3 a 0,8 y máximo de 1,4 a 2,2) con validación dejando un paciente afuera. **Las 6 vueltas eligieron 0,6 – 2,0.** Alrededor de ese valor el F1 casi no cambia: con 0,5–0,6 de mínimo y 1,8–2,2 de máximo da 0,945–0,946; con un mínimo de 0,8 baja a 0,905.
- **Borde:** se descartan las células que tocan una franja de 3 px del borde del círculo, porque están cortadas. En la evaluación también se sacan del GT, para no contarlas como errores.

## 9. Textura: GLCM (Bloque B)

- Se calcula sobre el **canal G del recorte**, con **32 niveles de gris**, a **distancias 1, 2 y 3 px** y **4 ángulos promediados** (así no depende de la orientación). Solo cuentan los pares de píxeles dentro de la célula.
- Hay 5 propiedades (contraste, homogeneidad, energía, correlación y entropía) × 3 distancias = 15 atributos.
- **Selección, sobre las células segmentadas de train:**
  - ranking por AUC, con el estadístico U de Mann-Whitney y su p-valor;
  - se descartan los redundantes (|Spearman| ≥ 0,9) y los que separan poco (< 0,6).
  - **Elegidos:** `contrast_d3` (AUC 0,984, p ≈ 10⁻¹¹²), `entropy_d2` (0,914) y `correlation_d1` (0,884).
- **Por qué el contraste:** el parásito es una mancha oscura dentro de un citoplasma claro y uniforme, y eso aumenta las diferencias de gris entre píxeles vecinos. El tamaño de la célula **no** separa las clases (AUC 0,52), así que lo que separa es la textura, no el tamaño.

## 10. Color (Bloque C)

- Se calculan **47 atributos de color por célula** (C4) sobre el recorte **normalizado por el fondo**: media, desvío y mediana de G, L, a, b y S; percentiles; fracción de píxeles oscuros o violetas; histogramas; y las manchas de C3.
- **Máscara de manchas oscuras (C3):** dentro de la célula, achicada 5 px para ignorar el borde, se marcan los píxeles con **G al menos 39 niveles por debajo de la mediana de G de esa célula**. Se cuenta una mancha si tiene **al menos 50 px**. Es relativa a cada célula, así que no depende de la tinción. Los parámetros se eligieron en C3 con validación por paciente, en train.
- **Criterio de color (C5): "parasitada si tiene al menos una mancha"** (`n_manchas ≥ 1`). Se comparó con 8 reglas de umbrales (1 atributo, AND, OR o mayoría de 3; Youden o máximo F1) con LOPO:

| Regla (LOPO, células del GT de train) | Parasitadas perdidas (de 217) | Precisión | Exactitud bal. media por paciente | F1 peor paciente |
|---|---:|---:|---:|---:|
| **Tiene mancha de C3 (elegida)** | **3** | 0,781 | **0,951** | **0,316** |
| Mayoría de 3 (b_rel_p1, contraste_G_mancha, a_desvio) | 13 | 0,883 | 0,900 | 0,154 |

  **Por qué esta regla:**
  - **pierde muchas menos parasitadas**, y en malaria es más grave no detectar una infección que marcar de más;
  - es **más estable entre pacientes**;
  - es **simple y explicable**: "hay una mancha oscura dentro del eritrocito".
- **Sobre las células segmentadas (C7):** los atributos de color siguen separando las clases. El AUC cambia ±0,02 respecto de las células del GT.

## 11. Clasificación final: color Y textura (Bloque B + C)

**Regla final: una célula es parasitada si tiene al menos una mancha oscura (color, C5) Y `contrast_d3` ≥ 0,709 (textura).**

- El umbral de textura es el que **maximiza F1** en las células segmentadas de train.
- Está guardado en `texture_eval/results/b_metodo_elegido.json`; `main.py` solo lo lee y lo aplica, no recalcula nada.

**Criterio de elección:** se compararon 10 estrategias con **LOPO sobre las células segmentadas de train** (las que ve el pipeline real):

| Estrategia | Precisión | Sensibilidad | F1 LOPO |
|---|---:|---:|---:|
| **Color C5 AND textura (elegida)** | **0,891** | 0,824 | **0,856** |
| Textura sola (contrast_d3) | 0,865 | 0,839 | 0,852 |
| Logística de textura | 0,832 | 0,844 | 0,838 |
| Color C5 sola | 0,757 | 0,894 | 0,820 |
| C5 OR textura | 0,742 | 0,910 | 0,817 |
| Logística color + textura | 0,872 | 0,754 | 0,809 |

- **Regla de elección:** se toma la estrategia de mayor F1 LOPO solo si supera a C5 sola por al menos 0,02. Si no, se queda C5, que es más simple. C5 AND textura la supera por 0,036.
- **Por qué combinar:** el color solo es muy sensible pero marca muchas sanas (por ejemplo, manchas por tinción o superposición). Exigir además una textura alta **baja los falsos positivos a menos de la mitad**, a costa de perder algunas parasitadas.
- **Para la defensa:** textura sola queda casi empatada (0,852). La combinación gana en precisión y usa las dos fuentes de información, que es lo que proponía el anteproyecto.

## 12. Parasitemia y máscara final (Bloque C)

- **Parasitemia de una imagen** = células clasificadas como parasitadas / total de eritrocitos detectados × 100.
- **Máscara de eritrocitos parasitados:** la unión de las máscaras de las células parasitadas (célula completa, no solo el parásito). Se guarda una por imagen en `data/shared/pred/parasitadas/mascaras/`, y en `parasitadas/imagenes/` la imagen original mostrando solo esas células.
- **Comparación con el GT:** se compara el **promedio por imagen** de la parasitemia estimada y de la real, y el **error medio por imagen**. No se compara una parasitemia global (todas las células juntas) con un promedio por imagen, porque se calculan distinto.

## 13. Compresión y descompresión JPEG (Bloque C, C6)

Cada imagen de train se recomprimió con calidades 95 a 10 y se descomprimió. Después se midieron la calidad de la imagen, el cambio en los atributos de color y el efecto sobre el criterio de color:

| Calidad | Tamaño (% del original) | PSNR G (dB) | SSIM | ΔE en células | F1 del criterio de color | Células que cambian de decisión |
|---|---:|---:|---:|---:|---:|---:|
| original | 100 | — | — | — | 0,906 | 0 |
| 85 | 34 | 49,2 | 0,982 | 0,7 | 0,904 | 1 |
| 75 | 22 | 48,0 | 0,983 | 0,9 | 0,908 | 3 |
| 50 | 16 | 46,1 | 0,974 | 1,4 | 0,910 | 2 |
| 10 | 13 | 38,3 | 0,952 | 4,6 | 0,912 | 11 |

**Conclusión:**
- **Hasta calidad 75** el archivo pasa a ocupar el 22 % y los atributos de color casi no cambian (Spearman > 0,97).
- **La clasificación es robusta** a la compresión: cambian como mucho 11 de alrededor de 2.350 células.
- Los atributos más afectados son los de variación fina del color (`a_desvio`, `b_rel_p1`), por el submuestreo de croma de JPEG. Con calidad 10 aparecen los bloques de 8×8.

## 14. Cómo se evalúa

| Qué | Métrica | Detalle |
|---|---|---|
| Segmentación (píxeles) | IoU y Dice | Máscara contra GT de eritrocitos, dentro del campo circular; promedio por imagen. |
| Detección de células | TP, FP, FN, precisión, sensibilidad, F1 | Emparejamiento uno a uno (algoritmo húngaro) con **IoU ≥ 0,5**. Se sacan del GT las células del borde. Los conteos se suman sobre todas las imágenes. |
| Clasificación | precisión, sensibilidad, especificidad, F1 | Sobre las células detectadas con pareja en el GT: clase predicha contra `clase_gt`. |
| Parasitemia | estimada contra real | Promedio por imagen y error medio por imagen (puntos porcentuales). |
| Final en test | detección de parasitadas de punta a punta | Solo las células marcadas como parasitadas contra las parasitadas del GT (IoU ≥ 0,5): mide segmentación y clasificación juntas. |

## 15. Resultados en train

*Son resultados de desarrollo: los parámetros se eligieron con estos datos. La estimación independiente es la de test.*

| Etapa | Resultado |
|---|---|
| Segmentación | IoU 0,822 · Dice 0,902 |
| Detección de células | 2.184 detectadas: 2.125 aciertos, 59 de más, 182 sin detectar. Precisión 0,973 · sensibilidad 0,921 · **F1 0,946** |
| Clasificación (199 parasitadas y 1.926 sanas emparejadas) | Precisión 0,911 · sensibilidad 0,824 · especificidad 0,992 · **F1 0,865** |
| De punta a punta | De las 217 parasitadas del GT: 18 se pierden en la segmentación y 164 se detectan y clasifican bien (**76 %**) |
| Parasitemia (promedio por imagen) | Estimada 8,88 % contra real 9,75 % · **error medio 1,3 puntos** por imagen |

F1 de la clasificación por paciente: 208 → 0,88 · 209 → 0,89 · 211 → 0,89 · 212 → 0,75 · **263 → 0,00** · 282 → 0,86.

## 16. Limitaciones

- **Pocas parasitadas por paciente:** tres pacientes de train tienen 4 o 5, y el F1 por paciente es muy ruidoso. En el **paciente 263** el método no acierta ninguna de sus 5. No se corrigió mirando ese resultado, porque sería sobreajustar.
- **La regla AND prioriza la precisión:** en parasitemias muy bajas puede pasar por alto infecciones leves.
- **Test tiene solo 2 pacientes:** su resultado va a tener mucha incertidumbre.
- **La selección de atributos y la elección del método usan todo train**; el LOPO solo reajusta umbrales. Las métricas LOPO son algo optimistas.
- **Segmentación:** grupos de células muy superpuestas pueden no separarse. Además, la detección no es perfecta (182 eritrocitos sin detectar).
- **Hay solo 10 glóbulos blancos en train** para validar su exclusión.

## 17. Cumplimiento de la consigna (temas obligatorios)

| Tema de la consigna | Dónde está en el TP |
|---|---|
| Contextualización (imágenes y dataset) | Sección 1; `common/verificar_dataset.py`; notebook C1 (tinción y espacios de color) |
| Mejoramiento de imágenes | CLAHE, filtro gaussiano y normalización de color por el fondo (sección 3) |
| Reconstrucción | Relleno de huecos (reconstrucción morfológica de los centros claros, sección 5) y reconstrucción de la imagen al descomprimir JPEG (sección 13) |
| Compresión y descompresión | Experimento JPEG, notebook C6 (sección 13) |
| Segmentación | Otsu, K-means y GMM; morfología; Watershed (secciones 4 a 8) |
| Análisis de texturas | GLCM (sección 9) |
| Resultados: visualización, estadística, comparaciones, significancia | Imágenes de parasitadas; AUC con p-valores de Mann-Whitney e intervalo de confianza por bootstrap; LOPO; tablas de sensibilidad; comparación de métodos (secciones 11 a 15) |

## 18. Bibliografía

- Walliander, M., et al. (2013). *Automated segmentation of blood cells in Giemsa stained digitized thin blood films*. Diagnostic Pathology, 8(Suppl. 1), S37. https://doi.org/10.1186/1746-1596-8-S1-S37
- Dataset: NIH-NLM-ThinBloodSmearsPf (Polygon Set), National Library of Medicine.

*Verificar los datos bibliográficos antes de la entrega final.*
