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

**Tres formas de usar el mismo pipeline.** Las tres llaman a `procesar_imagen` de `main.py`, que **no usa el GT** y aplica los parámetros fijos elegidos con train:

| Uso | Comando | Para qué |
|---|---|---|
| Desarrollo (train) | `python main.py` | Procesa train y lo **evalúa contra el GT**. Con esto se eligieron todos los parámetros. |
| Evaluación final (test) | `main("test")` con `EVALUACION_FINAL = True` | Una única vez, con todo fijo. Resultados en `pred/resultados_test/`. |
| **Casos incógnita** | `python predecir.py <imagen o carpeta>` | Imágenes nuevas **sin GT** (las que traiga la cátedra). Devuelve la parasitemia, la máscara de parasitadas y una imagen con cada célula marcada (rojo = parasitada, verde = sana). Resultados en `pred/incognita/`. |

**Visualización:** `python figuras.py train | test | incognita` arma, con los resultados ya guardados, **una figura por paciente** (sus 2 imágenes; al cerrarla aparece la siguiente), con una fila por imagen y 4 columnas:
1. original;
2. máscara de parasitadas;
3. solo las células parasitadas;
4. contornos: **un contorno por célula**, rojo si se clasificó como parasitada y verde si no.

No vuelve a correr el pipeline. La función de contornos está una sola vez, en `predecir.py`, y la usan los dos scripts.

`predecir.py` se comprobó con imágenes de train tratadas como incógnitas: da exactamente la misma parasitemia que `main.py` para esas imágenes.

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

## Cómo se eligieron los atributos de textura y color (secciones 9 a 11)

Los Bloques B y C siguieron los mismos pasos, **solo con train**:
1. Calcular **todos** los atributos de cada célula (15 de textura y 47 de color).
2. **Ranking:** AUC de cada atributo para separar parasitadas de sanas.
3. **Selección:** quedarse con los mejores, sin atributos redundantes (muy correlacionados entre sí).
4. Armar una **regla de decisión** y validarla dejando un paciente afuera (LOPO).

Primero se hizo con las células **del GT** (contornos perfectos, para explorar). Después se repitió con **las células que segmenta el pipeline** (notebook C7 para color y sección 8 de `b_textura_evaluacion2` para textura), porque son las que recibe el clasificador. La **combinación final** de color y textura se eligió en la sección 8 de `b_textura_evaluacion2`.

## 9. Textura: GLCM (Bloque B)

- Se calcula sobre el **canal G del recorte**, con **32 niveles de gris**, a **distancias 1, 2 y 3 px** y **4 ángulos promediados** (así no depende de la orientación). Solo cuentan los pares de píxeles dentro de la célula.
- Hay 5 propiedades (contraste, homogeneidad, energía, correlación y entropía) × 3 distancias = 15 atributos.
- **Selección, sobre las células segmentadas de train:**
  - ranking por AUC, con el estadístico U de Mann-Whitney y su p-valor;
  - se descartan los redundantes (|Spearman| ≥ 0,9) y los que separan poco (< 0,6).
  - **Ranking:** `contrast_d3` 0,984 · `contrast_d2` 0,981 · `homogeneity_d2` 0,935 (menor en parasitadas) · `entropy_d2` 0,914 · `correlation_d1` 0,884 · `energy_d1` 0,866 (menor en parasitadas).
  - **Elegidos:** `contrast_d3` (AUC 0,984, p ≈ 10⁻¹¹²), `entropy_d2` (0,914) y `correlation_d1` (0,884). Las otras distancias del contraste y la homogeneidad quedaron afuera por redundantes: miden casi lo mismo que `contrast_d3`.
- **Por qué el contraste:** el parásito es una mancha oscura dentro de un citoplasma claro y uniforme, y eso aumenta las diferencias de gris entre píxeles vecinos. El tamaño de la célula **no** separa las clases (AUC 0,52), así que lo que separa es la textura, no el tamaño.

## 10. Color (Bloque C)

- Se calculan **47 atributos de color por célula** (C4) sobre el recorte **normalizado por el fondo**: media, desvío y mediana de G, L, a, b y S; percentiles; fracción de píxeles oscuros o violetas; histogramas; y las manchas de C3.
- **Ranking de los atributos de color (C4, AUC promedio por paciente, train):**

| Atributo | Qué mide | AUC |
|---|---|---:|
| `b_rel_p1` | qué tan violeta es lo más violeta de la célula, respecto de su propia mediana | 0,974 |
| `contraste_G_mancha` | qué tan oscura es la mancha respecto del resto de la célula | 0,961 |
| `n_manchas` | cantidad de manchas oscuras (máscara de C3) | 0,958 |
| `a_desvio` | cuánto varía el canal a dentro de la célula | 0,957 |
| Medias de color (G, L, a, b, S) | color promedio de la célula | 0,69 – 0,81 |

  **Conclusión:** lo que separa **no es el color promedio**, sino **lo local**: un punto más oscuro o más violeta que el resto de la misma célula. El parásito no cambia el color de toda la célula, le agrega una mancha. Por eso los desvíos separan mucho mejor que las medias.
  **Seleccionados** (sin redundancia, |ρ| < 0,8): `b_rel_p1`, `contraste_G_mancha`, `a_desvio`, `G_hist_0`, `frac_violeta`, `b_desvio`, `b_hist_2` y `a_p95`.
- **Máscara de manchas oscuras (C3):** dentro de la célula, achicada 5 px para ignorar el borde, se marcan los píxeles con **G al menos 39 niveles por debajo de la mediana de G de esa célula**. Se cuenta una mancha si tiene **al menos 50 px**. Es relativa a cada célula, así que depende menos de la tinción, aunque con otra tinción pueden aparecer puntos oscuros que no son parásitos (ver sección 16). Los parámetros se eligieron en C3 con validación por paciente, en train.
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
- **Por qué la regla de color no es simplemente el mejor atributo (`b_rel_p1`):** como regla de un solo umbral, sobre las células segmentadas dio F1 0,740 (LOPO). "Tiene mancha" dio 0,820 y detecta más parasitadas.

## 11. Clasificación final: color Y textura (Bloque B + C)

**Regla final: una célula es parasitada si tiene al menos una mancha oscura (color, C5) Y `contrast_d3` ≥ 0,709 (textura).**

### 11.1. Explicado en simple

**¿Cómo se ve un eritrocito parasitado?** La tinción de Giemsa pinta el parásito de violeta oscuro. Al microscopio, un eritrocito infectado es una **célula rosada clara con un puntito oscuro adentro**; uno sano es **rosado parejo**. Las dos reglas buscan ese puntito, cada una desde un ángulo distinto.

**Regla de color: "tiene mancha".** Para cada célula:
1. Se calcula el **verde típico de esa célula** (la mediana de G). Oscuro = verde bajo.
2. Se marcan los píxeles **mucho más oscuros que lo típico de esa misma célula**: al menos 39 niveles menos.
3. Si esos píxeles forman una zona de **al menos 50 píxeles**, es una **mancha**. Las zonas más chicas son ruido y se ignoran.
4. Se ignoran los 5 píxeles del borde de la célula, que suele ser más oscuro y daría falsas manchas.

```
Célula sana:              Célula parasitada:
  ░░░░░░░                   ░░░░░░░
 ░░░░░░░░░                 ░░░░██░░░   ← mancha oscura (parásito)
 ░░░░░░░░░                 ░░░░██░░░
  ░░░░░░░                   ░░░░░░░
 0 manchas → no            1 mancha → sí (según color)
```

La comparación es **contra la misma célula**, no contra un valor fijo. Así funciona aunque una imagen esté más teñida que otra.

**Regla de textura: "contraste alto".**
- El **contraste GLCM** mide qué tan distintos son los grises de píxeles vecinos, a 3 píxeles de distancia (`contrast_d3`). Es un atributo **de textura**, no de color.
- Una célula sana es pareja, así que su contraste es bajo. Una parasitada tiene la mancha oscura al lado del citoplasma claro, así que su contraste es alto.
- Se probaron **los 15 atributos de textura** (5 medidas × 3 distancias) y **`contrast_d3` fue el que mejor separa** (AUC 0,984).
- Combinar varios atributos de textura (regresión logística con los 3 seleccionados) no lo superó: F1 0,838 contra 0,852.

**¿Por qué pedir las dos?** El color ve **el puntito oscuro** y la textura ve **que la célula dejó de ser pareja**. Pedir las dos a la vez evita marcar células que tienen una sola de las señales por otra causa, por ejemplo un resto de tinción o una superposición. Por eso los falsos positivos bajan a menos de la mitad.

### 11.2. Cómo se eligieron los umbrales (y dónde entra el GT)

1. `main.py` segmenta las imágenes de **train** y calcula los atributos de cada célula detectada. Por ejemplo, una célula tiene contraste 0,85 y otra 0,40.
2. Cada célula detectada se empareja con **su célula del GT** (IoU ≥ 0,5) y se le copia la clase: parasitada o sana (`clase_gt` en `etiquetas_train.csv`). **El GT solo aporta la respuesta correcta**; los atributos son siempre de nuestras células segmentadas.
3. Se prueban muchos umbrales posibles (contraste ≥ 0,5, ≥ 0,6, ≥ 0,7…). Para cada uno se cuentan los aciertos contra `clase_gt` y **se elige el que maximiza F1**. Así salió 0,709.
4. El umbral **queda fijo** en `b_metodo_elegido.json`. En una imagen nueva (test o un caso incógnita) se aplica tal cual, **sin mirar el GT**.

Es como estudiar con ejercicios que tienen las respuestas (train) para aprender el criterio, y después rendir el examen (test) sin las respuestas. Los parámetros de la mancha (39 niveles, 50 píxeles, 5 píxeles de borde) se eligieron de la misma forma en C3, con validación dejando un paciente afuera.

### 11.3. Comparación de estrategias

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

## 15b. Resultados en test (evaluación final, una única vez)

Test se corrió **una sola vez**, con todos los parámetros fijos. La versión del código quedó marcada en git con la etiqueta `evaluacion-final`, y después **no se cambió nada**. Son 4 imágenes de 2 pacientes:
- **231C89P50:** 23 eritrocitos parasitados;
- **305C212:** **ninguno**, es un paciente sano.

Lo que se imprimió quedó guardado en `data/shared/pred/resultados_test/resumen_test.txt`, y las figuras en `resultados_test/parasitadas/figura_test_<paciente>.png`.

### Comparación train – test

Entre corchetes, el intervalo de confianza del 95 % (Wilson).

| Etapa | Train | **Test** |
|---|---|---|
| Segmentación (IoU / Dice) | 0,822 / 0,902 | **0,821 / 0,902** |
| Detección de células (F1) | 0,946 (P 0,973 · S 0,921) | **0,963** (P 0,976 · S 0,950 [0,93–0,96]) |
| Clasificación, color Y textura (F1) | 0,865 (P 0,911 · S 0,824) | **0,791** (P 0,739 [0,54–0,87] · S 0,850 [0,64–0,95]) |
| Clasificación, solo color (F1, referencia) | 0,832 (P 0,777 · S 0,894) | 0,690 (P 0,526 · S 1,000) |
| Especificidad de la clasificación | 0,992 | **0,993** |
| De punta a punta: parasitadas del GT bien encontradas | 164 / 217 (76 %) | **17 / 23 (74 %)** [0,54–0,87] |
| Parasitemia: error medio por imagen | 1,33 puntos | **0,85 puntos** |

### Por imagen

| Imagen | Paciente | Parasitemia estimada | Parasitemia real | Sanas marcadas como parasitadas | Parasitadas que se pasaron |
|---|---|---:|---:|---:|---:|
| 162300 | 231 | 3,6 % | 3,8 % | 0 | 1 |
| 162504 | 231 | 6,2 % | 7,4 % | 0 | 2 |
| 105834 | 305 (sano) | 1,4 % | 0,0 % | 4 | – |
| 110230 | 305 (sano) | 0,7 % | 0,0 % | 2 | – |

De las 23 parasitadas de test: **3 se pierden en la segmentación** (no quedan como célula detectada), **3 se clasifican como sanas** y **17 se detectan bien**.

### Análisis

1. **La segmentación generaliza muy bien:** el IoU es el mismo que en train y la detección incluso mejora (F1 0,963). Los parámetros relativos a cada imagen (radio y área típicos) funcionan en pacientes nuevos.
2. **La clasificación baja respecto de train (F1 0,865 → 0,791), pero sigue siendo buena**, y la causa es clara:
   - en el **paciente infectado (231)** el método **no marca ninguna sana como parasitada** (precisión 1,0) y detecta 17 de 20 parasitadas segmentadas. Su F1 es 0,919;
   - **todos los falsos positivos (6) están en el paciente sano (305)**. Sus imágenes tienen una **tinción muy distinta a la de train**: gris azulada y pálida, contra rosada (ver la figura). Con otra tinción aparecen manchas oscuras que no son parásitos (por ejemplo, plaquetas o restos apoyados sobre un eritrocito).
3. **El F1 de 0,000 del paciente 305 no significa que falló todo:** ese paciente no tiene parasitadas, así que el F1 no está definido. Lo que importa es que marcó **6 de 566 eritrocitos (1 %)**, y la parasitemia estimada fue 1,4 % y 0,7 % en vez de 0 %.
4. **Test confirma la decisión de combinar color y textura.** Con solo color, el paciente sano habría tenido el triple de falsos positivos (18 en total contra 6) y el F1 caería a 0,690. La textura vuelve a eliminar la mayoría de los falsos positivos, igual que en train.
5. **La parasitemia se estima bien:** error medio de 0,85 puntos por imagen. En el paciente infectado queda a menos de 1,2 puntos de la real.
6. **Incertidumbre:** con 23 parasitadas, los intervalos de confianza son anchos. Por ejemplo, la sensibilidad de punta a punta (74 %) podría estar entre 54 % y 87 %. Por eso test se toma como **una confirmación de que el método generaliza**, no como una medida exacta.

**Implicancia clínica:** el sistema es útil para **estimar la parasitemia** en un paciente infectado, pero en un paciente sano con una tinción distinta puede dar una parasitemia baja (alrededor del 1 %) que no existe. Un resultado de ese orden debería confirmarse visualmente; la imagen de contornos sirve justamente para eso.

## 16. Limitaciones

- **Pocas parasitadas por paciente:** tres pacientes de train tienen 4 o 5, y el F1 por paciente es muy ruidoso. En el **paciente 263** el método no acierta ninguna de sus 5. No se corrigió mirando ese resultado, porque sería sobreajustar.
- **La regla AND prioriza la precisión:** en parasitemias muy bajas puede pasar por alto infecciones leves.
- **Test tiene solo 2 pacientes y 23 parasitadas:** los intervalos de confianza son anchos (sección 15b).
- **Sensibilidad a la tinción:** en test, un paciente con una tinción muy distinta a la de train (gris azulada) concentró todos los falsos positivos. La normalización por el fondo corrige el brillo y el balance de color, pero no un cambio de tinción tan grande. La parte más sensible es la textura: el contraste se calcula sobre el canal verde sin normalizar y se compara con un umbral fijo (0,709), así que si la tinción cambia el contraste de las células, el umbral puede dejar de ser adecuado. La regla de la mancha es relativa a cada célula, pero con otra tinción aparecen puntos oscuros (plaquetas, restos) que cuenta como manchas. Train tenía solo 6 pacientes, todos con tinción parecida. Una mejora sería entrenar con más variedad de tinciones o usar una normalización de tinción más completa.
- **La selección de atributos y la elección del método usan todo train**; el LOPO solo reajusta umbrales. Las métricas LOPO son algo optimistas.
- **Segmentación:** grupos de células muy superpuestas pueden no separarse. Además, la detección no es perfecta (182 eritrocitos sin detectar).
- **Parásitos en el borde de la célula:** cuando el parásito está justo en el borde del eritrocito, la segmentación a veces lo deja **afuera de la máscara** de la célula, que queda con forma de "C". La mancha oscura no entra en la célula, así que se clasifica como sana. Se ve en las imágenes de contornos (células verdes con un punto violeta pegado en el hueco) y explica parte de las parasitadas que se pierden. Una posible mejora sería un cierre más grande o rellenar la concavidad de cada célula (envolvente convexa) antes de clasificar. No se aplicó para no cambiar el pipeline ya validado.
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
| Resultados: visualización, estadística, comparaciones, significancia | Figura de resultados (`figuras.py`: original, máscara, parasitadas y contornos por célula); AUC con p-valores de Mann-Whitney e intervalo de confianza por bootstrap; LOPO; tablas de sensibilidad; comparación de métodos (secciones 11 a 15) |
| Entrega del modelo, evaluado por la cátedra con casos incógnita | `predecir.py`: aplica el pipeline a imágenes nuevas sin GT (sección 2) |

## 18. Bibliografía

- Walliander, M., et al. (2013). *Automated segmentation of blood cells in Giemsa stained digitized thin blood films*. Diagnostic Pathology, 8(Suppl. 1), S37. https://doi.org/10.1186/1746-1596-8-S1-S37
- Dataset: NIH-NLM-ThinBloodSmearsPf (Polygon Set), National Library of Medicine.

*Verificar los datos bibliográficos antes de la entrega final.*
