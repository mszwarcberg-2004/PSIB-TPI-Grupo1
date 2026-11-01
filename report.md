
# Decisiones y avances del proyecto

## 1. Selección y organización del dataset

- Seleccionamos 16 imágenes del Polygon Set del dataset NIH-NLM-ThinBloodSmearsPf, correspondientes a 8 pacientes (2 imágenes por paciente).
- Elegimos imágenes que incluyeran eritrocitos parasitados y no parasitados, buscando variedad en la tinción y distribución celular.
- Dividimos el dataset por paciente: 6 pacientes para entrenamiento (12 imágenes) y 2 para prueba (4 imágenes), evitando que imágenes de un mismo paciente aparezcan en ambos conjuntos.
- El subconjunto contiene 240 eritrocitos parasitados y 3.111 no parasitados, según el Ground Truth. Existe un desbalance entre las clases que deberemos considerar en la evaluación.
- Decidimos trabajar inicialmente a resolución original para conservar los detalles celulares.

## 2. Uso y verificación del Ground Truth

- Implementamos funciones compartidas para cargar imágenes RGB, leer anotaciones GT y seleccionar imágenes según su conjunto de entrenamiento o prueba.
- Verificamos la cantidad de imágenes, pacientes y anotaciones, así como la correcta separación entre los conjuntos.
- Generamos visualizaciones de los polígonos GT superpuestos sobre las imágenes originales.
- Comprobamos visualmente que los contornos revisados coinciden con las células correspondientes.
- Definimos que el GT se utilizará como referencia para ajustar parámetros con entrenamiento y evaluar los resultados, pero no para realizar la detección automática de células.

## 3. Exclusión del borde negro del microscopio

- Observamos que las imágenes presentan un campo de visión circular rodeado por un borde negro.
- Decidimos excluir ese borde antes de segmentar, para evitar que se confunda con estructuras celulares oscuras.
- Implementamos una máscara del campo de visión mediante umbralización, selección de la región conectada más grande y relleno de huecos.
- Incorporamos una pequeña erosión para evitar posibles artefactos próximos al borde.
- Realizamos una primera prueba visual y obtuvimos una máscara circular adecuada para la imagen analizada.

## 4. Segmentación inicial mediante Otsu

- Elegimos implementar Otsu como primer método de segmentación y referencia para comparar posteriormente con K-means y GMM.
- Decidimos trabajar inicialmente sobre el canal verde de RGB, siguiendo el antecedente de Walliander et al. (2013), que utiliza este canal para segmentar células sanguíneas teñidas con Giemsa.
- La elección busca aprovechar el contraste entre los eritrocitos y el fondo claro, aunque su desempeño deberá comprobarse experimentalmente.
- Aplicamos un suavizado gaussiano previo para reducir pequeñas variaciones de intensidad que podrían afectar la umbralización.
- Calculamos el umbral de Otsu utilizando únicamente los píxeles del campo circular, evitando que el borde negro influya en el histograma.
- La segmentación inicial genera una máscara binaria de posibles eritrocitos, sin distinguir todavía entre parasitados y no parasitados.

## 5. Operaciones morfológicas posteriores

- Decidimos separar la segmentación inicial de las operaciones morfológicas, para poder analizar sus resultados individualmente.
- Incorporamos apertura para eliminar pequeñas regiones y conexiones finas, y cierre para corregir discontinuidades de la máscara.
- Observamos que los eritrocitos pueden presentar un centro más claro que su periferia. Por eso, la umbralización puede dejar huecos negros dentro de células que deberían estar completamente segmentadas.
- Incorporamos el relleno de huecos interiores pequeños para recuperar la región completa de cada eritrocito.
- Limitamos el relleno según el área del hueco, para que no se confunda el centro de los eritrocitos con el fondo y así evitar modificar excesivamente la segmentación.

## 6. Estrategia general del proyecto

Definimos dos procedimientos principales:

1. **Segmentación dirigida por color:** localizar regiones cromáticas compatibles con infección y segmentar los eritrocitos completos que las contienen.
2. **Segmentación general:** identificar todos los eritrocitos mediante Otsu, K-means o GMM, y posteriormente seleccionar los parasitados utilizando características de color, textura o ambas.

En ambos casos, el resultado buscado es una máscara de los eritrocitos parasitados completos, no una máscara de los parásitos en su interior.

También incorporamos la detección de glóbulos blancos mediante características de color y área, con el objetivo de generar una máscara de exclusión.

La comparación final se realizará utilizando los polígonos del Ground Truth como referencia común.

## 7. Próximos pasos

- Visualizar las máscaras iniciales de Otsu y compararlas con las obtenidas después de las operaciones morfológicas.
- Evaluar distintos parámetros de limpieza y comprobar que no se unan eritrocitos incorrectamente.
- Implementar y comparar K-means y GMM.
- Incorporar la detección y exclusión de glóbulos blancos.
- Aplicar Watershed para separar eritrocitos en contacto.
- Extraer células individuales para su posterior análisis de color y textura.

## Bibliografía de referencia

Walliander, M., et al. (2013). *Automated segmentation of blood cells in Giemsa stained digitized thin blood films*. Diagnostic Pathology, 8(Suppl. 1), S37.

https://doi.org/10.1186/1746-1596-8-S1-S37
