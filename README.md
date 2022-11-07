# RN-VT
Redes Neuronales - Vision Transformers
### Introduccion ###

En 2022, Vision Transformer (ViT) surgió como una alternativa competitiva a las redes neuronales convolucionales (CNN) que actualmente son lo último en visión por computadora y, por lo tanto, se utilizan ampliamente en diferentes tareas de reconocimiento de imágenes. Los modelos de ViT superan al actual estado del arte (CNN) en casi x4 en términos de eficiencia y precisión computacional.  

Los modelos de transformadores se han convertido en el status quo de facto en el procesamiento del lenguaje natural (NLP). En la investigación de la visión artificial, recientemente ha habido un aumento en el interés por los transformadores de visión (ViT) y los perceptrones multicapa (MLP).  

Este artículo cubrirá los siguientes temas:  
*  ¿Qué es un transformador de visión (ViT)?
Lea más en: https://viso.ai/deep-learning/vision-transformer-vit/
* Uso de modelos ViT en reconocimiento de imágenes
Lea más en: https://viso.ai/deep-learning/vision-transformer-vit/
* ¿Cómo funcionan los transformadores de visión?
Lea más en: https://viso.ai/deep-learning/vision-transformer-vit/
* Casos de uso y aplicaciones de Vision Transformers
Lea más en: https://viso.ai/deep-learning/vision-transformer-vit/

### Transformador de visión (ViT) en reconocimiento de imágenes ###

Si bien la arquitectura Transformer se ha convertido en el estándar más alto para tareas que involucran procesamiento de lenguaje natural (NLP), sus casos de uso relacionados con Computer Vision (CV) siguen siendo solo algunos. En la visión por computadora, la atención se usa junto con las redes convolucionales (CNN) o se usa para sustituir ciertos aspectos de las redes convolucionales mientras se mantiene intacta toda su composición. Los algoritmos populares de reconocimiento de imágenes incluyen ResNet, VGG, YOLOv3 y YOLOv7.  

Sin embargo, esta dependencia de CNN no es obligatoria, y un transformador puro aplicado directamente a secuencias de parches de imágenes puede funcionar excepcionalmente bien en tareas de clasificación de imágenes.  

Recientemente, Vision Transformers (ViT) ha logrado un rendimiento altamente competitivo en los puntos de referencia para varias aplicaciones de visión artificial, como la clasificación de imágenes, la detección de objetos y la segmentación semántica de imágenes.  

### ¿Qué es un transformador de visión (ViT)? ###

El modelo Vision Transformer (ViT) se presentó en un artículo de investigación publicado como documento de conferencia en ICLR 2021 titulado "Una imagen vale 16 * 16 palabras: transformadores para el reconocimiento de imágenes a escala". Fue desarrollado y publicado por Neil Houlsby, Alexey Dosovitskiy y 10 autores más del Google Research Brain Team.  

El código de ajuste fino y los modelos ViT preentrenados están disponibles en GitHub de Google Research. Los encuentras aquí. Los modelos ViT se entrenaron previamente en los conjuntos de datos ImageNet e ImageNet-21k.  

### ¿Son los Transformers un método de Deep Learning? ###

Un transformador en el aprendizaje automático es un modelo de aprendizaje profundo que utiliza los mecanismos de atención, sopesando diferencialmente la importancia de cada parte de los datos de entrada. Los transformadores en el aprendizaje automático se componen de múltiples capas de autoatención. Se utilizan principalmente en los subcampos de IA de procesamiento de lenguaje natural (NLP) y visión artificial (CV).  

Los transformadores en el aprendizaje automático son fuertes promesas hacia un método de aprendizaje genérico que se puede aplicar a varias modalidades de datos, incluidos los avances recientes en la visión artificial que logran una precisión estándar de última generación con una mejor eficiencia de los parámetros.  

### Diferencia entre CNN y ViT (ViT vs. CNN) ###

Vision Transformer (ViT) logra resultados notables en comparación con las redes neuronales convolucionales (CNN) al tiempo que obtiene menos recursos computacionales para el entrenamiento previo. En comparación con las redes neuronales convolucionales (CNN), Vision Transformer (ViT) muestra un sesgo inductivo generalmente más débil que da como resultado una mayor dependencia de la regularización del modelo o el aumento de datos (AugReg) cuando se entrena en conjuntos de datos más pequeños.  

El ViT es un modelo visual basado en la arquitectura de un transformador diseñado originalmente para tareas basadas en texto. El modelo ViT representa una imagen de entrada como una serie de parches de imagen, como la serie de incrustaciones de palabras que se usan cuando se usan transformadores de texto, y predice directamente las etiquetas de clase para la imagen. ViT exhibe un rendimiento extraordinario cuando se entrena con suficientes datos, superando el rendimiento de una CNN de última generación similar con 4 veces menos recursos computacionales.   

Estos transformadores tienen altas tasas de éxito cuando se trata de modelos NLP y ahora también se aplican a imágenes para tareas de reconocimiento de imágenes. CNN usa matrices de píxeles, mientras que ViT divide las imágenes en tokens visuales. El transformador visual divide una imagen en parches de tamaño fijo, incrusta correctamente cada uno de ellos e incluye la incrustación posicional como entrada al codificador del transformador.  

Además, los modelos ViT superan a las CNN en casi cuatro veces en lo que respecta a la eficiencia y precisión computacional. La capa de autoatención en ViT hace posible incrustar información globalmente en la imagen general. El modelo también aprende datos de entrenamiento para codificar la ubicación relativa de los parches de imagen para reconstruir la estructura de la imagen.  

El codificador de transformador incluye:
* Capa de autoatención de varios cabezales (MSP): esta capa concatena todas las salidas de atención de forma lineal en las dimensiones correctas. Los muchos cabezales de atención ayudan a entrenar dependencias locales y globales en una imagen.    
* Capa de perceptrones multicapa (MLP): esta capa contiene una unidad lineal de error gaussiano (GELU) de dos capas.  
* Norma de capa (LN): se agrega antes de cada bloque, ya que no incluye nuevas dependencias entre las imágenes de entrenamiento. Esto ayuda a mejorar el tiempo de entrenamiento y el rendimiento general.  

Además, las conexiones residuales se incluyen después de cada bloque, ya que permiten que los componentes fluyan a través de la red directamente sin pasar por activaciones no lineales.  

En el caso de la clasificación de imágenes, la capa MLP implementa el cabezal de clasificación. Lo hace con una capa oculta en el momento del entrenamiento previo y una única capa lineal para el ajuste fino.  

![attention-map-vision-transformers-vit-1060x558](https://user-images.githubusercontent.com/95035101/200313816-833f1fe1-e820-4f19-9138-4745e0129359.jpg)

### ¿Qué son los mapas de atención de ViT? ###

La atención, más específicamente, la autoatención es uno de los bloques esenciales de los transformadores de aprendizaje automático. Es una primitiva computacional utilizada para cuantificar las interacciones de entidades por pares que ayudan a una red a aprender las jerarquías y alineaciones presentes dentro de los datos de entrada. La atención ha demostrado ser un elemento clave para que las redes de visión logren una mayor robustez.  

![confidence-and-attention-of-vision-transformers-vit](https://user-images.githubusercontent.com/95035101/200314001-0d6f05c3-506d-4a77-aa3d-ef6a64b08e15.jpg)

### Transformador de visión Arquitectura ViT ###

La arquitectura general del modelo de transformador de visión se proporciona paso a paso de la siguiente manera:  

1. Dividir una imagen en parches (tamaños fijos)
2. Aplanar los parches de imagen
3. Cree incrustaciones lineales de menor dimensión a partir de estos parches de imagen aplanados
4. Incluir incrustaciones posicionales
5. Alimente la secuencia como entrada a un codificador de transformador de última generación
6. Entrene previamente el modelo ViT con etiquetas de imagen, que luego se supervisa completamente en un gran conjunto de datos
7. Ajuste el conjunto de datos aguas abajo para la clasificación de imágenes

![vision-transformer-vit](https://user-images.githubusercontent.com/95035101/200314624-58894603-9cfa-4ca8-9ee5-ea0008dd2da3.jpg)

Si bien la arquitectura de transformador completo de ViT es una opción prometedora para las tareas de procesamiento de visión, el rendimiento de ViT sigue siendo inferior al de las alternativas de CNN de tamaño similar (como ResNet) cuando se entrena desde cero en un conjunto de datos de tamaño mediano como ImageNet.  

![vit-vision-transformers-performance-2021-1060x470](https://user-images.githubusercontent.com/95035101/200314779-b4619ceb-13f8-4df8-bc1c-11532575ff75.jpg)

### ¿Cómo funciona un transformador de visión (ViT)? ###

El rendimiento de un modelo de transformador de visión depende de decisiones como las del optimizador, la profundidad de la red y los hiperparámetros específicos del conjunto de datos. En comparación con ViT, las CNN son más fáciles de optimizar. La disparidad en un transformador puro es casar un transformador con un front-end de CNN. El vástago ViT habitual aprovecha una circunvolución de 16 * 16 con un paso de 16. En comparación, una convolución de 3*3 con zancada 2 aumenta la estabilidad y eleva la precisión. CNN convierte píxeles básicos en un mapa de características. Más tarde, un tokenizador convierte el mapa de características en una secuencia de tokens que luego se ingresan en el transformador.  

Luego, el transformador aplica la técnica de atención para crear una secuencia de tokens de salida. Eventualmente, un proyector vuelve a conectar los tokens de salida al mapa de características. Este último permite que el examen navegue por detalles potencialmente cruciales a nivel de píxel. Por lo tanto, esto reduce la cantidad de tokens que deben estudiarse, lo que reduce significativamente los costos. En particular, si el modelo ViT se entrena en grandes conjuntos de datos que tienen más de 14 millones de imágenes, puede superar a las CNN. Si no, la mejor opción es quedarse con ResNet o EfficientNet.  

El modelo de transformador de visión se entrena en un gran conjunto de datos incluso antes del proceso de ajuste. El único cambio es ignorar la capa MLP y agregar una nueva capa D multiplicada por KD*K, donde K es el número de clases del pequeño conjunto de datos. Para afinar en mejores resoluciones, se realiza la representación 2D de las incrustaciones de posición preentrenadas. Esto se debe a que las capas de revestimiento entrenables modelan las incrustaciones posicionales.

### Casos de uso y aplicaciones del transformador de visión del mundo real (ViT) ###

Los transformadores de visión tienen amplias aplicaciones en tareas populares de reconocimiento de imágenes, como detección de objetos, segmentación, clasificación de imágenes y reconocimiento de acciones. Además, los ViT se aplican en el modelado generativo y tareas de modelos múltiples, que incluyen la conexión a tierra visual, la respuesta a preguntas visuales y el razonamiento visual.  

El pronóstico de video y el reconocimiento de actividad son partes del procesamiento de video que requieren ViT. Además, la mejora de imágenes, la colorización y la superresolución de imágenes también utilizan modelos ViT. Por último, pero no menos importante, los ViT tienen numerosas aplicaciones en el análisis 3D, como la segmentación y la clasificación de nubes de puntos.

### Conclusion ###

El modelo de transformador de visión utiliza la autoatención de varios cabezales en Computer Vision sin requerir sesgos específicos de imagen. El modelo divide las imágenes en una serie de parches de inserción posicionales, que son procesados ​​por el codificador del transformador. Lo hace para comprender las características locales y globales que posee la imagen. Por último, pero no menos importante, ViT tiene una tasa de precisión más alta en un gran conjunto de datos con un tiempo de entrenamiento reducido.



