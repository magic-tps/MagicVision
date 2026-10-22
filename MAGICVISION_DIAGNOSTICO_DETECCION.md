# Por qué MagicVision puede omitir objetos

Una escena puede contener personas, escaleras o tachos que la app no mencione. Para entender el problema hay que revisar tanto lo que aprendió el detector como la fotografía que recibió y la respuesta que llegó al teléfono.

## El modelo aprende de ejemplos

Un detector necesita imágenes variadas y bien etiquetadas. Si aprende con pocos ejemplos de un objeto, puede tener dificultades para reconocerlo desde otros ángulos, con poca luz o parcialmente oculto.

En el conjunto original había muy pocas imágenes de entrenamiento con escaleras. Esto puede contribuir a las omisiones, pero no demuestra por sí solo la causa de cada fallo. Los recuentos originales están en la [distribución del dataset](MAGICVISION_DISTRIBUCION_DATASET.csv).

## Hay objetos que no tenían una categoría propia

El detector de referencia reconoce diez categorías y no incluye una específica para tachos de basura. Que exista una categoría llamada «obstáculo» no significa que reconozca cualquier objeto que pueda estorbar el paso.

La ampliación a 22 categorías incorpora los tachos y otros elementos del campus. Para aprovecharla, hace falta completar y evaluar el entrenamiento, además de actualizar el servicio utilizado por la app.

## La fotografía y la conexión también influyen

Una foto borrosa, oscura o con un objeto fuera del encuadre puede dificultar el reconocimiento. También hay que comprobar que el servicio esté disponible y que la respuesta corresponda a la captura actual.

Una descripción más extensa no demuestra que se hayan reconocido todos los objetos. La ausencia de una advertencia tampoco confirma que el camino esté despejado.

## Cómo comprobar una mejora

Conviene probar escenas conocidas con personas, escaleras y tachos desde distintas posiciones. En cada caso, se compara la fotografía enviada, los objetos reconocidos y el mensaje que escucha la persona.

Las pruebas deben identificar qué versión de la app y del modelo se usó. Los resultados guardados de un entrenamiento anterior no bastan para afirmar que la versión instalada ya mejoró.

La [guía de mejora del reconocimiento](magicvision_ml_improvement/README.md) explica cómo se preparan y evalúan los nuevos modelos.
