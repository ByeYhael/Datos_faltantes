# Análisis de Datos Faltantes, Atípicos y Cardinalidad en Conjuntos de Datos

## Introducción

El análisis de datos constituye una fase fundamental en cualquier proyecto de inteligencia artificial y aprendizaje automático. La calidad de los datos utilizados determina en gran medida el rendimiento y la validez de los modelos construidos. En este contexto, tres aspectos críticos requieren atención especial: los datos faltantes, los datos atípicos y la cardinalidad de las variables. Los datos faltantes representan una problemática frecuente en conjuntos de datos reales, surgen por diversas razones como errores de medición, fallas en sistemas de recolección o la negativa de participantes a proporcionar información. Los datos atípicos o outliers pueden distorsionar significativamente los resultados del análisis estadístico y afectar negativamente el entrenamiento de modelos predictivos. Por su parte, la cardinalidad alta en variables categóricas puede generar problemas de sobreajuste y aumentar la complejidad computacional. El presente informe tiene como objetivo aplicar técnicas de detección y tratamiento de datos faltantes, identificar valores atípicos mediante métodos estadísticos y analizar la cardinalidad de las variables en un conjunto de datos, siguiendo los procedimientos descritos en las fuentes bibliográficas consultadas.

## Marco Teórico

Los datos faltantes se definen como la ausencia de valores en ciertas observaciones dentro de un conjunto de datos. Según la literatura especializada, existen tres categorías principales de datos faltantes. La primera categoría es MCAR (Missing Completely At Random), donde los datos faltantes se distribuyen de manera completamente aleatoria y no existe relación con los valores de otras variables ni con los valores faltantes mismos. La segunda categoría es MAR (Missing At Random), en la cual la ausencia de datos depende de otras variables observadas en el conjunto de datos. La tercera categoría es MNAR (Missing Not At Random), donde los datos faltantes dependen del valor faltante mismo, lo cual representa el caso más problematico para el análisis.

Los datos atípicos o outliers son valores que se desvían significativamente del patrón general de los datos. Estos valores pueden originarse por errores de medición, entrada de datos incorrecta o fenómenos genuinos pero raros. Los métodos de detección incluyen el método del rango intercuartil (IQR), que considera como atípicos aquellos valores que se encuentran por debajo de Q1 - 1.5*IQR o por encima de Q3 + 1.5*IQR. Otro método común es el Z-Score, que identifica como atípicos aquellos valores con un puntaje Z mayor a 3 o menor a -3, indicando que el valor se encuentra a más de tres desviaciones estándar de la media.

La cardinalidad se refiere al número de valores únicos que puede tomar una variable. Una cardinalidad alta indica que una variable categórica tiene muchos valores únicos, mientras que una cardinalidad baja indica pocos valores únicos. La cardinalidad elevada puede generar problemas en la construcción de modelos, especialmente cuando se utilizan técnicas de one-hot encoding, ya que esto incrementa dramáticamente la dimensionalidad del conjunto de datos. Según las presentaciones de clase consultadas, es fundamental analizar la cardinalidad antes de aplicar cualquier transformación a los datos.

## Materiales y Métodos

Para el desarrollo de este análisis se utilizó el lenguaje de programación Python con las librerías pandas para la manipulación de datos y scikit-learn para las técnicas de imputación. Las técnicas aplicadas incluyeron el análisis de valores nulos mediante la función isnull().sum() de pandas, que permite identificar la cantidad de datos faltantes por cada columna del conjunto de datos. Para la imputación de datos numéricos se utilizó la técnica de imputación por la media, reemplazando los valores faltantes con el promedio de la columna correspondiente. Para variables categóricas se aplicó la imputación por la moda, utilizando el valor más frecuente.

La detección de datos atípicos se realizó mediante el método del rango intercuartil (IQR), calculando los percentiles 25 y 75 para cada variable numérica y aplicando la fórmula establecida. Adicionalmente, se consideró el uso del Z-Score para verificar la presencia de valores extremos. El análisis de cardinalidad se llevó a cabo mediante la función nunique(), que devuelve el número de valores únicos en cada columna, permitiendo identificar variables con alta cardinalidad especial durante la preparación de los que podrían requerir tratamiento datos.

La metodología seguida fue la siguiente: primero se cargó el conjunto de datos y se realizó una exploración inicial; posteriormente se identificaron los datos faltantes y su porcentaje por variable; a continuación se aplicaron las técnicas de imputación correspondientes; finalmente se detectaron los datos atípicos y se analizó la cardinalidad de todas las variables.

## Resultados y Discusión

El análisis cuantitativo de datos faltantes reveló la presencia de valores nulos en múltiples columnas del conjunto de datos. La proporción de datos faltantes varió según la variable, siendo algunas columnas más afectadas que otras. La identificación del tipo de datos faltantes requiere un análisis más profundo para determinar si corresponden a MCAR, MAR o MNAR. En el caso de datos MCAR, la imputación por media o moda resulta apropiada; para datos MAR, técnicas más sofisticadas como la imputación múltiple pueden ser necesarias; mientras que los datos MNAR requieren un tratamiento especial que considere el mecanismo de ausencia.

La detección de datos atípicos mediante el método IQR permitió identificar valores que se desvían significativamente de la distribución normal de los datos. Los resultados muestran que algunas variables presentan outliers tanto en los extremos inferiores como superiores de su distribución. Es importante verificar si estos valores atípicos representan errores genuinos que deben ser eliminados o corregidos, o si corresponden a fenómenos reales que deben ser conservados pero tratados con cautela durante el modelado. La presencia de outliers puede afectar significativamente estadísticas descriptivas como la media y la desviación estándar, distorsionando la interpretación de los resultados.

El análisis de cardinalidad evidenció que algunas variables categóricas presentan un número elevado de valores únicos. Las variables con cardinalidad alta requieren atención especial durante la fase de preprocesamiento, ya que pueden generar problemas de dimensionalidad al aplicar técnicas de codificación como one-hot encoding. Por otro lado, las variables con cardinalidad muy baja, incluso aquellas con un único valor, pueden ser candidatas para eliminación si no aportan información discriminativa al modelo. El balance entre preservar la información relevante y evitar la proliferación de características es un aspecto crítico que debe considerarse en todo proyecto de análisis de datos.

## Conclusión

El análisis de datos faltantes, atípicos y la evaluación de la cardinalidad constituyen pasos fundamentales en la preparación de datos para proyectos de inteligencia artificial. Los resultados obtenidos permiten identificar las variables que requieren tratamiento previo antes de construir modelos predictivos. Se recomienda documentar las decisiones tomadas respecto al manejo de datos faltantes y atípicos, considerando siempre el impacto de estas decisiones en los resultados finales del análisis.

## Bibliografía

Castro, G. (2024). *De 0 a 100 en Inteligencia Artificial*. Editorial unspecified.

Castro, G. (2024). Presentación 1: Datos Faltantes. Material de clase de Aprendizaje Automático, Universidad Autónoma de Querétaro.

Castro, G. (2024). Presentación 2: Datos Atípicos. Material de clase de Aprendizaje Automático, Universidad Autónoma de Querétaro.

Castro, G. (2024). Presentación 3: Cardinalidad. Material de clase de Aprendizaje Automático, Universidad Autónoma de Querétaro.

Géron, A. (2019). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (2nd ed.). O'Reilly Media.