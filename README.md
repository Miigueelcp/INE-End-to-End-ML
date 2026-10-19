# 📊 Análisis y Predicción del Mercado Laboral en España (INE)

> **Pipeline End-to-End para analizar y predecir el coste laboral en España mediante SQL, Machine Learning, clustering y explicabilidad con SHAP.**

## 📌 Descripción del proyecto

Este proyecto desarrolla un flujo de trabajo completo de **Data Science** a partir de datos oficiales del Instituto Nacional de Estadística (INE), con el objetivo de analizar la evolución del coste laboral por trabajador en España, estudiar las diferencias entre Comunidades Autónomas y sectores económicos, e identificar patrones en el mercado laboral.

Se integran datos del coste laboral y de las tasas de actividad, paro y empleo de la Encuesta de Población Activa (EPA). A partir de estos datos, se construye un proceso que abarca desde la extracción, limpieza e integración de la información mediante SQL hasta el análisis exploratorio, el entrenamiento de modelos predictivos y la interpretación de sus resultados.

El proyecto combina **Machine Learning supervisado y no supervisado**, utilizando modelos de regresión para predecir el coste laboral, K-Means para identificar perfiles territoriales y SHAP para interpretar las predicciones del modelo seleccionado.

## 🛠️ Tecnologías y Herramientas

* **Lenguaje:** Python 3.x
* **Manipulación de Datos:** Pandas, NumPy
* **Base de Datos:** MySQL / SQLAlchemy
* **Machine Learning:** Scikit-Learn, XGBoost
* **Explicabilidad (XAI):** SHAP
* **Visualización:** Matplotlib, Seaborn, AdjustText
* **Análisis estadístico**: correlaciones, análisis de distribuciones y detección de valores atípicos.

## 📁 Arquitectura del Proyecto

El código se organiza en módulos para separar la preparación de datos, el análisis exploratorio, el modelado y la visualización.

```text
├── 01_data_pipeline.py       # Pipeline de extracción, transformación y carga (ETL)
├── 02_ml_pipeline.py         # Pipeline de entrenamiento, evaluación y clustering
├── .gitignore                # Archivos excluidos del control de versiones
├── requirements.txt          # Dependencias del proyecto
└── src/
    ├── __init__.py
    ├── database.py           # Conexión y operaciones con MySQL
    ├── data_loader.py        # Carga de los datos originales
    ├── cleaning.py           # Limpieza y transformación de datos
    ├── eda.py                # Análisis exploratorio de datos
    ├── ml_preprocessing.py   # Preparación de variables para los modelos
    ├── ml_models.py          # Entrenamiento y predicción
    ├── ml_clustering.py      # Segmentación mediante K-Means
    └── visualizations.py     # Gráficos y análisis SHAP
```

## 🧠 Fases del Proyecto

### 1. **Ingeniería de Datos y SQL (ETL)**

   - Carga de los conjuntos de datos originales del INE.
   - Limpieza y normalización de nombres, categorías y variables.
   - Transformación de las fechas en variables temporales y preparación de los datos para su análisis.
   - Integración de las tablas mediante MySQL y SQLAlchemy.
   - Extracción de los datos integrados a Python y transformación al formato adecuado para el modelado.

### 2. **Análisis exploratorio de datos (EDA)**

Se realiza un análisis exploratorio para estudiar la evolución temporal del coste laboral, las diferencias territoriales y sectoriales, y su relación con los principales indicadores del mercado laboral.

El análisis incluye:

- Comparación del coste laboral entre Comunidades Autónomas y sectores económicos.
- Estudio de las subvenciones y bonificaciones de la Seguridad Social.
- Análisis de distribuciones y valores atípicos.
- Matrices de correlación de Pearson entre las variables del conjunto de datos.

Los valores atípicos se mantienen cuando se consideran compatibles con observaciones económicas reales, en lugar de eliminarlos automáticamente.

### 3. **Modelado predictivo supervisado**

Se comparan distintos modelos de regresión para predecir el coste laboral por trabajador:

- Regresión lineal.
- Random Forest.
- XGBoost.

**XGBoost se selecciona como modelo final** por su capacidad para representar relaciones no lineales, de acuerdo con los resultados obtenidos en la evaluación.

Para respetar el orden temporal de los datos, se emplea una división basada en el tiempo (time-based split), utilizando información hasta 2024 para el entrenamiento. Asimismo, se realizan proyecciones futuras teniendo en cuenta las variaciones estacionales observadas históricamente.

### 4. **Explicabilidad del modelo (SHAP)**

Se utilizan valores SHAP *(SHapley Additive exPlanations)* para interpretar las predicciones de XGBoost e identificar qué variables tienen mayor influencia en las estimaciones.

Se emplean gráficos *Beeswarm* y de importancia mediante barras para analizar la contribución de las variables y comprender mejor el comportamiento del modelo.

### 5. **Aprendizaje no supervisado: K-Means**

Se aplica K-Means para identificar grupos de Comunidades Autónomas y sectores con características similares en términos de coste laboral y tasa de paro.

Se comparan dos periodos:

- **Histórico (2008–2026)**: permite estudiar los perfiles observados a lo largo de la serie temporal.
- **Reciente (2020–2025)**: permite examinar los cambios en los perfiles territoriales durante los últimos años analizados.

Esta comparación ayuda a identificar similitudes y diferencias en la evolución del mercado laboral, así como cambios en la agrupación de las regiones y los sectores.

## 💡 Principales resultados e insights

### 📈 Evolución del coste laboral
Entre 2008 y 2025, el coste salarial experimentó un incremento acumulado aproximado del **48 %**, según los datos analizados. La evolución temporal presenta una asociación positiva con el coste laboral, reflejando su incremento sostenido a lo largo del periodo.

Este comportamiento puede estar relacionado con diversos factores económicos, como la inflación, las actualizaciones del salario mínimo interprofesional y los cambios en la composición del empleo. El análisis observacional, por sí solo, no permite atribuir el incremento a una causa concreta.

### 🗺️ Diferencias territoriales

La Comunidad de Madrid, el País Vasco y Cataluña destacan por sus mayores niveles de coste laboral en los análisis realizados.

Estas diferencias ponen de manifiesto la heterogeneidad territorial del mercado laboral español, aunque los resultados no permiten atribuirlas a un único factor económico o social.

### 🏭 Diferencias sectoriales

La industria presenta, en los datos analizados, costes laborales medios superiores a los de los servicios y la construcción.

Este resultado evidencia la importancia de considerar el sector económico al comparar los costes laborales entre territorios

### 🔎 Perfiles territoriales mediante K-Means

El análisis histórico identifica cuatro perfiles de agrupación, entre los que destaca un grupo de regiones como Andalucía y Extremadura, caracterizadas por tasas de paro relativamente elevadas y costes laborales más bajos.

En el periodo reciente (2020–2025), estas regiones continúan situándose en niveles de coste inferiores, mientras que sus tasas de desempleo muestran una evolución más favorable. La comparación permite estudiar cómo cambian los perfiles identificados entre ambos periodos.

### 💶 Subvenciones y bonificaciones

El análisis por trabajador y sector identifica diferencias en las subvenciones y bonificaciones de la Seguridad Social entre Comunidades Autónomas. Entre las regiones destacadas en los resultados se encuentran el País Vasco, Asturias, Aragón y Andalucía.

También se observa un máximo aproximado de **27 € por trabajador en el sector industrial en 2011**. Estos resultados describen la evolución de las cuantías registradas por trabajador, no el importe total de las ayudas recibidas por cada comunidad.


### 🔗 Relaciones entre las variables

El análisis de correlaciones de Pearson muestra las siguientes relaciones:

- **Tasa de empleo y tasa de paro:** presentan una asociación negativa marcada.
- **Tasa de actividad y tasa de empleo:** muestran una asociación positiva.
- **Coste laboral y tasa de paro:** presentan una relación negativa en el conjunto analizado.
- **Coste laboral y tasa de empleo:** muestran una relación positiva.
- **Año y coste laboral:** presentan una asociación positiva, coherente con el incremento observado a lo largo de la serie.
- **Trimestre:** muestra correlaciones lineales débiles con las variables analizadas, con un coeficiente máximo aproximado de 0,13 para el coste laboral.

Estas relaciones son descriptivas y no implican causalidad. Además, la correlación lineal débil del trimestre no descarta la existencia de patrones estacionales o relaciones no lineales, que pueden estudiarse mediante modelos predictivos.






- **Efecto Inflacionario:** La evolución histórica (2008-2025) muestra como la inflación ha tenido peso durante los ultimos años, donde desde el 2008 hasta el 2025 el coste salarial ha subido hasta un un 48 %, donde desde la crisis del COVID-19 ha sido cuando ha subido más el coste salarial y por tanto la inflación. El análisis SHAP revela que el **Año** es el factor de mayor peso en la predicción, confirmando el impacto de la inflación estructural y las subidas del SMI en la última década.
- **Brecha Territorial:** Comunidades como la Comunidad de Madrid, País Vasco y Cataluña tiran fuertemente al alza del coste laboral, reflejando una concentración de la riqueza y el empleo cualificado.
- **El Motor Industrial:** A nivel sectorial, la Industria es el sector que presenta un mayor impacto positivo sobre los costes laborales, distanciándose notablemente del sector Servicios y Construcción.
- **Evolución del Mapa Laboral (K-Means):** El clustering histórico (2008-2026) identificó 4 perfiles claros, destacando un grupo de regiones (como Andalucía o Extremadura) con altas tasas de paro y bajo coste. Sin embargo, el análisis del periodo reciente (2020-2025) muestra que, aunque estas CCAA siguen en el rango inferior de costes, han logrado reducir sustancialmente sus tasas de desempleo, haciendo desaparecer el clúster histórico de "alta crisis".
- **Subvenciones**: CCAA como País Vasco, Asturias, Aragón y Andalucía son las que más euros por trabajador reciben de subvenciones del Estado. También vemos como en los años donde la crisis financiera global es cuando más subvención por trabajador ha habido, con un pico de 27 euros en el sector de la industria en el año 2011. Por otro lado, vemos como los recortes sufridos en España muestran como estas subvenciones fueron aminorando hasta con las entradas de neuvos gobiernos haciendo eque estas crezcan.
- **Correlaciones**: La estructua de mercado nos muestra como la una fuerte tasa de empleo está ligada a una mucha menor tasa de paro. Así mismo, una mayot tasa de actividad esta fuertemente ligada a una mayot tasa de empleo. El coste laboral tiene una correlación negativa con la tasa de paro y positiva con la tasa de empleo, mostrando como los mercados laborales tensionados (alta ocupación y por tanto bajo paro) presionan los costes salariales al alza, mientras que los picos de desempleo tienden a contenerlos o reducirlos. La variable año  nos muestra una correlación positiva sólida con el coste laboral, esto nos muestra el encarecimeinto sostenido de lso lsalarios a lo largo de la serie histórica, impulsado por la inflación estructural y las actualzaciones del SMI. Paralelamente, el avance temporal regleja la recuperaicón macroeconómica general, correlacionando negativamente con el paro y postiivamente con el empleo. FInalmente la vairbale trimestre presenta coeficientes casi neturos en todas las métricas, siendo el más alto apenas con un 0.13 el coste laboral. Esto indica que la estacionalidad intra-anulao no sigue un patrón de correlación puramente lineal, lo que justifica perfectamente la elección metodolófgica de usar XGBOOST para detectar variaciones no lineales.

## 🚀 Cómo ejecutar el proyecto

1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/TuUsuario/Spain-Labor-Cost-Predictor.git
   cd Spain-Labor-Cost-Predictor
   ```

2. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configuración de Base de Datos:**

   Asegúrate de tener un servidor MySQL en ejecución y configura las credenciales (o variables de entorno) necesarias en el módulo `src/database.py`.

4. **Ejecutar pipelines:**
   - Primero procesa los datos crudos y súbelos a la base de datos:
     ```bash
     python 01_data_pipeline.py
     ```
   - Después entrena los modelos y genera las visualizaciones:
     ```bash
     python 02_ml_pipeline.py
     ```
  
*Desarrollado por Miguel Ángel Cumplido Pavón - [LinkedIn](https://www.linkedin.com/in/miguelángelcumplidopavón)*

