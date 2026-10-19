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

- **Histórico (2008–2026):** permite estudiar los perfiles observados a lo largo de la serie temporal.
- **Reciente (2020–2025):** permite examinar los cambios en los perfiles territoriales durante los últimos años analizados.

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


## 🚀 Cómo ejecutar el proyecto

### 1. **Clonar el repositorio**

   ```bash
   git clone https://github.com/TuUsuario/Spain-Labor-Cost-Predictor.git
   cd Spain-Labor-Cost-Predictor
   ```
Sustituye TuUsuario por tu nombre de usuario de GitHub y comprueba que el nombre del repositorio coincide con el publicado.

### 2. Instalar las dependencias

Se recomienda crear un entorno virtual para aislar las dependencias del proyecto:

```bash
python -m venv .venv
```

**Activar el entorno virtual en Windows:**

```bash
.venv\Scripts\activate
```

**Instalar las dependencias:**

```bash
pip install -r requirements.txt
```

### 3. **Configurar MySQL**

Asegúrate de disponer de un servidor MySQL en ejecución y configura los parámetros de conexión necesarios para el proyecto.

Las credenciales deben gestionarse mediante variables de entorno o un archivo de configuración local excluido del control de versiones. **No publiques contraseñas ni credenciales de acceso en GitHub**.


### 4. Ejecutar los pipelines

Primero, ejecuta el pipeline de preparación e integración de los datos:

```bash
python 01_data_pipeline.py
```

A continuación, ejecuta el pipeline de análisis y modelado:

```bash
python 02_ml_pipeline.py
```

La ejecución requiere que los archivos de datos estén disponibles en las rutas esperadas por el proyecto y que la conexión a MySQL esté correctamente configurada.



## 👨‍💻 Autor

 **Miguel Ángel Cumplido Pavón** 
 
 [LinkedIn](https://www.linkedin.com/in/miguelángelcumplidopavón)






