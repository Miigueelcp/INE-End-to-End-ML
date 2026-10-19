# 📊 Análisis y Predicción del Mercado Laboral en España (INE)

> **Pipeline End-to-End de Data Science para analizar y predecir el coste laboral en España mediante SQL, Machine Learning, clustering y explicabilidad con SHAP.**

Este proyecto es una solución integral de Machine Learning diseñada para analizar y predecir el coste laboral por trabajador en España (por ambos sexos), y a su vez, analizar qué Comunidades Autónomas tienen mayor semejanza en relación con las tasas de paro y el coste salarial. Se utilizan datos oficiales macroeconómicos del Instituto Nacional de Estadística (INE). 

El análisis abarca desde la extracción en bases de datos (SQL) hasta el despliegue de modelos predictivos y técnicas de explicabilidad (XAI).

## 🛠️ Tecnologías y Herramientas

* **Lenguaje:** Python 3.x
* **Manipulación de Datos:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn, XGBoost
* **Explicabilidad (XAI):** SHAP
* **Visualización:** Matplotlib, Seaborn, AdjustText
* **Base de Datos:** MySQL / SQLAlchemy

## 📁 Arquitectura del Proyecto

El código está estructurado siguiendo principios de modularidad y MLOps:

```text
├── 01_data_pipeline.py       # Orquestador del flujo ETL (Extracción, Limpieza y Carga a SQL)
├── 02_ml_pipeline.py         # Orquestador del entrenamiento, evaluación y clustering
├── .gitignore                # Archivos excluidos del control de versiones
├── requirements.txt          # Dependencias del proyecto
└── src/                      # Módulos core de la aplicación
    ├── __init__.py
    ├── database.py           # Conexión y operaciones con MySQL
    ├── data_loader.py        # Carga de archivos crudos
    ├── eda.py                # Exploración de datos base
    ├── cleaning.py           # Lógica de limpieza
    ├── ml_preprocessing.py   # Feature Engineering (One-Hot, StandardScaler)
    ├── ml_models.py          # Entrenamiento supervisado (XGBoost, Random Forest) y proyecciones
    ├── ml_clustering.py      # Aprendizaje no supervisado (K-Means)
    └── visualizations.py     # Gráficos, distribuciones y explicabilidad SHAP
```

## 🧠 Fases del Proyecto

1. **Ingeniería de Datos y SQL (ETL):** Carga inicial, limpieza y exploración gráfica de las encuestas (EPA y Costes Laborales). Integración de los dataframes en SQL y posterior extracción a Python, pivotando las variables a un formato *wide* optimizado para el modelado.

2. **Análisis Exploratorio Integrado (EDA):** Evaluación visual y estadística del dataset consolidado para extraer conclusiones preliminares. Incluye análisis de distribuciones, correlaciones de Pearson y detección de valores atípicos macroeconómicos (mantenidos de forma intencionada por ser representativos de crisis y picos económicos reales). 

3. **Modelado Predictivo Supervisado:** Predicción del coste laboral evaluando modelos de Regresión Lineal, Random Forest y **XGBoost** (seleccionado como el mejor por su capacidad para captar relaciones no lineales). Se aplicó una división temporal (*Time-based split*) entrenando hasta 2024 para evitar *Data Leakage*, y se proyectaron costes futuros aplicando deltas de estacionalidad histórica.

4. **Explicabilidad de Modelos (SHAP):** Apertura de la "caja negra" del modelo XGBoost. Utilización de valores SHAP (gráficos *Beeswarm* y de barras) para interpretar con precisión el impacto y el peso de cada variable macroeconómica en las predicciones finales.

5. **Aprendizaje No Supervisado (K-Means):** Aplicación de técnicas de *clustering* para segmentar y analizar las similitudes en la relación entre la tasa de paro y el coste salarial. Se contrasta la evolución histórica con la reciente para perfilar el comportamiento de las diferentes Comunidades Autónomas y sectores económicos.

## 💡 Conclusiones Clave y Business Insights

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

