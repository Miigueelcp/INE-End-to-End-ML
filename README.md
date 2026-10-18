# 📊 Predicción de Costes Laborales en España (INE)

> **Pipeline End-to-End aplicando XGBoost, clustering (K-Means) y explicabilidad (SHAP) sobre datos del INE para descubrir el impacto de la inflación y las diferencias regionales.**

Este proyecto es una solución integral de Machine Learning diseñada para analizar y predecir el coste laboral por trabajador en España, utilizando datos oficiales macroeconómicos del Instituto Nacional de Estadística (INE).

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
```

## 🧠 Fases del Proyecto

1. **Ingeniería de Datos (ETL):** Integración de encuestas de costes laborales y Encuesta de Población Activa (EPA). Pivotado de variables para crear un dataset de tipo *Wide* listo para modelado.

2. **Análisis Exploratorio (EDA):** Evaluación de distribuciones, correlaciones de Pearson y detección de valores atípicos macroeconómicos (los cuales se mantienen por ser representativos de crisis y picos económicos).

3. **Modelado Predictivo Supervisado:**
   - División temporal (*Time-based split*) entrenando hasta 2024 y evaluando en 2025/2026 para evitar *Data Leakage*.
   - Evaluación de Regresión Lineal, Random Forest y **XGBoost** (mejor modelo por su capacidad para captar relaciones no lineales).
   - Proyección de costes laborales futuros aplicando deltas de estacionalidad histórica.

4. **Explicabilidad con SHAP:** Apertura de la "caja negra" del modelo XGBoost para entender el peso de cada variable en la predicción final (Beeswarm y gráficos de barras).

5. **Aprendizaje No Supervisado (K-Means):** Análisis de la relación histórica y reciente entre la Tasa de Paro y el Coste Laboral por Comunidad Autónoma y sector económico.

## 💡 Conclusiones Clave y Business Insights

- **Efecto Inflacionario:** El análisis SHAP revela que el **Año** es el factor de mayor peso en la predicción, confirmando el impacto de la inflación estructural y las subidas del SMI en la última década.
- **Brecha Territorial:** Comunidades como la Comunidad de Madrid, País Vasco y Cataluña tiran fuertemente al alza del coste laboral, reflejando una concentración de la riqueza y el empleo cualificado.
- **El Motor Industrial:** A nivel sectorial, la Industria es el sector que presenta un mayor impacto positivo sobre los costes laborales, distanciándose notablemente del sector Servicios y Construcción.
- **Evolución del Mapa Laboral (K-Means):** El clustering histórico (2008-2026) identificó 4 perfiles claros, destacando un grupo de regiones (como Andalucía o Extremadura) con altas tasas de paro y bajo coste. Sin embargo, el análisis del periodo reciente (2020-2025) muestra que, aunque estas CCAA siguen en el rango inferior de costes, han logrado reducir sustancialmente sus tasas de desempleo, haciendo desaparecer el clúster histórico de "alta crisis".

## 🚀 Cómo ejecutar el proyecto

1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/TuUsuario/Spain-Labor-Cost-Predictor.git
   cd Spain-Labor-Cost-Predictor
    ├── ml_clustering.py      # Aprendizaje no supervisado (K-Means)
    └── visualizations.py     # Gráficos, distribuciones y explicabilidad SHAP

2. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt

3. **Configuración de Base de Datos:**
   
   Asegurate de teneer un servidor MySQL en ejecución y configura las credenciales (o variables de entorno) necesarias en el módulo src/database.py

4. **Ejecutar pipelines**
   - Primero procesa los datos crudos y súbelos a la base de datos: python 01_data_pipeline.py
   - Después entrena los modelos y genera las visualizaciones: python 02_ml_pipeline.py
  
Desarrollado por Miguel Ángel Cumplido Pavón - www.linkedin.com/in/miguelángelcumplidopavón

