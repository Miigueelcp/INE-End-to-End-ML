import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def entrenar_y_evaluar_modelos(df_integrado):
    """
    Entrena y evalúa varios modelos de regresión para estimar el coste laboral.
    Utiliza una división temporal (Train: <=2024, Test: >=2025).
    """
    print("\n--- Iniciando Entrenamiento de Modelos de Machine Learning ---")
    
    df_ml = df_integrado.copy()

    variables_numericas = ['ano', 'trimestre', 'tasa_actividad', 'tasa_empleo', 'tasa_paro']
    variables_categoricas = ['comunidades_autonomas', 'sectores_actividad']

    # Ordenamos temporalmente para evitar Data Leakage
    df_ml = df_ml.sort_values(['ano', 'trimestre']).reset_index(drop=True)

    # Entrenamiento hasta 2024
    train = df_ml[df_ml['ano'] <= 2024].copy()
    # Prueba de 2025 en adelante
    test = df_ml[df_ml['ano'] >= 2025].copy()

    X_train = train[variables_numericas + variables_categoricas]
    y_train = train['coste_en_euros']

    X_test = test[variables_numericas + variables_categoricas]
    y_test = test['coste_en_euros']

    print(f"Datos de Entrenamiento: {X_train.shape[0]} registros (<= 2024)")
    print(f"Datos de Evaluación: {X_test.shape[0]} registros (>= 2025)")

    # Preprocesamiento con ColumnTransformer
    preprocesamiento = ColumnTransformer(
        transformers=[
            ('numericas', StandardScaler(), variables_numericas),
            ('categoricas', OneHotEncoder(handle_unknown='ignore'), variables_categoricas)
        ]
    )

    # Definimos los modelos
    modelos = {
        'Regresión lineal': LinearRegression(),
        'Random Forest': RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=3, 
            random_state=42,
            n_jobs= 1 
        ),
        'XGBoost': XGBRegressor(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
            n_jobs= 1
        )
    }

    resultados = []
    modelos_entrenados = {}

    for nombre, algoritmo in modelos.items():
        print(f"Entrenando {nombre}...")
        
        # Pipeline completo
        pipeline = Pipeline(
            steps=[
                ('preprocesamiento', preprocesamiento),
                ('modelo', algoritmo)
            ]
        )

        # Entrenamiento y predicción
        pipeline.fit(X_train, y_train)
        predicciones = pipeline.predict(X_test)

        # Evaluación
        mae = mean_absolute_error(y_test, predicciones)
        rmse = np.sqrt(mean_squared_error(y_test, predicciones))
        r2 = r2_score(y_test, predicciones)

        resultados.append({
            'Modelo': nombre,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2
        })

        modelos_entrenados[nombre] = pipeline

    # Formateamos resultados
    df_resultados = pd.DataFrame(resultados).sort_values('MAE')
    
    print('\nResultados Comparativos de Modelos:')
    print(df_resultados.round(3).to_string(index=False))

    return df_resultados, modelos_entrenados, X_test, y_test

# Función para predecir el futuro del coste en euros de Q3 y Q4
def predecir_futuro_con_estacionalidad(df_integrado, modelo_entrenado):
    """
    Calcula la estacionalidad histórica (variación entre trimestres) 
    y la aplica al último dato real (Q2 2026) para proyectar 
    los costes laborales en Q3 y Q4 de 2026 usando el modelo predictivo.
    """
    print('\n--- Predicción con Estacionalidad Histórica Real (Q3 y Q4 2026) ---')

    df_ml = df_integrado.copy()
    lista_comunidades = df_ml['comunidades_autonomas'].unique()
    lista_sectores = df_ml['sectores_actividad'].unique()

    escenarios_futuros = []

    for sector in lista_sectores:
        for ccaa in lista_comunidades:
            historico_ccaa_sector = df_ml[
                (df_ml['comunidades_autonomas'] == ccaa) & 
                (df_ml['sectores_actividad'] == sector)
            ]

            # Usamos el dato base (Q2 2026) garantizado que existe
            ultimo_dato = historico_ccaa_sector[
                (historico_ccaa_sector['ano'] == 2026) &
                (historico_ccaa_sector['trimestre'] == 2)
            ]

            t_act_base = ultimo_dato['tasa_actividad'].values[0]
            t_emp_base = ultimo_dato['tasa_empleo'].values[0]
            t_par_base = ultimo_dato['tasa_paro'].values[0]

            # Medias de tasas para cada trimestre a lo largo de todos los años
            medias_trimestrales = historico_ccaa_sector.groupby('trimestre')[['tasa_actividad', 'tasa_empleo', 'tasa_paro']].mean()

            # Variación histórica exacta (Delta) de Q2 a Q3
            delta_q3_act = medias_trimestrales.loc[3, 'tasa_actividad'] - medias_trimestrales.loc[2, 'tasa_actividad']
            delta_q3_emp = medias_trimestrales.loc[3, 'tasa_empleo'] - medias_trimestrales.loc[2, 'tasa_empleo']
            delta_q3_par = medias_trimestrales.loc[3, 'tasa_paro'] - medias_trimestrales.loc[2, 'tasa_paro']

            # Variación histórica exacta (Delta) de Q3 a Q4
            delta_q4_act = medias_trimestrales.loc[4, 'tasa_actividad'] - medias_trimestrales.loc[3, 'tasa_actividad']
            delta_q4_emp = medias_trimestrales.loc[4, 'tasa_empleo'] - medias_trimestrales.loc[3, 'tasa_empleo']
            delta_q4_par = medias_trimestrales.loc[4, 'tasa_paro'] - medias_trimestrales.loc[3, 'tasa_paro']

            # Aplicamos estas variaciones reales a nuestros datos de 2026
            t_act_q3 = t_act_base + delta_q3_act
            t_emp_q3 = t_emp_base + delta_q3_emp
            t_par_q3 = t_par_base + delta_q3_par

            t_act_q4 = t_act_q3 + delta_q4_act
            t_emp_q4 = t_emp_q3 + delta_q4_emp
            t_par_q4 = t_par_q3 + delta_q4_par

            # Construimos los 2 trimestres futuros basados en probabilidades estadísticas
            df_temp = pd.DataFrame({
                'ano': [2026, 2026],
                'trimestre': [3, 4],
                'tasa_actividad': [t_act_q3, t_act_q4], 
                'tasa_empleo': [t_emp_q3, t_emp_q4], 
                'tasa_paro': [t_par_q3, t_par_q4],   
                'comunidades_autonomas': [ccaa, ccaa],
                'sectores_actividad': [sector, sector]
            })

            escenarios_futuros.append(df_temp)

    # Concatenamos y predecimos en bloque
    datos_futuros_global = pd.concat(escenarios_futuros, ignore_index=True)

    # El pipeline se encarga del escalado y codificación automáticamente
    predicciones = modelo_entrenado.predict(datos_futuros_global)
    datos_futuros_global['coste_estimado_euros'] = predicciones.round(2)

    columnas_reporte = [
        'sectores_actividad', 'comunidades_autonomas', 'trimestre', 
        'tasa_actividad', 'tasa_empleo', 'tasa_paro', 'coste_estimado_euros'
    ]

    print("\n✅ Predicciones calculadas con éxito. Muestra de los resultados:")
    print(datos_futuros_global[columnas_reporte].head(15).to_string(index=False))

    return datos_futuros_global