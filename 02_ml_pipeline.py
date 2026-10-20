from src.database import extraer_vista_ml
from src.ml_preprocessing import preparar_dataset_ml
from src.visualizations import (
    graficar_matriz_correlacion,
    graficar_distribuciones_ml,
    graficar_boxplots_ml,
    graficar_explicabilidad_shap
)
from src.ml_models import entrenar_y_evaluar_modelos, predecir_futuro_con_estacionalidad
from src.ml_clustering import ejecutar_analisis_kmeans
from pathlib import Path

def main():

    # 1. Extraemos los datos integrados desde SQL
    df_ml_raw = extraer_vista_ml()
    
    # Comprobación 
    print("\nPrimeras filas del dataset para Machine Learning:")
    print(df_ml_raw.head(3))

    # 2. Preprocesamiento: Pivotamos y cruzamos (Feature Engineering)
    df_integrado = preparar_dataset_ml(df_ml_raw)

    # Verificamos visualmente
    print(df_integrado.head())

    # 3. Análisis Exploratorio Pre-ML (EDA)
    print("\nGenerando gráficos de EDA para Machine Learning...")
    graficar_matriz_correlacion(df_integrado, guardar_como="matriz_correlacion.png")
    graficar_distribuciones_ml(df_integrado)
    graficar_boxplots_ml(df_integrado)
    
    print("\nAnálisis visual completado.")
    
    # 4. Machine Learning (Entrenamiento y Evaluación)
    df_resultados, modelos_entrenados, X_test, y_test = entrenar_y_evaluar_modelos(df_integrado)

    # 5. Predicción a futuro (Q3 y Q4 2026) usando el mejor modelo
    mejor_modelo = modelos_entrenados['XGBoost']
    df_proyecciones = predecir_futuro_con_estacionalidad(df_integrado, mejor_modelo)

    # Definimos la ruta para guardar las predicciones a futuro
    PROJECT_ROOT = Path(__file__).resolve().parent
    OUTPUTS_DIR = PROJECT_ROOT / "outputs"

    # Creamos la carpeta si no existe
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # Exportamos las predicciones a un CSV
    ruta_salida = OUTPUTS_DIR / "predicciones_q3_q4_2026.csv"
    df_proyecciones.to_csv(ruta_salida, index=False)

    # 6. Explicabilidad del modelo (SHAP)
    modelo_xgb_puro = modelos_entrenados['XGBoost'].named_steps['modelo']
    preprocesador = modelos_entrenados['XGBoost'].named_steps['preprocesamiento']
    graficar_explicabilidad_shap(modelo_xgb_puro, preprocesador, X_test, 
                                 guardar_como_beeswarm="shap_beeswarm.png", 
                                 guardar_como_bar="shap_importancia_bar.png")

    # 7. Machine Learning No Supervisado (Clustering con K-Means)
    # Analizamos la relación histórica entre el paro y el coste laboral por CCAA
    ejecutar_analisis_kmeans(df_integrado)

if __name__ == "__main__":
    main()