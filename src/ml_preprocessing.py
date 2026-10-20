import pandas as pd

def preparar_dataset_ml(df_tabla):
    """
    Toma el dataset crudo extraído de SQL, valida su integridad, 
    pivota las tasas de la EPA a formato ancho (features) y lo integra 
    con los costes para dejar un DataFrame listo para Machine Learning.
    """
    print("\n--- 1. Validación inicial de los datos extraídos ---")
    print(f"Filas iniciales: {len(df_tabla)} | Columnas: {df_tabla.shape[1]}")
    print(f"Duplicados exactos: {df_tabla.duplicated().sum()}")
    
    # 1. Recuperamos la tabla de costes (evitando duplicidades)
    claves_coste = [
        'comunidades_autonomas', 'ano', 'trimestre', 
        'sectores_actividad', 'tipo_de_coste'
    ]
    
    df_costes = (
        df_tabla[claves_coste + ['coste_en_euros']]
        .drop_duplicates()
        .copy()
    )
    print(f"Filas de costes únicas extraídas: {len(df_costes)}")

    # 2. Seleccionamos y pivotamos las tasas de la EPA (ambos sexos)
    df_epa_ambos = df_tabla.loc[
        df_tabla['sexo'] == 'ambos_sexos',
        ['comunidades_autonomas', 'ano', 'trimestre', 'tipo_de_tasa', 'valor_tasa']
    ].drop_duplicates()

    # Validamos consistencia en EPA antes del pivot
    clave_epa = ['comunidades_autonomas', 'ano', 'trimestre', 'tipo_de_tasa']
    inconsistencias = df_epa_ambos.groupby(clave_epa)['valor_tasa'].nunique().gt(1).sum()
    
    if inconsistencias > 0:
        print(f"⚠️ Advertencia: Encontradas {inconsistencias} combinaciones con valores inconsistentes en EPA.")

    # Convertimos las tasas en columnas (Features)
    df_epa_ancha = (
        df_epa_ambos
        .pivot(
            index=['comunidades_autonomas', 'ano', 'trimestre'],
            columns='tipo_de_tasa',
            values='valor_tasa'
        )
        .reset_index()
    )
    df_epa_ancha.columns.name = None  # Limpiamos el nombre del índice de columnas

    # 3. Volvemos a unir las tablas (Join final)
    df_integrado = pd.merge(
        df_costes,
        df_epa_ancha,
        on=['comunidades_autonomas', 'ano', 'trimestre'],
        how='inner',
        validate='many_to_one' 
    )

    print("\n--- 2. Dataset Integrado listo para ML ---")
    print(f"Dimensiones finales: {df_integrado.shape[0]} filas y {df_integrado.shape[1]} columnas.")
    
    duplicados_finales = df_integrado.duplicated(subset=claves_coste).sum()
    print(f"Duplicados en la clave principal: {duplicados_finales}")
    
    nulos_finales = df_integrado.isnull().sum().sum()
    print(f"Valores nulos totales en el dataset: {nulos_finales}\n")

    return df_integrado