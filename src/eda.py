def resumen_inicial(df_coste, df_epa):
    """
    Muestra un resumen exploratorio de los CSV originales.
    """

    datasets = {
        "Costes laborales": df_coste,
        "EPA": df_epa
    }

    for nombre, df in datasets.items():

        print(f"\n--- Análisis inicial: {nombre} ---\n")

        print("Primeras filas:")
        print(df.head(3))

        print("\nInformación del DataFrame:")
        df.info()

        print("\nEstadísticas descriptivas:")
        print(df.describe(include="all").T)

        print("\nValores nulos:")
        print(df.isna().sum())

        print("\nFilas duplicadas exactas:")
        print(df.duplicated().sum())