from src.data_loader import cargar_datos
from src.eda import resumen_inicial
from src.cleaning import limpiar_costes, limpiar_epa
from src.visualizations import (
    graficar_coste_laboral_ccaa, 
    graficar_subvenciones_ccaa, 
    graficar_evolucion_coste_laboral, 
    graficar_evolucion_subvenciones,
    graficar_composicion_coste_laboral, 
    graficar_distribucion_porcentual, 
    graficar_composicion_otros_costes
)
from src.database import exportar_a_mysql


def main():

     # 1. Carga de los CSV
    print("\nCargando datasets del INE...\n")

    df_coste, df_epa = cargar_datos()

    print("Datos cargados correctamente.")

    # 2. EDA inicial sobre los datos originales
    resumen_inicial(df_coste, df_epa)

    # 3. Limpieza de costes
    (df_coste_limpio, df_total_nacional_coste) = limpiar_costes(df_coste)

    # 4. Limpieza de EPA y agrupación autonómica
    (df_epa_limpio, df_total_nacional_epa, df_epa_ccaa) = limpiar_epa(df_epa)

    # 5. Mostramos las dimensiones de los resultados
    print('\n--- Resultados de la limpieza ---')

    print('Costes:', df_coste_limpio.shape)
    print('Total nacional de costes:', df_total_nacional_coste.shape)

    print('EPA provincial:', df_epa_limpio.shape)
    print('Total nacional de EPA:', df_total_nacional_epa.shape)
    print('EPA agregada por CCAA:', df_epa_ccaa.shape)

    # 6. Análisis Exploratorio (EDA) - Gráficos
    print('\nGenerando gráficos EDA...')
    orden_ccaa_coste = graficar_coste_laboral_ccaa(df_coste_limpio)

    # Le pasamos el mismo orden para que visualmente las CCAA estén alineadas 
    # en ambos gráficos y sea más fácil comparar
    graficar_subvenciones_ccaa(df_coste_limpio, orden_ccaa=orden_ccaa_coste)

    # Llamada a los gráficos de evolución
    graficar_evolucion_coste_laboral(df_coste_limpio)
    graficar_evolucion_subvenciones(df_coste_limpio)

    # Gráficos de composición
    df_pivote = graficar_composicion_coste_laboral(df_coste_limpio)
    graficar_distribucion_porcentual(df_pivote)
    graficar_composicion_otros_costes(df_coste_limpio)

    # 7. Carga de datos en SQL
    print('\nIniciando exportación a base de datos...')
    exportar_a_mysql(df_coste_limpio, df_epa_ccaa)

if __name__ == "__main__":
    main()