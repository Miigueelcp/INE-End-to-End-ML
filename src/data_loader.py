from pathlib import Path
import os
import pandas as pd


# Directorio raíz del proyecto
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Directorio de los CSV originales
DATA_DIR = PROJECT_ROOT / "data" 

# Rutas de los archivos
DATA_COSTE = (
    DATA_DIR
    / "Coste laboral por trabajador, comunidad autónoma, sectores de actividad.csv"
)

DATA_EPA = (
    DATA_DIR
    / "Tasas de actividad, paro y empleo por provincia y sexo.csv"
)


def cargar_datos():
    """
    Carga los CSV originales del INE y devuelve ambos DataFrames.
    """

    try:
        df_coste = pd.read_csv(
            DATA_COSTE,
            sep=";",
            encoding="utf-8-sig"
        )

        df_epa = pd.read_csv(
            DATA_EPA,
            sep=";",
            encoding="utf-8-sig"
        )

        return df_coste, df_epa
    
    except FileNotFoundError as e:
        print(f"\n❌ ERROR de ruta. \nEl script está buscando en: {DATA_DIR}")
        print("Asegúrate de que la carpeta 'data' está en la raíz del proyecto y contiene los archivos.")
        raise e