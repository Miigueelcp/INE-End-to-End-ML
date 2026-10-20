from src.visualizations import graficar_predicciones_q3_q4
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Rutas del proyecto
PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

RUTA_PREDICCIONES = OUTPUTS_DIR / "predicciones_q3_q4_2026.csv"


# Nombres legibles para mostrar los resultados
NOMBRES_CCAA = {
    "andalucia": "Andalucía",
    "aragon": "Aragón",
    "asturias_principado_de": "Asturias",
    "balears_illes": "Illes Balears",
    "canarias": "Canarias",
    "cantabria": "Cantabria",
    "castilla_la_mancha": "Castilla-La Mancha",
    "castilla_y_leon": "Castilla y León",
    "cataluna": "Cataluña",
    "comunitat_valenciana": "C. Valenciana",
    "extremadura": "Extremadura",
    "galicia": "Galicia",
    "madrid_comunidad_de": "Madrid",
    "murcia_region_de": "Murcia",
    "navarra_comunidad_foral_de": "Navarra",
    "pais_vasco": "País Vasco",
    "rioja_la": "La Rioja"
}

NOMBRES_SECTOR = {
    "industria": "Industria",
    "construccion": "Construcción",
    "servicios": "Servicios"
}


def euros(valor):
    """Formatea una cantidad en euros con formato español."""
    return f"{valor:,.2f} €".replace(",", "X").replace(".", ",").replace("X", ".")


def porcentaje(valor):
    """Formatea un porcentaje con dos decimales."""
    return f"{valor:.2f} %".replace(".", ",")


def main():

    # 1. Cargamos las predicciones previamente generadas
    if not RUTA_PREDICCIONES.exists():
        raise FileNotFoundError(
            f"No se encuentra el archivo de predicciones: {RUTA_PREDICCIONES}\n"
            "Ejecuta primero 02_ml_pipeline.py."
        )

    df = pd.read_csv(RUTA_PREDICCIONES)

    print("\n" + "=" * 60)
    print("RESUMEN DE PREDICCIONES DEL COSTE LABORAL")
    print("Tercer y cuarto trimestre de 2026")
    print("=" * 60)

    print(f"\nNúmero de predicciones: {len(df)}")

    # 2. Evolución general del coste laboral
    medias_trimestre = (
        df.groupby("trimestre")["coste_estimado_euros"].mean()
    )

    q3 = medias_trimestre.loc[3]
    q4 = medias_trimestre.loc[4]

    diferencia = q4 - q3
    variacion = (diferencia / q3) * 100

    print("\n1. EVOLUCIÓN GENERAL")
    print(f"Coste medio estimado en Q3: {euros(q3)}")
    print(f"Coste medio estimado en Q4: {euros(q4)}")
    print(f"Diferencia estimada: {euros(diferencia)}")
    print(f"Variación estimada: {porcentaje(variacion)}")

    # 3. Comparación por sector
    medias_sector = (
        df.groupby(["sectores_actividad", "trimestre"])
        ["coste_estimado_euros"]
        .mean()
        .unstack()
    )

    medias_sector["diferencia_euros"] = (
        medias_sector[4] - medias_sector[3]
    )
    medias_sector["variacion_pct"] = (
        medias_sector["diferencia_euros"] / medias_sector[3] * 100
    )

    print("\n2. RESULTADOS POR SECTOR")

    for sector, fila in medias_sector.iterrows():
        nombre = NOMBRES_SECTOR.get(sector, sector)

        print(f"\n{nombre}")
        print(f"  Q3: {euros(fila[3])}")
        print(f"  Q4: {euros(fila[4])}")
        print(f"  Variación: {porcentaje(fila['variacion_pct'])}")

    # 4. Comparación territorial
    medias_ccaa = (
        df.groupby(["comunidades_autonomas", "trimestre"])
        ["coste_estimado_euros"]
        .mean()
        .unstack()
    )

    medias_ccaa["variacion_pct"] = (
        (medias_ccaa[4] - medias_ccaa[3])
        / medias_ccaa[3] * 100
    )

    print("\n3. COMUNIDADES CON MAYOR COSTE ESTIMADO EN Q4")

    for comunidad, fila in medias_ccaa.nlargest(3, 4).iterrows():
        nombre = NOMBRES_CCAA.get(comunidad, comunidad)
        print(f"{nombre}: {euros(fila[4])}")

    comunidad_menor = medias_ccaa[4].idxmin()
    nombre_menor = NOMBRES_CCAA.get(
        comunidad_menor, comunidad_menor
    )

    print(f"\nMenor coste medio estimado en Q4: {nombre_menor}")
    print(f"Importe: {euros(medias_ccaa.loc[comunidad_menor, 4])}")

    # Comunidades con mayor y menor variación porcentual
    comunidad_mayor_variacion = medias_ccaa["variacion_pct"].idxmax()
    comunidad_menor_variacion = medias_ccaa["variacion_pct"].idxmin()

    print("\n4. VARIACIÓN TERRITORIAL ENTRE Q3 Y Q4")

    nombre = NOMBRES_CCAA.get(
        comunidad_mayor_variacion, comunidad_mayor_variacion
    )
    print(
        f"Mayor incremento: {nombre} "
        f"({porcentaje(medias_ccaa.loc[comunidad_mayor_variacion, 'variacion_pct'])})"
    )

    nombre = NOMBRES_CCAA.get(
        comunidad_menor_variacion, comunidad_menor_variacion
    )
    print(
        f"Menor incremento: {nombre} "
        f"({porcentaje(medias_ccaa.loc[comunidad_menor_variacion, 'variacion_pct'])})"
    )

    # 5. Tasas de actividad, empleo y paro incluidas en el CSV
    # Cada comunidad aparece repetida por sector; la contamos una sola vez.
    columnas_tasas = [
        "tasa_actividad",
        "tasa_empleo",
        "tasa_paro"
    ]

    df_tasas = df.drop_duplicates(
        subset=["comunidades_autonomas", "trimestre"]
    )

    medias_tasas = df_tasas.groupby("trimestre")[columnas_tasas].mean()

    print("\n5. INDICADORES DEL MERCADO LABORAL")

    nombres_tasas = {
        "tasa_actividad": "Tasa de actividad",
        "tasa_empleo": "Tasa de empleo",
        "tasa_paro": "Tasa de paro"
    }

    for columna, nombre in nombres_tasas.items():
        valor_q3 = medias_tasas.loc[3, columna]
        valor_q4 = medias_tasas.loc[4, columna]
        diferencia_pp = valor_q4 - valor_q3

        print(
            f"{nombre}: {valor_q3:.2f} % -> {valor_q4:.2f} % "
            f"({diferencia_pp:+.2f} puntos porcentuales)"
        )

    print("\n" + "=" * 60)
    print("Resumen finalizado.")
    print("=" * 60)

    df = pd.read_csv(RUTA_PREDICCIONES)

    print("Generando gráfico de predicciones...")
    graficar_predicciones_q3_q4(df)

if __name__ == "__main__":
    main()
