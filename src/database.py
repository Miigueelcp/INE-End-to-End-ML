import os
from sqlalchemy import create_engine, text
import pandas as pd

def exportar_a_mysql(df_coste_limpio, df_epa_ccaa, usuario='root', puerto='3306', nombre_bd='portfolio_ine'):
    """
    Crea la base de datos local si no existe y exporta los DataFrames limpios a MySQL.
    La contraseña se lee de la variable de entorno 'MYSQL_PASSWORD' por seguridad.
    """
    
    contrasena = os.getenv('MYSQL_PASSWORD','') 
    
    if not contrasena:
        raise ValueError("❌ ERROR: Contraseña no encontrada. Configura la variable de entorno MYSQL_PASSWORD.")

    # 2. Configuración del servidor
    cadena_servidor = f'mysql+pymysql://{usuario}:{contrasena}@localhost:{puerto}'
    engine_servidor = create_engine(cadena_servidor)

    print('\nConectando a MySQL y preparando entorno...')

    # 3. Creamos la base de datos de forma segura
    try:
        with engine_servidor.connect() as conexion:
            conexion.execute(text(f"CREATE DATABASE IF NOT EXISTS {nombre_bd};"))
        print(f"✅ Base de datos '{nombre_bd}' lista y operativa.")
    except Exception as e:
        print("❌ Error al conectar o crear la base de datos. ¿Está MySQL encendido?")
        raise e

    # 4. Conectamos directamente a la nueva base de datos
    cadena_bd = f'{cadena_servidor}/{nombre_bd}'
    engine_bd = create_engine(cadena_bd)

    # 5. Exportamos las tablas
    print('Exportando la tabla de Costes...')
    df_coste_limpio.to_sql(
        name='costes_laborales',
        con=engine_bd,
        if_exists='replace',
        index=False
    )

    print('Exportando la tabla de la EPA...')
    df_epa_ccaa.to_sql(
        name='epa_tasas', 
        con=engine_bd, 
        if_exists='replace', 
        index=False
    )

    print('✅ Datos exportados a MySQL con éxito.\n')


def extraer_vista_ml(usuario='root', puerto='3306', nombre_bd='portfolio_ine'):
    """
    Se conecta a la base de datos MySQL y extrae la vista o tabla 'costes_epa_integrado' 
    para iniciar la fase de Machine Learning.
    """
    # 1. Recuperamos la contraseña 
    contrasena = os.getenv('MYSQL_PASSWORD','') 
    
    if not contrasena:
        raise ValueError("❌ ERROR: Contraseña no encontrada.")

    # 2. Creamos el motor conectado específicamente a tu base de datos
    cadena_bd = f'mysql+pymysql://{usuario}:{contrasena}@localhost:{puerto}/{nombre_bd}'
    engine_bd = create_engine(cadena_bd)

    print('\nExtrayendo la vista desde MySQL para Machine Learning...')
    query_ml = "SELECT * FROM costes_epa_integrado;"
    
    # 3. Extraemos los datos
    try:
        df_tabla = pd.read_sql(query_ml, con=engine_bd)
        print("✅ Datos extraídos con éxito.")
        print(f"Dimensión del dataset: {df_tabla.shape[0]} filas y {df_tabla.shape[1]} columnas.")
        return df_tabla
    except Exception as e:
        print("\n❌ Error al extraer los datos.")
        print("Verifica que la vista o tabla 'costes_epa_integrado' existe en tu base de datos.")
        raise e