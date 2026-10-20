import pandas as pd


# ============================================================
# 1. MAPAS DE TRANSFORMACIÓN
# ============================================================

MAPA_SECTORES = {
    (
        'B_S Industria, construcción y servicios '
        '(excepto actividades de los hogares como empleadores '
        'y de organizaciones y organismos extraterritoriales)'
    ): 'total_sectores',
    'Industria': 'industria',
    'Construcción': 'construccion',
    'Servicios': 'servicios'
}


MAPA_TASAS = {
    'Tasa de actividad': 'tasa_actividad',
    'Tasa de paro de la población': 'tasa_paro',
    'Tasa de empleo de la población': 'tasa_empleo'
}


MAPA_COMUNIDADES_EPA = {
    # Andalucía
    'almeria': 'andalucia',
    'cadiz': 'andalucia',
    'cordoba': 'andalucia',
    'granada': 'andalucia',
    'huelva': 'andalucia',
    'jaen': 'andalucia',
    'malaga': 'andalucia',
    'sevilla': 'andalucia',

    # Aragón
    'huesca': 'aragon',
    'teruel': 'aragon',
    'zaragoza': 'aragon',

    # Principado de Asturias
    'asturias': 'asturias_principado_de',

    # Illes Balears
    'balears_illes': 'balears_illes',

    # Canarias
    'palmas_las': 'canarias',
    'santa_cruz_de_tenerife': 'canarias',

    # Cantabria
    'cantabria': 'cantabria',

    # Castilla-La Mancha
    'albacete': 'castilla_la_mancha',
    'ciudad_real': 'castilla_la_mancha',
    'cuenca': 'castilla_la_mancha',
    'guadalajara': 'castilla_la_mancha',
    'toledo': 'castilla_la_mancha',

    # Castilla y León
    'avila': 'castilla_y_leon',
    'burgos': 'castilla_y_leon',
    'leon': 'castilla_y_leon',
    'palencia': 'castilla_y_leon',
    'salamanca': 'castilla_y_leon',
    'segovia': 'castilla_y_leon',
    'soria': 'castilla_y_leon',
    'valladolid': 'castilla_y_leon',
    'zamora': 'castilla_y_leon',

    # Cataluña
    'barcelona': 'cataluna',
    'girona': 'cataluna',
    'lleida': 'cataluna',
    'tarragona': 'cataluna',

    # Comunidad Valenciana
    'alicante/alacant': 'comunitat_valenciana',
    'castellon/castello': 'comunitat_valenciana',
    'valencia/valencia': 'comunitat_valenciana',

    # Extremadura
    'badajoz': 'extremadura',
    'caceres': 'extremadura',

    # Galicia
    'coruna_a': 'galicia',
    'lugo': 'galicia',
    'ourense': 'galicia',
    'pontevedra': 'galicia',

    # Comunidad de Madrid
    'madrid': 'madrid_comunidad_de',

    # Región de Murcia
    'murcia': 'murcia_region_de',

    # Comunidad Foral de Navarra
    'navarra': 'navarra_comunidad_foral_de',

    # País Vasco
    'araba/alava': 'pais_vasco',
    'bizkaia': 'pais_vasco',
    'gipuzkoa': 'pais_vasco',

    # La Rioja
    'rioja_la': 'rioja_la'
}


# ============================================================
# 2. FUNCIONES AUXILIARES
# ============================================================

def normalizar_texto(serie):
    """
    Normaliza texto: minúsculas, sin tildes,
    sin comas ni guiones y espacios sustituidos por '_'.
    """

    return (
        serie
        .str.lower()
        .str.replace(',', '', regex=False)
        .str.replace('-', '', regex=False)
        .str.normalize('NFKD')
        .str.encode('ascii', errors='ignore')
        .str.decode('utf-8')
        .str.strip()
        .str.replace(r'\s+', ' ', regex=True)
        .str.replace(' ', '_', regex=False)
    )


def convertir_decimal(serie):
    """
    Convierte números con formato español a valores numéricos.

    Ejemplo: '2.345,67' -> 2345.67
    """

    valores = (serie.astype('string')
        .str.replace('.', '', regex=False)
        .str.replace(',', '.', regex=False)
    )

    return pd.to_numeric(valores, errors='coerce')


# ============================================================
# 3. LIMPIEZA DE COSTES LABORALES
# ============================================================

def limpiar_costes(df_coste):
    """
    Limpia el dataset de costes laborales.

    Devuelve:
        df_coste_limpio:
            Datos de las comunidades autónomas.
        df_total_nacional_coste:
            Observaciones del total nacional.
    """

    df = df_coste.copy()

    # Normalizamos los nombres de las columnas
    df.columns = (
        df.columns
        .str.lower()
        .str.replace(' ', '_', regex=False)
    )

    # Simplificamos las columnas
    mapa_costes = {
        'comunidades_y_ciudades_autónomas':
            'comunidades_autonomas',
        'sectores_de_actividad_cnae_2009':
            'sectores_actividad',
        'componentes_del_coste': 'coste'
    }

    df = df.rename(columns=mapa_costes)

    # Eliminamos el código inicial de la comunidad autónoma
    df['comunidades_autonomas'] = (
        df['comunidades_autonomas']
        .str.replace(r'^\d+\s+', '', regex=True)
    )

    # Simplificamos los sectores
    df['sectores_actividad'] = (
        df['sectores_actividad']
        .map(MAPA_SECTORES)
    )

    # Normalizamos comunidad autónoma y componente del coste
    for columna in ['comunidades_autonomas', 'coste']:
        df[columna] = normalizar_texto(df[columna])

    # Normalizamos el nombre del componente de Seguridad Social
    df['coste'] = (
        df['coste']
        .str.replace(
            's.social',
            'seguridad_social',
            regex=False
        )
    )

    # Convertimos los importes a números decimales
    df['total'] = convertir_decimal(df['total'])

    # Separamos año y trimestre
    df['ano'] = df['periodo'].str[:4].astype(int)
    df['trimestre'] = df['periodo'].str[-1].astype(int)

    # Eliminamos la columna original del periodo
    df = df.drop(columns=['periodo'])

    # Separamos el total nacional
    df_total_nacional_coste = df[
        df['comunidades_autonomas'] == 'total_nacional'
    ].copy()

    # Conservamos las comunidades autónomas
    df_coste_limpio = df[
        df['comunidades_autonomas'] != 'total_nacional'
    ].copy()

    return df_coste_limpio, df_total_nacional_coste


# ============================================================
# 4. LIMPIEZA DE LA EPA
# ============================================================

def limpiar_epa(df_epa):
    """
    Limpia el dataset provincial de la EPA y genera
    una tabla agregada por comunidad autónoma.

    Devuelve:
        df_epa_limpio:
            Datos provinciales con la CCAA asignada.
        df_total_nacional_epa:
            Observaciones del total nacional.
        df_epa_ccaa:
            Media simple de las tasas provinciales por CCAA.
    """

    df = df_epa.copy()

    # Normalizamos los nombres de las columnas
    df.columns = df.columns.str.lower()

    # Eliminamos el código inicial de las provincias
    df['provincias'] = (
        df['provincias']
        .str.replace(r'^\d+\s+', '', regex=True)
    )

    # Simplificamos los nombres de las tasas
    df['tasas'] = df['tasas'].replace(MAPA_TASAS)

    # Normalizamos las categorías de texto
    for columna in ['provincias', 'tasas', 'sexo']:
        df[columna] = normalizar_texto(df[columna])

    # Separamos año y trimestre
    df['ano'] = df['periodo'].str[:4].astype(int)
    df['trimestre'] = df['periodo'].str[-1].astype(int)

    # Eliminamos el periodo original
    df = df.drop(columns=['periodo'])

    # Convertimos la tasa a número decimal
    df['total'] = convertir_decimal(df['total'])

    # Separamos el total nacional
    df_total_nacional_epa = df[
        df['provincias'] == 'total_nacional'
    ].copy()

    # Eliminamos el total nacional y las ciudades autónomas
    df = df[
        ~df['provincias'].isin([
            'total_nacional',
            'ceuta',
            'melilla'
        ])
    ].copy()

    # Asignamos cada provincia a su comunidad autónoma
    df['comunidades_autonomas'] = (
        df['provincias']
        .map(MAPA_COMUNIDADES_EPA)
    )

    # Evitamos que un error en el mapeo pase inadvertido
    provincias_sin_ccaa = df.loc[
        df['comunidades_autonomas'].isna(),
        'provincias'
    ].unique()

    if len(provincias_sin_ccaa) > 0:
        raise ValueError(
            'Existen provincias sin comunidad autónoma asignada: '
            f'{provincias_sin_ccaa}'
        )

    # Dataset provincial limpio
    df_epa_limpio = df.copy()

    # Agrupamos las tasas provinciales a nivel autonómico
    columnas_agrupar = [
        'comunidades_autonomas',
        'ano',
        'trimestre',
        'sexo',
        'tasas'
    ]

    df_epa_ccaa = (
        df_epa_limpio
        .groupby(columnas_agrupar, as_index=False)
        .agg(
            total=('total', 'mean'),
            n_provincias=('provincias', 'nunique'),
            n_valores_validos=('total', 'count')
        )
    )

    return (df_epa_limpio, df_total_nacional_epa, df_epa_ccaa)