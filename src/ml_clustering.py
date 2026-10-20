import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from adjustText import adjust_text
from pathlib import Path
import os

# Diccionarios globales para el formateo de las visualizaciones
NOMBRES_CCAA_KMEANS = {
    'andalucia': 'AND', 'aragon': 'ARA', 'asturias_principado_de': 'AST',
    'balears_illes': 'BAL', 'canarias': 'CAN', 'cantabria': 'CTB',
    'castilla_la_mancha': 'CLM', 'castilla_y_leon': 'CYL',
    'cataluna': 'CAT', 'comunitat_valenciana': 'VAL', 'extremadura': 'EXT',
    'galicia': 'GAL', 'madrid_comunidad_de': 'MAD', 'murcia_region_de': 'MUR',
    'navarra_comunidad_foral_de': 'NAV', 'pais_vasco': 'PV', 'rioja_la': 'RIO'
}

ABREVIATURAS_SECTOR = {
    'industria': 'IND', 'construccion': 'CON', 'servicios': 'SER'
}

NOMBRES_SECTOR = {
    'industria': 'Industria', 'construccion': 'Construcción', 'servicios': 'Servicios'
}

def _agrupar_periodo(df, inicio, fin):
    """Filtra el DataFrame por un rango de años y calcula medias de paro y coste."""

    datos = df[df['ano'].between(inicio, fin)].copy()

    return (
        datos.groupby(
            ['comunidades_autonomas', 'sectores_actividad'],
            as_index=False
        )
        .agg(
            tasa_paro=('tasa_paro', 'mean'),
            coste_en_euros=('coste_en_euros', 'mean'),
            n_periodos=('ano', 'size')
        )
    )

def evaluar_silhouette_kmeans(X_escalado):
    """Calcula y grafica el Silhouette Score para determinar el K óptimo."""

    resultados_silhouette = {}

    for k in range(2, 7):

        modelo = KMeans(n_clusters=k, random_state=42, n_init=10)
        etiquetas = modelo.fit_predict(X_escalado)

        resultados_silhouette[k] = silhouette_score(X_escalado, etiquetas)

    plt.figure(figsize=(8, 5))

    plt.plot(list(resultados_silhouette.keys()), list(resultados_silhouette.values()), 
             marker='o', linewidth=2, color='#1e56a0')
    plt.title('Evaluación del número de clusters (Histórico 2008-2026)\n', fontsize=14, fontweight='bold')

    plt.xlabel('Número de clusters (K)')
    plt.ylabel('Silhouette Score')

    plt.xticks(list(resultados_silhouette.keys()))
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

def _contar_solapes(textos, fig):
    """Cuenta pares de etiquetas cuyas cajas se intersectan para ajustar visualmente."""

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    cajas = [t.get_window_extent(renderer) for t in textos]

    return sum(
        cajas[i].overlaps(cajas[j])
        for i in range(len(cajas))
        for j in range(i + 1, len(cajas))
    )

def _ajustar_etiquetas(etiquetas, ax, df, intento):
    """Llama a adjust_text con parámetros adaptativos según el intento."""

    flecha = dict(arrowstyle='-', color='gray', lw=0.5, alpha=0.6)
    x, y = df['tasa_paro'].values, df['coste_en_euros'].values

    adjust_text(
        etiquetas, ax=ax, x=x, y=y,
        expand=(1.4 + 0.2 * intento, 1.6 + 0.3 * intento),
        force_text=(0.8, 1.2), force_static=(0.6, 0.8),
        force_pull=(0.005, 0.005), ensure_inside_axes=True,
        max_move=(80, 80), iter_lim=5000, arrowprops=flecha
    )

def representar_clusters(df, inicio, fin, subtitulo, media_paro_historica, media_coste_historica, 
                         limites_x, limites_y, incluir_sector=True, guardar_como=None):
    """Genera el gráfico de dispersión de los clusters K-Means y ajusta las etiquetas."""

    fig, ax = plt.subplots(figsize=(20, 12))

    df = df.assign(
        Sector=df['sectores_actividad'].map(NOMBRES_SECTOR),
        Cluster=df['cluster']
    )

    sns.scatterplot(
        data=df, x='tasa_paro', y='coste_en_euros', hue='Cluster',
        palette='viridis', style='Sector', s=100, alpha=0.85, edgecolor='black', ax=ax
    )

    ax.axvline(media_paro_historica, color='red', linestyle='--', alpha=0.7, label='Media histórica del paro')
    ax.axhline(media_coste_historica, color='red', linestyle=':', alpha=0.7, label='Media histórica del coste')

    ax.set_title(
        f'Segmentación K-Means: paro y coste laboral por CCAA y sector\n'
        f'{subtitulo} ({inicio}-{fin})',
        fontsize=17, fontweight='bold', pad=18
    )
    ax.set_xlabel('Tasa de Paro Media (%)', fontsize=14, labelpad=10)
    ax.set_ylabel('Coste Laboral Medio por Trabajador (€)', fontsize=14, labelpad=10)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, pos: f"{int(v):,}".replace(',', '.')))
    
    ax.set_xlim(limites_x)
    ax.set_ylim(limites_y)
    ax.grid(True, linestyle='--', alpha=0.25)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
    plt.subplots_adjust(top=0.90, bottom=0.08, left=0.07, right=0.82)

    etiquetas = []

    for _, fila in df.iterrows():

        comunidad = NOMBRES_CCAA_KMEANS.get(fila['comunidades_autonomas'], fila['comunidades_autonomas'])
        sector = ABREVIATURAS_SECTOR.get(fila['sectores_actividad'], fila['sectores_actividad'])
        texto = f'{comunidad} ({sector})' if incluir_sector else comunidad

        etiquetas.append(ax.text(fila['tasa_paro'], fila['coste_en_euros'], texto, fontsize=7.5, fontweight='500'))

    # Ajuste adaptativo de etiquetas
    fig.canvas.draw()

    for intento in range(8):

        _ajustar_etiquetas(etiquetas, ax, df, intento)
        n = _contar_solapes(etiquetas, fig)
        print(f'[{inicio}-{fin}] Intento {intento + 1}: {n} solapes entre etiquetas')

        if n == 0:
            break

    # Guardamos las imágenes en la carpeta images/
    if guardar_como is not None:
    
        # Define la raiz del proyecyo y la carpeta de imágenes
        project_root = Path(__file__).resolve().parents[1] 
        images_dir = project_root / "images"
            
        # Se crea la carpeta si no existe
        images_dir.mkdir(parents=True, exist_ok=True)
            
        # Construye la ruta final de la imagen
        ruta_imagen = images_dir / guardar_como
            
        # Las guardamos
        fig.savefig(ruta_imagen, dpi=300, bbox_inches='tight')
        print(f"✅ Gráfico guardado en: {ruta_imagen}")
            
    plt.show()
    return fig

def ejecutar_analisis_kmeans(df_integrado):
    """
    Función principal que coordina la preparación de datos, evaluación, 
    entrenamiento del modelo K-Means y la visualización de los resultados.
    """

    print('\n--- Segmentación K-Means: Evolución Paro vs Coste Laboral ---')

    df_base = df_integrado.copy()

    df_historico = _agrupar_periodo(df_base, 2008, 2026)
    df_reciente = _agrupar_periodo(df_base, 2020, 2025)

    variables = ['tasa_paro', 'coste_en_euros']
    X_historico = df_historico[variables]

    scaler = StandardScaler()
    X_historico_escalado = scaler.fit_transform(X_historico)

    evaluar_silhouette_kmeans(X_historico_escalado)

    # Entrenamiento con 4 clusters (K óptimo)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df_historico['cluster'] = kmeans.fit_predict(X_historico_escalado)

    X_reciente_escalado = scaler.transform(df_reciente[variables])
    df_reciente['cluster'] = kmeans.predict(X_reciente_escalado)

    print('\n--- Perfil medio de los clusters históricos ---')
    resumen_hist = df_historico.groupby('cluster').agg(
        tasa_paro_media=('tasa_paro', 'mean'),
        coste_laboral_medio=('coste_en_euros', 'mean'),
        numero_observaciones=('comunidades_autonomas', 'size')
    ).round(2)

    print(resumen_hist)

    print('\n--- Perfil medio de los clusters recientes ---')
    resumen_rec = df_reciente.groupby('cluster').agg(
        tasa_paro_media=('tasa_paro', 'mean'),
        coste_laboral_medio=('coste_en_euros', 'mean'),
        numero_observaciones=('comunidades_autonomas', 'size')
    ).round(2)
    
    print(resumen_rec)

    # Cálculo de referencias visuales para alinear ambos gráficos
    media_paro_historica = df_historico['tasa_paro'].mean()
    media_coste_historica = df_historico['coste_en_euros'].mean()

    todos_datos = pd.concat([df_historico[variables], df_reciente[variables]], ignore_index=True)
    margen_paro = (todos_datos['tasa_paro'].max() - todos_datos['tasa_paro'].min()) * 0.12
    margen_coste = (todos_datos['coste_en_euros'].max() - todos_datos['coste_en_euros'].min()) * 0.12
    
    limites_x = (todos_datos['tasa_paro'].min() - margen_paro, todos_datos['tasa_paro'].max() + margen_paro)
    limites_y = (todos_datos['coste_en_euros'].min() - margen_coste, todos_datos['coste_en_euros'].max() + margen_coste)

    fig_hist = representar_clusters(
        df_historico, 2008, 2026, 'Análisis Histórico Global', 
        media_paro_historica, media_coste_historica, limites_x, limites_y,
        guardar_como='grafico_historico.png' 
    )

    fig_rec = representar_clusters(
        df_reciente, 2020, 2025, 'Radiografía del Periodo Reciente',
        media_paro_historica, media_coste_historica, limites_x, limites_y,
        guardar_como='grafico_reciente.png' 
    )