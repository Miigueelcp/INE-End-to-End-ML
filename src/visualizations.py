import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import shap
import scipy.sparse

# ============================================================
# 1. DICCIONARIOS DE FORMATO 
# ============================================================

NOMBRES_CCAA_LIMPIO = {
    'andalucia': 'Andalucía', 'aragon': 'Aragón', 'asturias_principado_de': 'Asturias',
    'balears_illes': 'Illes Balears', 'canarias': 'Canarias', 'cantabria': 'Cantabria',
    'castilla_la_mancha': 'Castilla-La Mancha', 'castilla_y_leon': 'Castilla y León',
    'cataluna': 'Cataluña', 'comunitat_valenciana': 'C. Valenciana', 'extremadura': 'Extremadura',
    'galicia': 'Galicia', 'madrid_comunidad_de': 'Madrid', 'murcia_region_de': 'Murcia',
    'navarra_comunidad_foral_de': 'Navarra', 'pais_vasco': 'País Vasco', 'rioja_la': 'La Rioja'
}

NOMBRES_SECTOR_LIMPIO = {
    'industria': 'Industria', 'construccion': 'Construcción', 'servicios': 'Servicios'
}

# ============================================================
# 2. FUNCIONES BASE
# ============================================================

def _crear_grafico_ccaa_sector(df, orden_ccaa, titulo, ylabel):
    """
    Función interna para generar la plantilla base del gráfico de barras.
    Evita repetir el código de diseño en cada gráfica.
    """

    fig, ax = plt.subplots(figsize=(16, 8))

    sns.barplot(
        data=df,
        x='CCAA_Limpia',
        y='total',
        hue='Sector_Limpio',
        palette='viridis',
        order=orden_ccaa,        
        errorbar=None,
        edgecolor='white',      
        linewidth=1,
        alpha=0.9,
        ax=ax
    )

    sns.despine()

    plt.title(titulo, fontsize=17, fontweight='bold', pad=20)
    plt.xlabel('', fontsize=12) 
    plt.ylabel(ylabel, fontsize=13, labelpad=10)

    plt.xticks(rotation=45, ha='right', fontsize=11, fontweight='500')

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: f"{int(y):,}".replace(',', '.')))
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    plt.legend(
        title='Sector de Actividad', 
        bbox_to_anchor=(1.02, 1), 
        loc='upper left',
        title_fontsize=12,
        fontsize=11,
        frameon=True
    )

    plt.tight_layout()
    plt.show()

# ============================================================
# 3. FUNCIONES DE GRÁFICOS ESPECÍFICOS
# ============================================================

def graficar_coste_laboral_ccaa(df_coste_limpio):
    """Genera el gráfico de coste laboral total por CCAA y sector."""
    
    df_coste_laboral = df_coste_limpio[
        (df_coste_limpio['coste'] == 'coste_laboral_total') &
        (df_coste_limpio['sectores_actividad'] != 'total_sectores')
    ].copy()

    df_coste_laboral['CCAA_Limpia'] = df_coste_laboral['comunidades_autonomas'].map(NOMBRES_CCAA_LIMPIO)
    df_coste_laboral['Sector_Limpio'] = df_coste_laboral['sectores_actividad'].map(NOMBRES_SECTOR_LIMPIO)

    orden_ccaa = (
        df_coste_laboral.groupby('CCAA_Limpia')['total']
        .mean()
        .sort_values(ascending=False)
        .index
    )

    _crear_grafico_ccaa_sector(
        df=df_coste_laboral,
        orden_ccaa=orden_ccaa,
        titulo='Coste Laboral Total Medio por Comunidad Autónoma y Sector',
        ylabel='Coste Laboral Medio (€)'
    )
    
    return orden_ccaa # Devolvemos el orden para usarlo en otros gráficos si hace falta


def graficar_subvenciones_ccaa(df_coste_limpio, orden_ccaa=None):
    """Genera el gráfico de subvenciones por CCAA y sector."""
    
    df_subvenciones = df_coste_limpio[
        (df_coste_limpio['coste'] == 'subvenciones_y_bonificaciones_de_la_seguridad_social') & 
        (df_coste_limpio['sectores_actividad'] != 'total_sectores')
    ].copy()

    df_subvenciones['CCAA_Limpia'] = df_subvenciones['comunidades_autonomas'].map(NOMBRES_CCAA_LIMPIO)
    df_subvenciones['Sector_Limpio'] = df_subvenciones['sectores_actividad'].map(NOMBRES_SECTOR_LIMPIO)

    _crear_grafico_ccaa_sector(
        df=df_subvenciones,
        orden_ccaa=orden_ccaa,
        titulo='Subvenciones y Bonificaciones Medias por Comunidad Autónoma y Sector',
        ylabel='Subvenciones (Media en €)'
    )


# ============================================================
# 4. FUNCIÓN BASE DE DISEÑO PARA LÍNEAS (EVOLUCIÓN TEMPORAL)
# ============================================================

def _crear_grafico_evolucion(df, titulo, ylabel):
    """
    Función interna para generar la plantilla base de los gráficos de líneas.
    """
    fig, ax = plt.subplots(figsize=(16, 8))

    sns.lineplot(
        data=df,
        x='ano',
        y='total',
        hue='Sector_Limpio',
        marker='o',
        markersize=8,       
        linewidth=2.5,      
        errorbar=None,
        palette='viridis',  
        ax=ax
    )

    sns.despine()

    plt.title(titulo, fontsize=17, fontweight='bold', pad=20)
    plt.xlabel('Año', fontsize=13, labelpad=10)
    plt.ylabel(ylabel, fontsize=13, labelpad=10)

    # Eje X con números enteros
    eje_x_años = np.sort(df['ano'].unique())
    plt.xticks(eje_x_años, rotation=45, fontsize=11, fontweight='500')

    # Formateo de miles en el eje Y
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: f"{int(y):,}".replace(',', '.')))

    ax.grid(True, linestyle='--', alpha=0.4)

    plt.legend(
        title='Sector de Actividad', 
        bbox_to_anchor=(1.02, 1), 
        loc='upper left',
        title_fontsize=12,
        fontsize=11,
        frameon=True
    )

    plt.subplots_adjust(top=0.90, bottom=0.12, left=0.06, right=0.85)
    plt.show()

# ============================================================
# 5. FUNCIONES DE GRÁFICOS DE EVOLUCIÓN
# ============================================================

def graficar_evolucion_coste_laboral(df_coste_limpio):
    """Genera el gráfico de evolución histórica del coste laboral total."""
    
    df_coste_laboral = df_coste_limpio[
        (df_coste_limpio['coste'] == 'coste_laboral_total') &
        (df_coste_limpio['sectores_actividad'] != 'total_sectores')
    ].copy()

    df_coste_laboral['Sector_Limpio'] = df_coste_laboral['sectores_actividad'].map(NOMBRES_SECTOR_LIMPIO)

    _crear_grafico_evolucion(
        df=df_coste_laboral,
        titulo='Evolución Histórica del Coste Laboral Medio por Sector (2008-2026)',
        ylabel='Coste Laboral Medio (€)'
    )


def graficar_evolucion_subvenciones(df_coste_limpio):
    """Genera el gráfico de evolución histórica de las subvenciones."""
    
    df_subvenciones = df_coste_limpio[
        (df_coste_limpio['coste'] == 'subvenciones_y_bonificaciones_de_la_seguridad_social') & 
        (df_coste_limpio['sectores_actividad'] != 'total_sectores')
    ].copy()

    df_subvenciones['Sector_Limpio'] = df_subvenciones['sectores_actividad'].map(NOMBRES_SECTOR_LIMPIO)

    _crear_grafico_evolucion(
        df=df_subvenciones,
        titulo='Evolución Histórica de Subvenciones y Bonificaciones por Sector (2008-2026)',
        ylabel='Volumen Medio de Subvenciones (€)'
    )

# ============================================================
# 6. FUNCIÓN BASE PARA BARRAS APILADAS (COMPOSICIÓN)
# ============================================================

def _crear_grafico_apilado(df_pivot, titulo, ylabel, traduccion_leyenda, es_porcentaje=False):
    """
    Función interna para generar la plantilla base de gráficos de barras apiladas.
    Permite formatear el eje Y como valor monetario o como porcentaje.
    """
    
    ax = df_pivot.plot(
        kind='bar',
        stacked=True,
        figsize=(12, 7),
        colormap='Set2',
        edgecolor='black',
        alpha=0.85
    )

    sns.despine()

    plt.title(titulo, fontsize=15, fontweight='bold', pad=15)
    plt.xlabel('Sector de Actividad', fontsize=12, labelpad=10)
    plt.ylabel(ylabel, fontsize=12, labelpad=10)

    # Formateo dinámico del eje X con el diccionario global
    nombres_eje_x = [
        NOMBRES_SECTOR_LIMPIO.get(nombre.get_text().lower(), nombre.get_text().capitalize()) 
        for nombre in ax.get_xticklabels()
    ]
    ax.set_xticklabels(nombres_eje_x, rotation=0, fontsize=11)

    # Formateo condicional del eje Y y etiquetas interiores
    if es_porcentaje:
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(100))
        ax.set_ylim(0, 100)
        # Imprimimos el % dentro de cada barra
        for c in ax.containers:
            etiquetas_barras = [f'{v.get_height():.1f}%' if v.get_height() > 0 else '' for v in c]
            ax.bar_label(c, labels=etiquetas_barras, label_type='center', fontsize=10, fontweight='600')
    else:
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: f"{int(y):,}".replace(',', '.')))

    # Limpieza y formateo de la leyenda
    handles, labels = ax.get_legend_handles_labels()
    etiquetas_limpias = [traduccion_leyenda.get(label, label) for label in labels]

    plt.legend(
        handles, 
        etiquetas_limpias, 
        title='Componente', 
        bbox_to_anchor=(1.02, 1), 
        loc='upper left',
        title_fontsize=12,
        fontsize=11,
        frameon=True
    )

    plt.tight_layout()
    plt.show()

# ============================================================
# 7. FUNCIONES DE COMPOSICIÓN Y PORCENTAJES
# ============================================================

def graficar_composicion_coste_laboral(df_coste_limpio):
    """Genera gráfico de barras apiladas del Coste Salarial vs Otros Costes."""
    
    componentes = ['coste_salarial_total', 'otros_costes']
    
    df_composicion = df_coste_limpio[
        (df_coste_limpio['coste'].isin(componentes)) & 
        (df_coste_limpio['sectores_actividad'] != 'total_sectores')
    ].copy()

    df_pivote = df_composicion.pivot_table(
        index='sectores_actividad', columns='coste', values='total', aggfunc='mean'
    )

    traduccion = {
        'coste_salarial_total': 'Coste Salarial Total',
        'otros_costes': 'Otros Costes (Cotizaciones, etc.)'
    }

    _crear_grafico_apilado(
        df_pivot=df_pivote,
        titulo='Composición del Coste Laboral por Sector',
        ylabel='Coste Medio por Trabajador (€)',
        traduccion_leyenda=traduccion,
        es_porcentaje=False
    )
    
    # Devolvemos el pivote por si se quiere usar para el gráfico de porcentajes
    return df_pivote


def graficar_composicion_otros_costes(df_coste_limpio):
    """Genera gráfico apilado separando percepciones, cotizaciones y subvenciones."""
    
    componentes = [
        'coste_por_percepciones_no_salariales',
        'coste_por_cotizaciones_obligatorias',
        'subvenciones_y_bonificaciones_de_la_seguridad_social'
    ]

    df_otros = df_coste_limpio[
        (df_coste_limpio['coste'].isin(componentes)) &
        (df_coste_limpio['sectores_actividad'] != 'total_sectores')
    ].copy()

    df_otros['total_ajustado'] = df_otros['total']
    
    # Asignamos signo negativo a las subvenciones
    es_subvencion = (df_otros['coste'] == 'subvenciones_y_bonificaciones_de_la_seguridad_social')
    df_otros.loc[es_subvencion, 'total_ajustado'] *= -1

    df_pivote = df_otros.pivot_table(
        index='sectores_actividad', columns='coste', values='total_ajustado', aggfunc='mean'
    )

    traduccion = {
        'coste_por_cotizaciones_obligatorias': 'Cotizaciones Obligatorias',
        'coste_por_percepciones_no_salariales': 'Percepciones No Salariales',
        'subvenciones_y_bonificaciones_de_la_seguridad_social': 'Subvenciones y Bonificaciones (-)'
    }

    # Como necesitamos añadir la línea horizontal del cero, llamamos a la función 
    # y luego aprovechamos que plt retiene el estado actual para añadirla antes del show final
    _crear_grafico_apilado(
        df_pivot=df_pivote,
        titulo='Composición de Otros Costes Laborales por Sector',
        ylabel='Aportación al coste medio (€)',
        traduccion_leyenda=traduccion,
        es_porcentaje=False
    )


def graficar_distribucion_porcentual(df_pivote_principal):
    """Transforma el pivote de costes absolutos en porcentajes y lo grafica."""
    
    df_pct = (
        df_pivote_principal.div(df_pivote_principal.sum(axis=1), axis=0) * 100
    )

    traduccion = {
        'coste_salarial_total': 'Coste Salarial Total',
        'otros_costes': 'Otros Costes (Cotizaciones, etc.)'
    }

    _crear_grafico_apilado(
        df_pivot=df_pct,
        titulo='Distribución Porcentual del Coste Laboral por Sector',
        ylabel='Proporción del Coste Total (%)',
        traduccion_leyenda=traduccion,
        es_porcentaje=True
    )


# ============================================================
# 8. EDA PARA MACHINE LEARNING (CORRELACIONES Y DISTRIBUCIONES)
# ============================================================

def graficar_matriz_correlacion(df):
    """Genera un mapa de calor con la correlación de Pearson."""

    variables_analisis = ['ano', 'trimestre', 'tasa_actividad', 'tasa_empleo', 'tasa_paro', 'coste_en_euros']

    nombres_limpios = {
        'ano': 'Año',
        'trimestre': 'Trimestre',
        'tasa_actividad': 'Tasa Actividad (%)',
        'tasa_empleo': 'Tasa Empleo (%)',
        'tasa_paro': 'Tasa Paro (%)',
        'coste_en_euros': 'Coste Laboral (€)'
    }

    df_corr = df[variables_analisis].copy()
    df_corr.rename(columns=nombres_limpios, inplace=True)
    matriz_corr = df_corr.corr()

    mask = np.triu(np.ones_like(matriz_corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(11, 8))

    sns.heatmap(
        matriz_corr, mask=mask, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1, 
        linewidths=1.5, linecolor='white', annot_kws={"size": 11, "fontweight": "500"},
        cbar_kws={'shrink': 0.8, 'label': 'Índice de Correlación de Pearson'}, ax=ax                     
    )

    fig.suptitle('Matriz de Correlaciones Lineales', fontsize=18, fontweight='bold', y=0.95)
    ax.set_title(
        'Relación entre variables temporales, laborales y coste en euros (Q1 2008 - Q2 2026)', 
        fontsize=13, color='#444444', pad=20
    )

    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(rotation=0, fontsize=11)
    plt.subplots_adjust(top=0.85, bottom=0.18, left=0.18, right=0.95)
    plt.show()


def graficar_distribuciones_ml(df):
    """Genera histogramas con curvas KDE para evaluar la normalidad."""

    variables_metricas = ['tasa_actividad', 'tasa_empleo', 'tasa_paro', 'coste_en_euros']

    nombres_limpios = {
        'tasa_actividad': 'Tasa de Actividad (%)',
        'tasa_empleo': 'Tasa de Empleo (%)',
        'tasa_paro': 'Tasa de Paro (%)',
        'coste_en_euros': 'Coste Laboral (€)'
    }

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 9))
    axes = axes.flatten()

    for i, col in enumerate(variables_metricas):
        sns.histplot(
            data=df, x=col, kde=True, stat='density',
            color='#1e56a0', edgecolor='white', alpha=0.7, ax=axes[i]
        )
        axes[i].set_title(nombres_limpios[col], fontsize=13, fontweight='bold', pad=12)
        axes[i].set_xlabel('')
        axes[i].set_ylabel('Densidad' if i % 2 == 0 else '')

        if col == 'coste_en_euros':
            axes[i].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{int(x):,}".replace(',', '.')))

    sns.despine(fig=fig)

    fig.suptitle('Análisis de Distribución y Normalidad (Histogramas y Curvas KDE)', fontsize=17, fontweight='bold')

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def graficar_boxplots_ml(df):
    """Genera diagramas de caja para la detección visual de valores atípicos (outliers)."""

    variables_metricas = ['tasa_actividad', 'tasa_empleo', 'tasa_paro', 'coste_en_euros']

    nombres_limpios = {
        'tasa_actividad': 'Tasa de Actividad (%)',
        'tasa_empleo': 'Tasa de Empleo (%)',
        'tasa_paro': 'Tasa de Paro (%)',
        'coste_en_euros': 'Coste Laboral (€)'
    }

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 9))
    axes = axes.flatten()

    for i, col in enumerate(variables_metricas):
        sns.boxplot(
            data=df, y=col, color='#4cb140', width=0.4, 
            linewidth=1.5, fliersize=6, ax=axes[i]
        )
        axes[i].set_title(nombres_limpios[col], fontsize=13, fontweight='bold', pad=12)
        axes[i].set_ylabel('')
        axes[i].grid(axis='y', linestyle='--', alpha=0.4)

        if col == 'coste_en_euros':
            axes[i].yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: f"{int(y):,}".replace(',', '.')))

    fig.suptitle('Detección de Valores Atípicos (Diagramas de Caja)', fontsize=17, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# ============================================================
# 9. EXPLICABILIDAD DE MACHINE LEARNING (SHAP)
# ============================================================

def _limpiar_nombres_shap(nombre):
    """Función auxiliar para limpiar los nombres de variables del transformador."""
    
    nombres_exactos = {
        'tasa_paro': 'Tasa de Paro (%)',
        'tasa_empleo': 'Tasa de Empleo (%)',
        'tasa_actividad': 'Tasa de Actividad (%)',
        'ano': 'Año',
        'trimestre': 'Trimestre'
    }

    if nombre in nombres_exactos:
        return nombres_exactos[nombre]
    
    # Limpieza de variables categóricas (Sectores)
    if nombre.startswith('sectores_actividad_'):
        sector = nombre.replace('sectores_actividad_', '').capitalize()
        return f'Sector: {sector}'
        
    # Limpieza de variables categóricas (CCAA)
    if nombre.startswith('comunidades_autonomas_'):
        ccaa = nombre.replace('comunidades_autonomas_', '').replace('_', ' ').title()
        ccaa = ccaa.replace(' De ', ' de ').replace(' La ', ' la ').replace(' Y ', ' y ')
        return f'CCAA: {ccaa}'
        
    return nombre

def graficar_explicabilidad_shap(modelo_xgb, preprocesador, X_test):
    """
    Calcula los valores SHAP del modelo XGBoost sobre una muestra de X_test 
    y genera los gráficos de Beeswarm (dirección) y Bar (importancia).
    """

    print('\n--- Explicabilidad del Modelo con SHAP ---')

    # 1. Preparación de los datos (Muestra de 1000 para que SHAP no tarde una eternidad)
    X_test_muestra = X_test.sample(n=min(1000, len(X_test)), random_state=42)
    X_test_transformado = preprocesador.transform(X_test_muestra)

    if scipy.sparse.issparse(X_test_transformado):
        X_test_transformado = X_test_transformado.toarray()

    nombres_variables = preprocesador.get_feature_names_out()
    df_X_test_shap = pd.DataFrame(X_test_transformado, columns=nombres_variables)

    # Limpieza de prefijos de Pipeline
    df_X_test_shap.columns = (
        df_X_test_shap.columns
        .str.replace('numericas__', '', regex=False)
        .str.replace('categoricas__', '', regex=False)
    )

    # Aplicamos la limpieza humana a todas las columnas
    df_X_test_shap.columns = [_limpiar_nombres_shap(col) for col in df_X_test_shap.columns]

    # 2. Cálculo de valores SHAP
    explainer = shap.TreeExplainer(modelo_xgb)
    shap_values = explainer.shap_values(df_X_test_shap)

    # ============================================================
    # Gráfico 1: SHAP Summary Plot (Beeswarm - Impacto y dirección)
    # ============================================================
    fig1 = plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, df_X_test_shap, max_display=12, show=False, plot_size=None)

    plt.gca().tick_params(axis='y', labelsize=10) 
    plt.gca().tick_params(axis='x', labelsize=11)
    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{int(x):,}".replace(',', '.')))

    plt.title('Impacto y Dirección de cada variable en el Coste Laboral\n(SHAP Beeswarm)\n', fontsize=16, fontweight='bold')
    plt.xlabel('Impacto SHAP (Efecto real en el coste laboral, €)', fontsize=12, labelpad=10)
    plt.subplots_adjust(left=0.28, right=0.92, top=0.88, bottom=0.12)
    plt.show()

    # ============================================================
    # Gráfico 2: SHAP Bar Plot (Importancia absoluta media)
    # ============================================================
    fig2 = plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, df_X_test_shap, plot_type="bar", max_display=12, color="#1e56a0", show=False, plot_size=None)

    plt.gca().tick_params(axis='y', labelsize=10) 
    plt.gca().tick_params(axis='x', labelsize=11)
    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{int(x):,}".replace(',', '.')))

    plt.title('Variables más Importantes para el Modelo\n(Media Absoluta SHAP)\n', fontsize=16, fontweight='bold')
    plt.xlabel('Impacto medio |SHAP| (cambio medio en el coste laboral, €)', fontsize=12, labelpad=10)
    plt.subplots_adjust(left=0.28, right=0.95, top=0.88, bottom=0.12)
    plt.show()