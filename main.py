
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from datetime import datetime
import numpy as np
from scipy import stats
import logging

# --- Configuración de logging ---
logging.basicConfig(filename='eda_log.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuración inicial ---
output_dir = 'graficos'
os.makedirs(output_dir, exist_ok=True)

# --- Funciones de validación ---
def validate_data(df, required_cols):
    """Valida que el dataset contenga las columnas requeridas y analiza valores inválidos."""
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Falta la columna requerida: {col}")
    if not pd.api.types.is_numeric_dtype(df['TIME_PERIOD']):
        raise ValueError("TIME_PERIOD debe ser numérico")
    
    # Verificar valores negativos en OBS_VALUE
    negative_values = df[df['OBS_VALUE'].lt(0)]
    if not negative_values.empty:
        logging.warning(f"Se encontraron {len(negative_values)} valores negativos en OBS_VALUE")
        negative_values[['geo', 'Sector_name', 'OBS_VALUE', 'TIME_PERIOD']].to_csv(
            os.path.join(output_dir, 'negative_values.csv'), index=False)
    return negative_values

# --- Funciones de análisis de datos ---
def get_summary_stats(df):
    """Genera estadísticas descriptivas y calidad de datos."""
    summary = {
        'Total registros': len(df),
        'Total columnas': len(df.columns),
        'Columnas numéricas': [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])],
        'TIME_PERIOD': {
            'Media': df['TIME_PERIOD'].mean(),
            'Mediana': df['TIME_PERIOD'].median(),
            'Desviación estándar': df['TIME_PERIOD'].std(),
            'Rango': df['TIME_PERIOD'].max() - df['TIME_PERIOD'].min(),
            'IQR': df['TIME_PERIOD'].quantile(0.75) - df['TIME_PERIOD'].quantile(0.25)
        },
        'OBS_VALUE': {
            'Media': df['OBS_VALUE'].mean(),
            'Mediana': df['OBS_VALUE'].median(),
            'Desviación estándar': df['OBS_VALUE'].std(),
            'Rango': df['OBS_VALUE'].max() - df['OBS_VALUE'].min(),
            'IQR': df['OBS_VALUE'].quantile(0.75) - df['OBS_VALUE'].quantile(0.25)
        },
        'Valores faltantes': df.isnull().sum().to_dict(),
        'Outliers': {
            'TIME_PERIOD': len(df[np.abs(stats.zscore(df['TIME_PERIOD'], nan_policy='omit')) > 3]),
            'OBS_VALUE': len(df[np.abs(stats.zscore(df['OBS_VALUE'], nan_policy='omit')) > 3])
        }
    }
    return summary

def get_top_correlations(corr_matrix, threshold=0.8):
    """Identifica las correlaciones más significativas."""
    corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                corr_pairs.append({
                    'País 1': corr_matrix.columns[i],
                    'País 2': corr_matrix.columns[j],
                    'Correlación': corr_matrix.iloc[i, j]
                })
    return pd.DataFrame(corr_pairs)

# --- Funciones de visualización ---
def generate_histogram(df, output_dir):
    plt.figure(figsize=(10, 6))
    sns.histplot(df['OBS_VALUE'], bins=50, kde=True)
    plt.title('Distribución de Emisiones (Gg CO2 equivalente)')
    plt.xlabel('OBS_VALUE')
    plt.ylabel('Frecuencia')
    plt.yscale('log')
    plt.savefig(os.path.join(output_dir, 'histograma_obs_value.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Gráfico interactivo
    fig = px.histogram(df, x='OBS_VALUE', nbins=50, log_y=True, title='Distribución de Emisiones (Interactivo)')
    fig.write_html(os.path.join(output_dir, 'histograma_interactivo.html'))

def generate_boxplot_sector(df, output_dir):
    top_sectors = df['Sector_name'].value_counts().nlargest(10).index
    df_top_sectors = df[df['Sector_name'].isin(top_sectors)]
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='OBS_VALUE', y='Sector_name', data=df_top_sectors, palette='viridis')
    plt.title('Distribución de Emisiones por Top 10 Sectores')
    plt.xlabel('OBS_VALUE (Gg CO2 equivalente)')
    plt.ylabel('Sector')
    plt.xscale('log')
    plt.savefig(os.path.join(output_dir, 'boxplot_obs_value_sector.png'), dpi=300, bbox_inches='tight')
    plt.close()

def generate_boxplot_country(df, output_dir):
    top_countries = df['geo'].value_counts().nlargest(10).index
    df_top_countries = df[df['geo'].isin(top_countries)]
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='OBS_VALUE', y='geo', data=df_top_countries, palette='plasma')
    plt.title('Distribución de Emisiones por Top 10 Países')
    plt.xlabel('OBS_VALUE (Gg CO2 equivalente)')
    plt.ylabel('País')
    plt.xscale('log')
    plt.savefig(os.path.join(output_dir, 'boxplot_obs_value_pais.png'), dpi=300, bbox_inches='tight')
    plt.close()

def generate_time_series(df, output_dir):
    plt.figure(figsize=(14, 7))
    mean_emissions = df.groupby('TIME_PERIOD')['OBS_VALUE'].mean()
    sns.lineplot(x=mean_emissions.index, y=mean_emissions.values, marker='o')
    plt.title('Evolución de Emisiones Promedio (2010-2023)')
    plt.xlabel('Año')
    plt.ylabel('Media de OBS_VALUE (Gg CO2 equivalente)')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'serie_temporal_obs_value.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Gráfico interactivo
    fig = px.line(x=mean_emissions.index, y=mean_emissions.values, title='Evolución de Emisiones (Interactivo)')
    fig.update_layout(xaxis_title='Año', yaxis_title='Media de OBS_VALUE (Gg CO2 equivalente)')
    fig.write_html(os.path.join(output_dir, 'serie_temporal_interactiva.html'))

def generate_correlation_heatmap(df, output_dir):
    emissions_pivot = df.pivot_table(index='TIME_PERIOD', columns='geo', values='OBS_VALUE', aggfunc='sum')
    if emissions_pivot.shape[1] > 1:
        corr_matrix = emissions_pivot.corr()
        plt.figure(figsize=(16, 14))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
        plt.title('Correlación de Emisiones entre Países')
        plt.savefig(os.path.join(output_dir, 'heatmap_correlacion.png'), dpi=300, bbox_inches='tight')
        plt.close()
        return corr_matrix
    return None

def generate_outliers_analysis(df, output_dir):
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='geo', y='OBS_VALUE', data=df)
    plt.xticks(rotation=45)
    plt.title('Valores Atípicos de Emisiones por País')
    plt.xlabel('País')
    plt.ylabel('OBS_VALUE (Gg CO2 equivalente)')
    plt.yscale('log')
    plt.savefig(os.path.join(output_dir, 'outliers_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

def generate_top_sectors(df, output_dir):
    top_5_sectors = df.groupby('Sector_name')['OBS_VALUE'].mean().nlargest(5)
    df_top_5 = df[df['Sector_name'].isin(top_5_sectors.index)]
    plt.figure(figsize=(12, 6))
    sns.barplot(x='OBS_VALUE', y='Sector_name', data=df_top_5, palette='magma')
    plt.title('Top 5 Sectores Contaminantes')
    plt.xlabel('Emisiones Promedio (Gg CO2 equivalente)')
    plt.ylabel('Sector')
    plt.savefig(os.path.join(output_dir, 'top_sectores.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Gráfico de pastel
    plt.figure(figsize=(8, 8))
    plt.pie(top_5_sectors, labels=top_5_sectors.index, autopct='%1.1f%%', colors=sns.color_palette('magma', 5))
    plt.title('Contribución de Top 5 Sectores a Emisiones Totales')
    plt.savefig(os.path.join(output_dir, 'pie_top_sectores.png'), dpi=300, bbox_inches='tight')
    plt.close()

def generate_regional_comparison(df, output_dir):
    nordic_countries = ['Denmark', 'Finland', 'Sweden']
    southern_countries = ['Spain', 'Italy', 'Portugal', 'Greece']
    df_regions = df[df['geo'].isin(nordic_countries + southern_countries)].copy()
    df_regions['Region'] = df_regions['geo'].apply(lambda x: 'Norte' if x in nordic_countries else 'Sur')
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Region', y='OBS_VALUE', data=df_regions)
    plt.title('Comparación de Emisiones: Norte vs Sur')
    plt.xlabel('Región')
    plt.ylabel('OBS_VALUE (Gg CO2 equivalente)')
    plt.yscale('log')
    plt.savefig(os.path.join(output_dir, 'comparacion_regional.png'), dpi=300, bbox_inches='tight')
    plt.close()

def generate_nordic_trend(df, output_dir):
    nordic_countries = ['Denmark', 'Finland', 'Sweden']
    df_nordic = df[df['geo'].isin(nordic_countries)]
    plt.figure(figsize=(14, 7))
    sns.lineplot(x='TIME_PERIOD', y='OBS_VALUE', hue='geo', data=df_nordic)
    plt.title('Evolución de Emisiones en Países Nórdicos')
    plt.xlabel('Año')
    plt.ylabel('OBS_VALUE (Gg CO2 equivalente)')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'nordicos_evolucion.png'), dpi=300, bbox_inches='tight')
    plt.close()

def generate_scatter_plot(df, output_dir):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='TIME_PERIOD', y='OBS_VALUE', hue='geo', size='OBS_VALUE', data=df, sizes=(20, 200))
    plt.title('Dispersión de Emisiones por Año y País')
    plt.xlabel('Año')
    plt.ylabel('OBS_VALUE (Gg CO2 equivalente)')
    plt.yscale('log')
    plt.savefig(os.path.join(output_dir, 'scatter_plot.png'), dpi=300, bbox_inches='tight')
    plt.close()

# --- Cargar y limpiar datos ---
try:
    df = pd.read_csv("data.csv")
except FileNotFoundError:
    logging.error("El archivo data.csv no se encuentra en el directorio")
    raise FileNotFoundError("El archivo data.csv no se encuentra en el directorio")

required_cols = ['OBS_VALUE', 'Sector_name', 'geo', 'TIME_PERIOD']
negative_values = validate_data(df, required_cols)

# Filtrar países de la UE
eu_countries_iso = [
    'AT', 'BE', 'BG', 'HR', 'CY', 'CZ', 'DK', 'EE', 'FI', 'FR', 'DE', 'EL',
    'HU', 'IE', 'IT', 'LV', 'LT', 'LU', 'MT', 'NL', 'PL', 'PT', 'RO', 'SK',
    'SI', 'ES', 'SE'
]
iso_to_country_name = {
    'AT': 'Austria', 'BE': 'Belgium', 'BG': 'Bulgaria', 'HR': 'Croatia', 'CY': 'Cyprus',
    'CZ': 'Czech Republic', 'DK': 'Denmark', 'EE': 'Estonia', 'FI': 'Finland',
    'FR': 'France', 'DE': 'Germany', 'EL': 'Greece', 'HU': 'Hungary', 'IE': 'Ireland',
    'IT': 'Italy', 'LV': 'Latvia', 'LT': 'Lithuania', 'LU': 'Luxembourg', 'MT': 'Malta',
    'NL': 'Netherlands', 'PL': 'Poland', 'PT': 'Portugal', 'RO': 'Romania',
    'SK': 'Slovakia', 'SI': 'Slovenia', 'ES': 'Spain', 'SE': 'Sweden'
}

df_eu = df[df['geo'].isin(eu_countries_iso)].copy()
df_eu['geo'] = df_eu['geo'].map(iso_to_country_name)
df_eu['OBS_VALUE'] = pd.to_numeric(df_eu['OBS_VALUE'], errors='coerce')

# Filtrar valores negativos y registrar
df_eu = df_eu[df_eu['OBS_VALUE'] >= 0]
logging.info(f"Se filtraron {len(negative_values)} valores negativos de OBS_VALUE")
df_eu.dropna(subset=['OBS_VALUE', 'Sector_name', 'geo', 'TIME_PERIOD'], inplace=True)

# --- Generar estadísticas ---
summary_stats = get_summary_stats(df_eu)
pd.DataFrame(summary_stats['OBS_VALUE'], index=['Media', 'Mediana', 'Desviación estándar', 'Rango', 'IQR']).to_html(os.path.join(output_dir, 'obs_value_stats.html'))
pd.DataFrame(summary_stats['Valores faltantes'], index=['Valores faltantes']).T.to_html(os.path.join(output_dir, 'missing_data.html'))

# --- Generar visualizaciones ---
generate_histogram(df_eu, output_dir)
generate_boxplot_sector(df_eu, output_dir)
generate_boxplot_country(df_eu, output_dir)
generate_time_series(df_eu, output_dir)
corr_matrix = generate_correlation_heatmap(df_eu, output_dir)
generate_outliers_analysis(df_eu, output_dir)
generate_top_sectors(df_eu, output_dir)
generate_regional_comparison(df_eu, output_dir)
generate_nordic_trend(df_eu, output_dir)
generate_scatter_plot(df_eu, output_dir)

# --- Generar correlaciones significativas ---
if corr_matrix is not None:
    top_correlations = get_top_correlations(corr_matrix)
    top_correlations.to_html(os.path.join(output_dir, 'top_correlations.html'))

# --- Generar HTML ---
current_date = datetime.now().strftime("%Y-%m-%d %H:%M")
html_content = f"""
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Informe de Análisis de Emisiones</title>
    <style>
        :root {{
            --title-color: #0d2a0f;
            --primary-color: #114518;
            --secondary-color: #2e7d32;
            --background-color: #f0f7f0;
        }}
        body {{
            font-family: 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: var(--primary-color);
            margin: 0;
            padding: 0;
        }}
        h1, h2 {{
            color: var(--title-color);
        }}
        h3 {{
            color: var(--primary-color);
        }}
        .content-wrapper {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: var(--background-color);
        }}
        .report-header {{
            border-bottom: 2px solid #e1e1e1;
            margin-bottom: 30px;
            padding-bottom: 20px;
        }}
        .chart-figure {{
            margin: 40px 0;
            padding: 20px;
            background: #e8f5e9;
            border: 1px solid #c8e6c9;
            border-radius: 5px;
        }}
        .chart-figure img {{
            max-width: 100%;
            height: auto;
            display: block;
            margin: 0 auto;
        }}
        figcaption {{
            font-style: italic;
            margin-top: 10px;
cdb
            color: #666;
            text-align: center;
        }}
        .report-footer {{
            margin-top: 50px;
            text-align: center;
            font-size: 0.85em;
            color: #666;
            border-top: 1px solid #ddd;
            padding: 15px 0;
            background-color: white;
        }}
        .badge {{
            background-color: var(--title-color);
            color: white;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.7em;
            vertical-align: middle;
        }}
        .eu-logo {{
            margin-right: 15px;
            filter: drop-shadow(0 2px 4px rgba(0,0,0,0.1));
            max-width: 60px;
        }}
        .data-methodology summary {{
            color: var(--title-color);
            cursor: pointer;
        }}
        .data-methodology summary:hover {{
            color: var(--secondary-color);
        }}
        .data-methodology {{
            background: white;
            padding: 15px;
            border-radius: 4px;
            margin-bottom: 25px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}
        .section-intro {{
            color: #555;
            margin-bottom: 30px;
            max-width: 800px;
        }}
        .comparison-section {{
            margin-top: 40px;
            padding: 20px;
            background: #e8f5e9;
            border-radius: 5px;
        }}
        .comparison-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }}
        th {{
            background-color: var(--primary-color);
            color: white;
        }}
        .conclusions, .data-quality {{
            margin-top: 40px;
            padding: 20px;
            background: #e8f5e9;
            border-radius: 5px;
        }}
        @media (max-width: 768px) {{
            .content-wrapper {{
                padding: 10px;
            }}
            .eu-logo {{
                max-width: 40px;
            }}
            .report-header {{
                padding-bottom: 15px;
            }}
            .chart-figure {{
                padding: 10px;
            }}
        }}
        @media (max-width: 480px) {{
            h1 {{
                font-size: 1.5em;
            }}
            h2 {{
                font-size: 1.2em;
            }}
            .report-meta {{
                font-size: 0.9em;
            }}
        }}
    </style>
</head>
<body>
    <div class="content-wrapper">
        <header class="report-header">
            <div class="header-content" style="display: flex; align-items: center;">
                <div class="logo-container">
                    <img src="graficos/eu-logo.png" alt="Logo UE" class="eu-logo">
                </div>
                <div>
                    <h1>Análisis de Emisiones GHG - Unión Europea <span class="badge">2023</span></h1>
                    <p class="report-meta">Generado el: {current_date} | fuente: EEA</p>
                </div>
            </div>
        </header>
        <main>
            <section aria-labelledby="summary-section">
                <h2 id="summary-section">Resumen General</h2>
                <p>El dataset analizado contiene información sobre emisiones de gases de efecto invernadero con:</p>
                <ul>
                    <li>Total de registros: {summary_stats['Total registros']}</li>
                    <li>Total de columnas: {summary_stats['Total columnas']}</li>
                    <li>Columnas numéricas: {', '.join(summary_stats['Columnas numéricas'])}</li>
                </ul>
                <h3>Estadísticas Descriptivas</h3>
                <h4>Columna: OBS_VALUE</h4>
                {open(os.path.join(output_dir, 'obs_value_stats.html')).read()}
                <h4>Calidad de Datos</h4>
                {open(os.path.join(output_dir, 'missing_data.html')).read()}
                <div class="data-quality">
                    <h4>Valores Negativos</h4>
                    <p>Se detectaron {len(negative_values)} valores negativos en OBS_VALUE, que fueron filtrados para este análisis. Ver detalles en <a href="graficos/negative_values.csv">negative_values.csv</a>.</p>
                </div>
                <h4>Detección de Outliers</h4>
                <ul>
                    <li>TIME_PERIOD: {summary_stats['Outliers']['TIME_PERIOD']} valores atípicos detectados (Z > 3)</li>
                    <li>OBS_VALUE: {summary_stats['Outliers']['OBS_VALUE']} valores atípicos detectados (Z > 3)</li>
                </ul>
            </section>
            <section aria-labelledby="viz-section">
                <h2 id="viz-section">Visualizaciones Clave</h2>
                <details class="data-methodology">
                    <summary>Metodología y Fuentes</summary>
                    <p>El siguiente análisis visual representa las emisiones de gases de efecto invernadero en la Unión Europea, medida en Gigagramos de CO2 equivalente (Gg CO2eq). Los datos provienen de la Agencia Europea de Medio Ambiente (EEA) y cubren el período 2010-2023 utilizando la clasificación IPCC 2006.</p>
                    <ul>
                        <li><strong>Fuente de datos:</strong> European Environment Agency (EEA)</li>
                        <li><strong>Última actualización:</strong> July 2025</li>
                        <li><strong>Métrica:</strong> Emisiones brutas (inventario nacional)</li>
                    </ul>
                </details>
                <figure class="chart-figure">
                    <h3>Distribución de Emisiones (OBS_VALUE)</h3>
                    <img src="graficos/histograma_obs_value.png" alt="Histograma mostrando distribución de emisiones en Gg CO2 equivalentes" aria-describedby="fig1-caption">
                    <figcaption id="fig1-caption">Figura 1: Distribución logarítmica de valores de emisión</figcaption>
                    <iframe src="graficos/histograma_interactivo.html" width="100%" height="400px" frameborder="0" aria-label="Histograma interactivo de emisiones"></iframe>
                </figure>
                <figure class="chart-figure">
                    <h3>Distribución Sectorial</h3>
                    <img src="graficos/boxplot_obs_value_sector.png" alt="Diagrama de caja por sectores económicos" aria-describedby="fig2-caption">
                    <figcaption id="fig2-caption">Figura 2: Comparativa de emisiones por sector industrial</figcaption>
                </figure>
                <figure class="chart-figure">
                    <h3>Comparativa por País</h3>
                    <img src="graficos/boxplot_obs_value_pais.png" alt="Diagrama de caja por países" aria-describedby="fig3-caption">
                    <figcaption id="fig3-caption">Figura 3: Rangos de emisión por nación (Gg CO2)</figcaption>
                </figure>
                <figure class="chart-figure">
                    <h3>Evolución Temporal</h3>
                    <img src="graficos/serie_temporal_obs_value.png" alt="Tendencias anuales de emisiones" aria-describedby="fig4-caption">
                    <figcaption id="fig4-caption">Figura 4: Cambios en emisiones promedio (2010-2023)</figcaption>
                    <iframe src="graficos/serie_temporal_interactiva.html" width="100%" height="400px" frameborder="0" aria-label="Serie temporal interactiva de emisiones"></iframe>
                </figure>
                <figure class="chart-figure">
                    <h3>Matriz de Correlación de Emisiones</h3>
                    <img src="graficos/heatmap_correlacion.png" alt="Mapa de calor de correlaciones entre países" aria-describedby="fig5-caption">
                    <figcaption id="fig5-caption">Figura 5: Correlación de emisiones entre países europeos (valores cercanos a 1 indican patrones similares, -1 indican patrones opuestos)</figcaption>
                    <h4>Correlaciones Significativas (>0.8)</h4>
                    {open(os.path.join(output_dir, 'top_correlations.html')).read() if corr_matrix is not None else '<p>No hay correlaciones significativas.</p>'}
                </figure>
                <figure class="chart-figure">
                    <h3>Análisis de Valores Atípicos</h3>
                    <img src="graficos/outliers_analysis.png" alt="Distribución de valores atípicos" aria-describedby="fig6-caption">
                    <figcaption id="fig6-caption">Figura 6: Países y sectores con mayor presencia de valores extremos</figcaption>
                </figure>
                <figure class="chart-figure">
                    <h3>Top 5 Sectores Contaminantes</h3>
                    <img src="graficos/top_sectores.png" alt="Sectores con mayores emisiones" aria-describedby="fig7-caption">
                    <figcaption id="fig7-caption">Figura 7: Sectores que generan mayor cantidad de emisiones</figcaption>
                    <img src="graficos/pie_top_sectores.png" alt="Contribución porcentual de sectores" aria-describedby="fig8-caption">
                    <figcaption id="fig8-caption">Figura 8: Contribución porcentual de los top 5 sectores</figcaption>
                </figure>
                <figure class="chart-figure">
                    <h3>Dispersión de Emisiones</h3>
                    <img src="graficos/scatter_plot.png" alt="Dispersión de emisiones por año y país" aria-describedby="fig9-caption">
                    <figcaption id="fig9-caption">Figura 9: Relación entre emisiones, años y países</figcaption>
                </figure>
            </section>
            <section class="comparison-section">
                <h2>Comparación Regional</h2>
                <div class="comparison-grid">
                    <figure class="chart-figure">
                        <h3>Emisiones Norte vs Sur</h3>
                        <img src="graficos/comparacion_regional.png" alt="Comparación entre regiones europeas" aria-describedby="fig10-caption">
                        <figcaption id="fig10-caption">Figura 10: Comparativa de emisiones entre regiones norte y sur</figcaption>
                    </figure>
                    <figure class="chart-figure">
                        <h3>Evolución Países Nórdicos</h3>
                        <img src="graficos/nordicos_evolucion.png" alt="Tendencias en países nórdicos" aria-describedby="fig11-caption">
                        <figcaption id="fig11-caption">Figura 11: Tendencias de emisiones en países nórdicos</figcaption>
                    </figure>
                </div>
            </section>
            <section class="conclusions">
                <h2>Conclusiones</h2>
                <ul>
                    <li>Variabilidad significativa en las emisiones entre diferentes sectores, con {df_eu.groupby('Sector_name')['OBS_VALUE'].mean().idxmax()} como el sector más contaminante.</li>
                    <li>Se detectaron {summary_stats['Outliers']['OBS_VALUE']} valores atípicos en OBS_VALUE que requieren investigación adicional.</li>
                    <li>El país con mayores emisiones promedio es {df_eu.groupby('geo')['OBS_VALUE'].mean().idxmax()}.</li>
                    <li>La tendencia general de emisiones es {'decreciente' if df_eu.groupby('TIME_PERIOD')['OBS_VALUE'].mean().iloc[-1] < df_eu.groupby('TIME_PERIOD')['OBS_VALUE'].mean().iloc[0] else 'creciente'} desde 2010.</li>
                    <li>Se filtraron {len(negative_values)} valores negativos en OBS_VALUE, posiblemente debido a errores de datos o ajustes por captura de carbono.</li>
                </ul>
            </section>
        </main>
        <footer class="report-footer">
            <p>Generado automáticamente con Python | {current_date}</p>
        </footer>
    </div>
</body>
</html>
"""

# Guardar el HTML
with open('informe_emisiones.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

print("Informe HTML y gráficos generados exitosamente.")
logging.info("Informe HTML y gráficos generados exitosamente.")
