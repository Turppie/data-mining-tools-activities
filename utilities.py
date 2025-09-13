"""
🔧 EDA Utilities - Pipeline de Análisis Exploratorio de Datos
===========================================================

Funciones robustas y reutilizables para análisis univariado, multivariado 
y discretización de variables basadas en el pipeline del dataset Titanic.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import (
    skew, kurtosis, entropy, shapiro, normaltest, 
    spearmanr, pearsonr, kendalltau, ttest_ind, 
    f_oneway, mannwhitneyu, kruskal, chi2_contingency
)
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 📊 ANÁLISIS UNIVARIADO - VARIABLES CUANTITATIVAS
# ============================================================================

def analisis_cuantitativo_completo(df, columnas=None, figsize=(15, 10)):
    """
    Análisis univariado completo para variables cuantitativas.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos
    columnas : list, optional
        Lista de columnas a analizar. Si None, analiza todas las numéricas
    figsize : tuple
        Tamaño de las figuras
    
    Returns:
    --------
    pd.DataFrame : Resumen estadístico completo
    """
    if columnas is None:
        columnas = df.select_dtypes(include=[np.number]).columns.tolist()
    
    resultados = []
    
    for col in columnas:
        serie = df[col].dropna()
        if len(serie) == 0:
            continue
            
        # Métricas básicas
        media = serie.mean()
        mediana = serie.median()
        moda = serie.mode().iloc[0] if not serie.mode().empty else np.nan
        
        # Dispersión
        desv_std = serie.std()
        varianza = serie.var()
        rango = serie.max() - serie.min()
        iqr = serie.quantile(0.75) - serie.quantile(0.25)
        mad = np.median(np.abs(serie - mediana))
        cv = (desv_std / media * 100) if media != 0 else np.nan
        
        # Forma de distribución
        asimetria = skew(serie, bias=False)
        curtosis_val = kurtosis(serie, fisher=True, bias=False)
        
        # Pruebas de normalidad
        if len(serie) <= 5000:
            _, p_shapiro = shapiro(serie.sample(min(len(serie), 5000), random_state=42))
        else:
            p_shapiro = np.nan
        _, p_dagostino = normaltest(serie)
        
        # Outliers (método IQR)
        q1, q3 = serie.quantile([0.25, 0.75])
        limite_inf = q1 - 1.5 * iqr
        limite_sup = q3 + 1.5 * iqr
        outliers = ((serie < limite_inf) | (serie > limite_sup)).sum()
        
        resultados.append({
            'Variable': col,
            'N_válidos': len(serie),
            'N_missing': df[col].isna().sum(),
            'Media': media,
            'Mediana': mediana,
            'Moda': moda,
            'Desv_Std': desv_std,
            'Varianza': varianza,
            'Rango': rango,
            'IQR': iqr,
            'MAD': mad,
            'CV_%': cv,
            'Asimetría': asimetria,
            'Curtosis': curtosis_val,
            'Outliers_IQR': outliers,
            'p_Shapiro': p_shapiro,
            'p_DAgostino': p_dagostino,
            'Normal_Shapiro': 'Sí' if p_shapiro > 0.05 else 'No' if not pd.isna(p_shapiro) else 'N/A',
            'Normal_DAgostino': 'Sí' if p_dagostino > 0.05 else 'No'
        })
    
    return pd.DataFrame(resultados).round(4)

def visualizar_cuantitativas(df, columnas=None, figsize=(15, 12)):
    """
    Visualización completa para variables cuantitativas.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos
    columnas : list, optional
        Lista de columnas a analizar
    figsize : tuple
        Tamaño de la figura
    """
    if columnas is None:
        columnas = df.select_dtypes(include=[np.number]).columns.tolist()
    
    n_cols = len(columnas)
    n_rows = (n_cols + 2) // 3  # 3 columnas por fila
    
    fig, axes = plt.subplots(n_rows, 3, figsize=figsize)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for i, col in enumerate(columnas):
        serie = df[col].dropna()
        
        # Histograma + KDE
        sns.histplot(serie, kde=True, ax=axes[i], alpha=0.7)
        axes[i].axvline(serie.mean(), color='red', linestyle='--', alpha=0.8, label=f'Media: {serie.mean():.2f}')
        axes[i].axvline(serie.median(), color='green', linestyle='--', alpha=0.8, label=f'Mediana: {serie.median():.2f}')
        axes[i].set_title(f'{col}\nAsimetría: {skew(serie, bias=False):.3f} | Curtosis: {kurtosis(serie, fisher=True, bias=False):.3f}')
        axes[i].legend(fontsize=8)
    
    # Ocultar subplots vacíos
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
    # Boxplots separados
    if len(columnas) > 1:
        fig, ax = plt.subplots(figsize=(max(8, len(columnas) * 1.5), 6))
        df[columnas].boxplot(ax=ax)
        ax.set_title('Distribución y Outliers - Boxplots')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

# ============================================================================
# 📊 ANÁLISIS UNIVARIADO - VARIABLES CUALITATIVAS
# ============================================================================

def analisis_cualitativo_completo(df, columnas=None):
    """
    Análisis univariado completo para variables cualitativas.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos
    columnas : list, optional
        Lista de columnas categóricas a analizar
        
    Returns:
    --------
    dict : Diccionario con resúmenes por variable
    """
    if columnas is None:
        columnas = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    resultados = {}
    
    for col in columnas:
        serie = df[col].dropna()
        if len(serie) == 0:
            continue
            
        # Frecuencias
        freq_abs = serie.value_counts()
        freq_rel = serie.value_counts(normalize=True)
        
        # Diversidad
        k = len(freq_rel)  # número de categorías
        H = entropy(freq_rel, base=2)  # entropía Shannon
        H_max = np.log2(k) if k > 1 else 0
        G = 1 - (freq_rel**2).sum()  # índice Gini
        G_max = 1 - 1/k if k > 1 else 0
        
        resultados[col] = {
            'n_valid': len(serie),
            'n_missing': df[col].isna().sum(),
            'n_categories': k,
            'moda': freq_abs.index[0],
            'freq_moda': freq_abs.iloc[0],
            'prop_moda': freq_rel.iloc[0],
            'entropia_bits': H,
            'entropia_max': H_max,
            'entropia_norm': H / H_max if H_max > 0 else 0,
            'gini': G,
            'gini_max': G_max,
            'gini_norm': G / G_max if G_max > 0 else 0,
            'freq_abs': freq_abs,
            'freq_rel': freq_rel
        }
    
    return resultados

def visualizar_cualitativas(df, columnas=None, figsize=(15, 10), max_categories=10):
    """
    Visualización para variables cualitativas.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos
    columnas : list, optional
        Lista de columnas categóricas
    figsize : tuple
        Tamaño de la figura
    max_categories : int
        Máximo número de categorías a mostrar (agrupa el resto en "Otros")
    """
    if columnas is None:
        columnas = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    n_cols = len(columnas)
    n_rows = (n_cols + 1) // 2  # 2 columnas por fila
    
    fig, axes = plt.subplots(n_rows, 2, figsize=figsize)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for i, col in enumerate(columnas):
        serie = df[col].dropna()
        freq = serie.value_counts()
        
        # Agrupar categorías raras en "Otros"
        if len(freq) > max_categories:
            top_cats = freq.head(max_categories - 1)
            otros = freq.tail(len(freq) - max_categories + 1).sum()
            freq_plot = pd.concat([top_cats, pd.Series({'Otros': otros})])
        else:
            freq_plot = freq
        
        # Gráfico de barras
        freq_plot.plot(kind='bar', ax=axes[i], alpha=0.8)
        axes[i].set_title(f'{col}\nCategorías: {len(freq)} | Entropía: {entropy(serie.value_counts(normalize=True), base=2):.2f} bits')
        axes[i].set_xlabel('Categorías')
        axes[i].set_ylabel('Frecuencia')
        axes[i].tick_params(axis='x', rotation=45)
    
    # Ocultar subplots vacíos
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.show()

# ============================================================================
# 🔗 ANÁLISIS MULTIVARIADO
# ============================================================================

def correlaciones_completas(df, columnas=None, metodos=['pearson', 'spearman', 'kendall']):
    """
    Análisis completo de correlaciones con p-valores.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos
    columnas : list, optional
        Lista de columnas numéricas a analizar
    metodos : list
        Métodos de correlación a calcular
        
    Returns:
    --------
    dict : Diccionario con matrices de correlación y p-valores
    """
    if columnas is None:
        columnas = df.select_dtypes(include=[np.number]).columns.tolist()
    
    df_num = df[columnas].copy()
    resultados = {}
    
    for metodo in metodos:
        if metodo == 'pearson':
            corr_func = pearsonr
        elif metodo == 'spearman':
            corr_func = spearmanr
        elif metodo == 'kendall':
            corr_func = kendalltau
        else:
            continue
            
        # Matriz de correlaciones
        corr_matrix = df_num.corr(method=metodo)
        
        # Matriz de p-valores
        n_vars = len(columnas)
        p_matrix = np.zeros((n_vars, n_vars))
        
        for i in range(n_vars):
            for j in range(n_vars):
                if i == j:
                    p_matrix[i, j] = 0
                else:
                    # Eliminar NaN para el cálculo
                    col1, col2 = columnas[i], columnas[j]
                    common_idx = df_num[[col1, col2]].dropna().index
                    if len(common_idx) > 2:
                        _, p_val = corr_func(df_num.loc[common_idx, col1], 
                                           df_num.loc[common_idx, col2])
                        p_matrix[i, j] = p_val
                    else:
                        p_matrix[i, j] = np.nan
        
        p_values = pd.DataFrame(p_matrix, index=columnas, columns=columnas)
        
        resultados[metodo] = {
            'correlaciones': corr_matrix,
            'p_valores': p_values
        }
    
    return resultados

def resumen_correlaciones(df, columnas=None, metodo='spearman', alpha=0.05):
    """
    Resumen de correlaciones por pares con interpretación.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos
    columnas : list, optional
        Lista de columnas numéricas
    metodo : str
        Método de correlación ('pearson', 'spearman', 'kendall')
    alpha : float
        Nivel de significancia
        
    Returns:
    --------
    pd.DataFrame : Resumen de correlaciones por pares
    """
    correlaciones = correlaciones_completas(df, columnas, [metodo])
    corr_matrix = correlaciones[metodo]['correlaciones']
    p_matrix = correlaciones[metodo]['p_valores']
    
    # Crear lista de pares únicos
    pairs_data = []
    cols = corr_matrix.columns
    
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            var1, var2 = cols[i], cols[j]
            corr = corr_matrix.loc[var1, var2]
            p_val = p_matrix.loc[var1, var2]
            
            # Clasificar fuerza
            abs_corr = abs(corr)
            if abs_corr < 0.1:
                fuerza = "Muy débil"
            elif abs_corr < 0.3:
                fuerza = "Débil"
            elif abs_corr < 0.5:
                fuerza = "Moderada"
            elif abs_corr < 0.7:
                fuerza = "Fuerte"
            else:
                fuerza = "Muy fuerte"
                
            pairs_data.append({
                'Par_Variables': f"{var1} ↔ {var2}",
                f'{metodo.capitalize()}_ρ': corr,
                'p-valor': p_val,
                f'Significativo_α{str(alpha).replace("0.", "")}': 'Sí' if p_val < alpha else 'No',
                'Fuerza': fuerza
            })
    
    # Crear DataFrame y ordenar por p-valor
    df_resultado = pd.DataFrame(pairs_data)
    df_resultado = df_resultado.sort_values('p-valor')
    df_resultado[f'{metodo.capitalize()}_ρ'] = df_resultado[f'{metodo.capitalize()}_ρ'].round(4)
    df_resultado['p-valor'] = df_resultado['p-valor'].round(6)
    
    return df_resultado

def visualizar_correlaciones(df, columnas=None, metodo='spearman', figsize=(16, 12)):
    """
    Visualización completa de correlaciones.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos
    columnas : list, optional
        Lista de columnas numéricas
    metodo : str
        Método de correlación
    figsize : tuple
        Tamaño de la figura
    """
    correlaciones = correlaciones_completas(df, columnas, [metodo])
    corr_matrix = correlaciones[metodo]['correlaciones']
    p_matrix = correlaciones[metodo]['p_valores']
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Heatmap 1: Correlaciones
    sns.heatmap(corr_matrix, annot=True, fmt=".3f", cmap="coolwarm", 
                center=0, ax=axes[0], cbar_kws={'label': f'Correlación {metodo}'})
    axes[0].set_title(f'Correlaciones de {metodo.capitalize()}', fontweight='bold')
    
    # Heatmap 2: P-valores
    sns.heatmap(p_matrix, annot=True, fmt=".3f", cmap="Reds_r", 
                ax=axes[1], cbar_kws={'label': 'p-valor'})
    axes[1].set_title('P-valores', fontweight='bold')
    
    # Heatmap 3: Solo significativas
    mask_nonsig = p_matrix > 0.05
    sns.heatmap(corr_matrix, annot=True, fmt=".3f", cmap="coolwarm", 
                center=0, mask=mask_nonsig, ax=axes[2],
                cbar_kws={'label': f'Correlación {metodo} (p < 0.05)'})
    axes[2].set_title('Solo Correlaciones Significativas', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

# ============================================================================
# 🔄 COMPARACIÓN GRUPOS (CUANTI vs CUALI)
# ============================================================================

def comparar_grupos_cuanti(df, var_continua, var_categorica, alpha=0.05):
    """
    Comparación de grupos para variable continua vs categórica.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos
    var_continua : str
        Nombre de la variable continua
    var_categorica : str
        Nombre de la variable categórica
    alpha : float
        Nivel de significancia
        
    Returns:
    --------
    dict : Resultados de las pruebas estadísticas
    """
    # Preparar datos
    data_clean = df[[var_continua, var_categorica]].dropna()
    grupos = data_clean[var_categorica].unique()
    n_grupos = len(grupos)
    
    if n_grupos < 2:
        return {"error": "Se necesitan al menos 2 grupos"}
    
    resultados = {
        'variable_continua': var_continua,
        'variable_categorica': var_categorica,
        'n_grupos': n_grupos,
        'grupos': grupos.tolist()
    }
    
    # Separar datos por grupo
    grupos_data = []
    for grupo in grupos:
        grupo_datos = data_clean[data_clean[var_categorica] == grupo][var_continua].values
        grupos_data.append(grupo_datos)
        
        # Estadísticas descriptivas por grupo
        resultados[f'media_{grupo}'] = np.mean(grupo_datos)
        resultados[f'mediana_{grupo}'] = np.median(grupo_datos)
        resultados[f'n_{grupo}'] = len(grupo_datos)
    
    # Función para Cohen's d
    def cohens_d(x, y):
        n1, n2 = len(x), len(y)
        s1, s2 = np.var(x, ddof=1), np.var(y, ddof=1)
        s_p = np.sqrt(((n1-1)*s1 + (n2-1)*s2) / (n1+n2-2))
        return (np.mean(x) - np.mean(y)) / s_p
    
    if n_grupos == 2:
        # Pruebas para 2 grupos
        grupo1_data, grupo2_data = grupos_data[0], grupos_data[1]
        
        # T-test (Welch)
        stat_t, p_t = ttest_ind(grupo1_data, grupo2_data, equal_var=False)
        d_cohen = cohens_d(grupo1_data, grupo2_data)
        
        # Mann-Whitney U
        stat_mw, p_mw = mannwhitneyu(grupo1_data, grupo2_data, alternative='two-sided')
        
        resultados.update({
            'test_t_stat': stat_t,
            'test_t_pvalue': p_t,
            'test_t_significativo': p_t < alpha,
            'cohens_d': d_cohen,
            'mann_whitney_stat': stat_mw,
            'mann_whitney_pvalue': p_mw,
            'mann_whitney_significativo': p_mw < alpha
        })
        
    else:
        # Pruebas para k > 2 grupos
        # ANOVA
        stat_f, p_f = f_oneway(*grupos_data)
        
        # Eta cuadrado
        k = len(grupos_data)
        n = sum(len(g) for g in grupos_data)
        eta2 = (stat_f * (k - 1)) / (stat_f * (k - 1) + (n - k))
        
        # Kruskal-Wallis
        stat_kw, p_kw = kruskal(*grupos_data)
        
        resultados.update({
            'anova_f_stat': stat_f,
            'anova_pvalue': p_f,
            'anova_significativo': p_f < alpha,
            'eta_cuadrado': eta2,
            'kruskal_wallis_stat': stat_kw,
            'kruskal_wallis_pvalue': p_kw,
            'kruskal_wallis_significativo': p_kw < alpha
        })
    
    return resultados

def visualizar_grupos_cuanti(df, var_continua, var_categorica, figsize=(15, 5)):
    """
    Visualización de comparación de grupos.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos
    var_continua : str
        Variable continua
    var_categorica : str
        Variable categórica
    figsize : tuple
        Tamaño de la figura
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Boxplot
    sns.boxplot(data=df, x=var_categorica, y=var_continua, ax=axes[0])
    axes[0].set_title('Distribución por Grupo')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Histogramas superpuestos
    for grupo in df[var_categorica].dropna().unique():
        datos_grupo = df[df[var_categorica] == grupo][var_continua].dropna()
        axes[1].hist(datos_grupo, alpha=0.7, label=f'{grupo} (n={len(datos_grupo)})', density=True)
    axes[1].set_xlabel(var_continua)
    axes[1].set_ylabel('Densidad')
    axes[1].set_title('Distribuciones Superpuestas')
    axes[1].legend()
    
    # Violin plot
    sns.violinplot(data=df, x=var_categorica, y=var_continua, ax=axes[2])
    axes[2].set_title('Densidad por Grupo')
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

# ============================================================================
# 🔗 ANÁLISIS CATEGÓRICO vs CATEGÓRICO
# ============================================================================

def analizar_independencia_categoricas(df, var1, var2, alpha=0.05):
    """
    Análisis de independencia entre dos variables categóricas.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos
    var1, var2 : str
        Nombres de las variables categóricas
    alpha : float
        Nivel de significancia
        
    Returns:
    --------
    dict : Resultados del análisis de independencia
    """
    # Tabla de contingencia
    tabla_contingencia = pd.crosstab(df[var1], df[var2])
    
    # Chi-cuadrado
    chi2, p_valor, grados_libertad, frecuencias_esperadas = chi2_contingency(tabla_contingencia)
    
    # Cramér's V
    n = tabla_contingencia.sum().sum()
    r, k = tabla_contingencia.shape
    phi2 = chi2 / n
    cramers_v = np.sqrt(phi2 / min(r-1, k-1)) if min(r-1, k-1) > 0 else np.nan
    
    # Cramér's V corregido (bias correction)
    phi2_corr = max(0, phi2 - (k-1)*(r-1)/(n-1))
    denom_corr = min(k-1, r-1) - (k-1)*(r-1)/(n-1)
    cramers_v_corregido = np.sqrt(phi2_corr / denom_corr) if denom_corr > 0 else np.nan
    
    # Clasificar fuerza de asociación
    if cramers_v < 0.1:
        fuerza = "Muy débil"
    elif cramers_v < 0.3:
        fuerza = "Moderada"
    elif cramers_v < 0.5:
        fuerza = "Fuerte"
    else:
        fuerza = "Muy fuerte"
    
    # Porcentajes por fila
    porcentajes_fila = (tabla_contingencia.div(tabla_contingencia.sum(axis=1), axis=0) * 100).round(1)
    
    resultados = {
        'variables': f"{var1} ↔ {var2}",
        'tabla_contingencia': tabla_contingencia,
        'porcentajes_fila': porcentajes_fila,
        'chi2_estadistico': chi2,
        'p_valor': p_valor,
        'grados_libertad': grados_libertad,
        'cramers_v': cramers_v,
        'cramers_v_corregido': cramers_v_corregido,
        'significativo': p_valor < alpha,
        'fuerza_asociacion': fuerza,
        'n_total': n,
        'frecuencias_esperadas': pd.DataFrame(frecuencias_esperadas, 
                                            index=tabla_contingencia.index,
                                            columns=tabla_contingencia.columns)
    }
    
    return resultados

def visualizar_categoricas(df, var1, var2, figsize=(15, 5)):
    """
    Visualización para variables categóricas.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos
    var1, var2 : str
        Variables categóricas
    figsize : tuple
        Tamaño de la figura
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Tabla de contingencia como heatmap
    tabla = pd.crosstab(df[var1], df[var2])
    sns.heatmap(tabla, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title('Tabla de Contingencia')
    
    # Gráfico de barras agrupadas
    tabla.plot(kind='bar', ax=axes[1], alpha=0.8)
    axes[1].set_title('Frecuencias por Categoría')
    axes[1].set_xlabel(var1)
    axes[1].set_ylabel('Frecuencia')
    axes[1].legend(title=var2)
    axes[1].tick_params(axis='x', rotation=45)
    
    # Porcentajes por fila
    porcentajes = (tabla.div(tabla.sum(axis=1), axis=0) * 100)
    sns.heatmap(porcentajes, annot=True, fmt='.1f', cmap='Oranges', ax=axes[2])
    axes[2].set_title('Porcentajes por Fila (%)')
    
    plt.tight_layout()
    plt.show()

# ============================================================================
# 🔄 DISCRETIZACIÓN DE VARIABLES
# ============================================================================

def discretizar_variable(df, columna, metodos=['equal_width', 'quantiles', 'kmeans', 'supervised'], 
                        n_bins=5, target=None, **kwargs):
    """
    Discretización de una variable con múltiples métodos.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos
    columna : str
        Nombre de la columna a discretizar
    metodos : list
        Lista de métodos a aplicar
    n_bins : int
        Número de bins
    target : str, optional
        Variable objetivo para método supervisado
    **kwargs : dict
        Argumentos adicionales para DecisionTreeClassifier
        
    Returns:
    --------
    dict : Resultados de discretización por método
    """
    serie = df[columna].dropna()
    if len(serie) == 0:
        return {"error": "Serie vacía"}
    
    resultados = {
        'variable': columna,
        'n_valores': len(serie),
        'metodos': {}
    }
    
    for metodo in metodos:
        if metodo == 'equal_width':
            # Intervalos de igual ancho
            bins = pd.cut(serie, bins=n_bins, include_lowest=True)
            edges = pd.cut(serie, bins=n_bins, include_lowest=True, retbins=True)[1]
            
            resultados['metodos']['equal_width'] = {
                'bins': bins,
                'edges': edges,
                'frecuencias': bins.value_counts().sort_index(),
                'descripcion': f"Intervalos de igual ancho ({n_bins} bins)"
            }
            
        elif metodo == 'quantiles':
            # Cuantiles (igual frecuencia)
            bins = pd.qcut(serie, q=n_bins, duplicates='drop')
            edges = pd.qcut(serie, q=n_bins, duplicates='drop', retbins=True)[1]
            
            resultados['metodos']['quantiles'] = {
                'bins': bins,
                'edges': edges,
                'frecuencias': bins.value_counts().sort_index(),
                'descripcion': f"Cuantiles - igual frecuencia ({n_bins} bins)"
            }
            
        elif metodo == 'kmeans':
            # K-Means clustering
            discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='kmeans')
            bins_encoded = discretizer.fit_transform(serie.to_frame()).astype(int).ravel()
            edges = discretizer.bin_edges_[0]
            
            # Crear labels interpretables
            bin_labels = []
            for i in range(len(edges)-1):
                bin_labels.append(f"[{edges[i]:.2f}, {edges[i+1]:.2f})")
            
            resultados['metodos']['kmeans'] = {
                'bins_encoded': bins_encoded,
                'edges': edges,
                'bin_labels': bin_labels,
                'frecuencias': pd.Series(bins_encoded).value_counts().sort_index(),
                'descripcion': f"K-Means clustering ({n_bins} bins)"
            }
            
        elif metodo == 'supervised' and target is not None:
            # Método supervisado con árboles de decisión
            mask = df[[columna, target]].dropna().index
            X = df.loc[mask, [columna]].values
            y = df.loc[mask, target].values
            
            # Parámetros por defecto para el árbol
            tree_params = {
                'max_depth': 3,
                'min_samples_leaf': 50,
                'random_state': 42
            }
            tree_params.update(kwargs)
            
            clf = DecisionTreeClassifier(**tree_params)
            clf.fit(X, y)
            
            # Extraer umbrales de corte
            thresholds = sorted([t for t in clf.tree_.threshold if t != -2])
            
            # Crear bins usando los umbrales
            if thresholds:
                edges = [-np.inf] + thresholds + [np.inf]
                bins = pd.cut(serie, bins=edges, include_lowest=True)
                
                resultados['metodos']['supervised'] = {
                    'bins': bins,
                    'thresholds': thresholds,
                    'edges': edges,
                    'frecuencias': bins.value_counts().sort_index(),
                    'tree_params': tree_params,
                    'descripcion': f"Supervisado - Decision Tree (target: {target})"
                }
    
    return resultados

def visualizar_discretizacion(df, columna, resultados_discretizacion, figsize=(16, 12)):
    """
    Visualización de métodos de discretización.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos
    columna : str
        Nombre de la columna
    resultados_discretizacion : dict
        Resultados de discretizar_variable()
    figsize : tuple
        Tamaño de la figura
    """
    serie = df[columna].dropna()
    metodos = list(resultados_discretizacion['metodos'].keys())
    n_metodos = len(metodos)
    
    fig, axes = plt.subplots(2, n_metodos, figsize=figsize)
    if n_metodos == 1:
        axes = axes.reshape(-1, 1)
    
    fig.suptitle(f'Comparación de Métodos de Discretización - {columna}', fontsize=16, fontweight='bold')
    
    for i, metodo in enumerate(metodos):
        resultado = resultados_discretizacion['metodos'][metodo]
        
        # Fila 1: Histograma con líneas de corte
        axes[0, i].hist(serie, bins=30, alpha=0.7, density=True, color='lightblue')
        
        if metodo in ['equal_width', 'quantiles']:
            edges = resultado['edges']
            for edge in edges[1:-1]:
                axes[0, i].axvline(edge, color='red', linestyle='--', alpha=0.8)
        elif metodo == 'kmeans':
            edges = resultado['edges']
            for edge in edges[1:-1]:
                axes[0, i].axvline(edge, color='purple', linestyle='--', alpha=0.8)
        elif metodo == 'supervised':
            thresholds = resultado['thresholds']
            for threshold in thresholds:
                axes[0, i].axvline(threshold, color='darkred', linestyle='--', alpha=0.8)
        
        axes[0, i].set_title(f'{metodo.replace("_", " ").title()}')
        axes[0, i].set_xlabel(columna)
        axes[0, i].set_ylabel('Densidad')
        
        # Fila 2: Frecuencias por bin
        frecuencias = resultado['frecuencias']
        if metodo == 'kmeans':
            axes[1, i].bar(range(len(frecuencias)), frecuencias.values)
            axes[1, i].set_xlabel('Bin ID')
            axes[1, i].set_xticks(range(len(frecuencias)))
        else:
            frecuencias.plot(kind='bar', ax=axes[1, i], alpha=0.8)
            axes[1, i].tick_params(axis='x', rotation=45)
        
        axes[1, i].set_ylabel('Frecuencia')
        axes[1, i].set_title('Distribución de Frecuencias')
    
    plt.tight_layout()
    plt.show()

def comparar_discretizacion_supervivencia(df, columna, target, resultados_discretizacion):
    """
    Análisis de supervivencia por bins para métodos de discretización.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos
    columna : str
        Columna discretizada
    target : str
        Variable objetivo (0/1)
    resultados_discretizacion : dict
        Resultados de discretizar_variable()
        
    Returns:
    --------
    pd.DataFrame : Tasas de supervivencia por método y bin
    """
    data_clean = df[[columna, target]].dropna()
    resultados_supervivencia = []
    
    for metodo, resultado in resultados_discretizacion['metodos'].items():
        if metodo in ['equal_width', 'quantiles', 'supervised']:
            bins = resultado['bins']
            
            # Alinear con datos limpios
            bins_aligned = bins.loc[data_clean.index]
            
            for bin_name in bins_aligned.unique():
                if pd.isna(bin_name):
                    continue
                    
                mask = bins_aligned == bin_name
                grupo_data = data_clean[mask]
                
                if len(grupo_data) > 0:
                    tasa_supervivencia = grupo_data[target].mean()
                    resultados_supervivencia.append({
                        'Método': metodo,
                        'Bin': str(bin_name),
                        'N_pasajeros': len(grupo_data),
                        'Tasa_supervivencia': tasa_supervivencia,
                        'N_supervivientes': grupo_data[target].sum()
                    })
        
        elif metodo == 'kmeans':
            bins_encoded = resultado['bins_encoded']
            bin_labels = resultado['bin_labels']
            
            # Crear Series para facilitar análisis
            bins_series = pd.Series(bins_encoded, index=data_clean.index)
            
            for bin_id in sorted(pd.Series(bins_encoded).unique()):
                mask = bins_series == bin_id
                grupo_data = data_clean[mask]
                
                if len(grupo_data) > 0:
                    tasa_supervivencia = grupo_data[target].mean()
                    bin_label = bin_labels[bin_id] if bin_id < len(bin_labels) else f"Bin {bin_id}"
                    
                    resultados_supervivencia.append({
                        'Método': metodo,
                        'Bin': bin_label,
                        'N_pasajeros': len(grupo_data),
                        'Tasa_supervivencia': tasa_supervivencia,
                        'N_supervivientes': grupo_data[target].sum()
                    })
    
    df_supervivencia = pd.DataFrame(resultados_supervivencia)
    return df_supervivencia.round(3)

# ============================================================================
# 🎯 FUNCIONES DE PIPELINE COMPLETO
# ============================================================================

def pipeline_eda_completo(df, target=None, figsize_univariado=(15, 12), figsize_multivariado=(16, 8)):
    """
    Pipeline completo de EDA.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos
    target : str, optional
        Variable objetivo para análisis supervisado
    figsize_univariado, figsize_multivariado : tuple
        Tamaños de figuras
        
    Returns:
    --------
    dict : Resultados completos del EDA
    """
    print("🔍 INICIANDO PIPELINE EDA COMPLETO")
    print("=" * 60)
    
    # Información general del dataset
    print(f"\n📊 INFORMACIÓN GENERAL DEL DATASET")
    print(f"Dimensiones: {df.shape}")
    print(f"Variables numéricas: {len(df.select_dtypes(include=[np.number]).columns)}")
    print(f"Variables categóricas: {len(df.select_dtypes(include=['object', 'category']).columns)}")
    print(f"Valores faltantes: {df.isnull().sum().sum()}")
    
    resultados = {
        'info_general': {
            'dimensiones': df.shape,
            'n_numericas': len(df.select_dtypes(include=[np.number]).columns),
            'n_categoricas': len(df.select_dtypes(include=['object', 'category']).columns),
            'valores_faltantes': df.isnull().sum().sum()
        }
    }
    
    # 1. ANÁLISIS UNIVARIADO - CUANTITATIVAS
    print(f"\n📈 ANÁLISIS UNIVARIADO - VARIABLES CUANTITATIVAS")
    print("-" * 50)
    cols_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if target and target in cols_numericas:
        cols_numericas.remove(target)
    
    if cols_numericas:
        resultados['univariado_cuantitativo'] = analisis_cuantitativo_completo(df, cols_numericas)
        print("✅ Análisis cuantitativo completado")
        
        visualizar_cuantitativas(df, cols_numericas, figsize_univariado)
        print("✅ Visualizaciones cuantitativas generadas")
    
    # 2. ANÁLISIS UNIVARIADO - CUALITATIVAS
    print(f"\n📊 ANÁLISIS UNIVARIADO - VARIABLES CUALITATIVAS")
    print("-" * 50)
    cols_categoricas = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if cols_categoricas:
        resultados['univariado_cualitativo'] = analisis_cualitativo_completo(df, cols_categoricas)
        print("✅ Análisis cualitativo completado")
        
        visualizar_cualitativas(df, cols_categoricas, figsize_univariado)
        print("✅ Visualizaciones cualitativas generadas")
    
    # 3. ANÁLISIS MULTIVARIADO - CORRELACIONES
    if len(cols_numericas) > 1:
        print(f"\n🔗 ANÁLISIS MULTIVARIADO - CORRELACIONES")
        print("-" * 50)
        
        resultados['correlaciones'] = resumen_correlaciones(df, cols_numericas)
        print("✅ Análisis de correlaciones completado")
        
        visualizar_correlaciones(df, cols_numericas, figsize=figsize_multivariado)
        print("✅ Visualizaciones de correlaciones generadas")
    
    # 4. COMPARACIÓN DE GRUPOS (si hay target categórico)
    if target and target in cols_categoricas:
        print(f"\n⚖️ COMPARACIÓN DE GRUPOS CON TARGET: {target}")
        print("-" * 50)
        
        resultados['comparacion_grupos'] = {}
        
        for col in cols_numericas:
            resultado = comparar_grupos_cuanti(df, col, target)
            resultados['comparacion_grupos'][col] = resultado
            print(f"✅ Análisis {col} vs {target} completado")
        
        # Visualizaciones
        for col in cols_numericas[:4]:  # Limitar a 4 para no saturar
            visualizar_grupos_cuanti(df, col, target)
    
    # 5. ANÁLISIS DE INDEPENDENCIA (categóricas vs categóricas)
    if len(cols_categoricas) > 1:
        print(f"\n🔗 ANÁLISIS DE INDEPENDENCIA - VARIABLES CATEGÓRICAS")
        print("-" * 50)
        
        resultados['independencia_categoricas'] = {}
        
        for i, var1 in enumerate(cols_categoricas):
            for var2 in cols_categoricas[i+1:]:
                if target and var2 == target:
                    resultado = analizar_independencia_categoricas(df, var1, var2)
                    resultados['independencia_categoricas'][f"{var1}_vs_{var2}"] = resultado
                    print(f"✅ Análisis {var1} vs {var2} completado")
                    
                    visualizar_categoricas(df, var1, var2)
    
    print(f"\n🎯 PIPELINE EDA COMPLETADO")
    print("=" * 60)
    
    return resultados

def resumen_ejecutivo_eda(resultados_eda):
    """
    Genera un resumen ejecutivo de los resultados del EDA.
    
    Parameters:
    -----------
    resultados_eda : dict
        Resultados del pipeline_eda_completo()
        
    Returns:
    --------
    str : Resumen ejecutivo en formato texto
    """
    resumen = []
    resumen.append("🎯 RESUMEN EJECUTIVO - ANÁLISIS EXPLORATORIO DE DATOS")
    resumen.append("=" * 65)
    
    # Información general
    info = resultados_eda['info_general']
    resumen.append(f"\n📊 CARACTERÍSTICAS DEL DATASET:")
    resumen.append(f"   • Dimensiones: {info['dimensiones'][0]} filas × {info['dimensiones'][1]} columnas")
    resumen.append(f"   • Variables numéricas: {info['n_numericas']}")
    resumen.append(f"   • Variables categóricas: {info['n_categoricas']}")
    resumen.append(f"   • Valores faltantes: {info['valores_faltantes']}")
    
    # Análisis cuantitativo
    if 'univariado_cuantitativo' in resultados_eda:
        cuanti = resultados_eda['univariado_cuantitativo']
        resumen.append(f"\n📈 VARIABLES CUANTITATIVAS:")
        
        # Variables con mayor asimetría
        var_asimetrica = cuanti.loc[cuanti['Asimetría'].abs().idxmax()]
        resumen.append(f"   • Mayor asimetría: {var_asimetrica['Variable']} (skew = {var_asimetrica['Asimetría']:.3f})")
        
        # Variables con outliers
        vars_outliers = cuanti[cuanti['Outliers_IQR'] > 0]['Variable'].tolist()
        resumen.append(f"   • Variables con outliers: {len(vars_outliers)} de {len(cuanti)}")
        
        # Normalidad
        vars_normales = cuanti[cuanti['Normal_DAgostino'] == 'Sí']['Variable'].tolist()
        resumen.append(f"   • Variables normales (D'Agostino): {len(vars_normales)} de {len(cuanti)}")
    
    # Correlaciones
    if 'correlaciones' in resultados_eda:
        corr = resultados_eda['correlaciones']
        corr_sig = corr[corr['Significativo_α05'] == 'Sí']
        corr_fuerte = corr_sig[corr_sig['Fuerza'].isin(['Fuerte', 'Muy fuerte'])]
        
        resumen.append(f"\n🔗 CORRELACIONES:")
        resumen.append(f"   • Correlaciones significativas: {len(corr_sig)} de {len(corr)}")
        resumen.append(f"   • Correlaciones fuertes: {len(corr_fuerte)}")
        
        if len(corr_fuerte) > 0:
            top_corr = corr_fuerte.iloc[0]
            resumen.append(f"   • Correlación más fuerte: {top_corr['Par_Variables']} (ρ = {top_corr.iloc[1]:.3f})")
    
    # Comparación de grupos
    if 'comparacion_grupos' in resultados_eda:
        grupos = resultados_eda['comparacion_grupos']
        resumen.append(f"\n⚖️ COMPARACIÓN DE GRUPOS:")
        
        diferencias_sig = 0
        for var, resultado in grupos.items():
            if 'test_t_significativo' in resultado and resultado['test_t_significativo']:
                diferencias_sig += 1
            elif 'anova_significativo' in resultado and resultado['anova_significativo']:
                diferencias_sig += 1
        
        resumen.append(f"   • Variables con diferencias significativas: {diferencias_sig} de {len(grupos)}")
    
    # Independencia categóricas
    if 'independencia_categoricas' in resultados_eda:
        independencia = resultados_eda['independencia_categoricas']
        asociaciones_sig = sum(1 for r in independencia.values() if r['significativo'])
        
        resumen.append(f"\n🔗 VARIABLES CATEGÓRICAS:")
        resumen.append(f"   • Asociaciones significativas: {asociaciones_sig} de {len(independencia)}")
        
        if asociaciones_sig > 0:
            # Buscar asociación más fuerte
            max_cramer = 0
            max_vars = ""
            for resultado in independencia.values():
                if resultado['significativo'] and resultado['cramers_v'] > max_cramer:
                    max_cramer = resultado['cramers_v']
                    max_vars = resultado['variables']
            
            if max_vars:
                resumen.append(f"   • Asociación más fuerte: {max_vars} (Cramér's V = {max_cramer:.3f})")
    
    resumen.append(f"\n" + "=" * 65)
    
    return "\n".join(resumen)

# ============================================================================
# 🔧 FUNCIONES AUXILIARES
# ============================================================================

def calcular_bins_freedman_diaconis(serie):
    """
    Calcula el número óptimo de bins usando la regla de Freedman-Diaconis.
    
    Parameters:
    -----------
    serie : pd.Series
        Serie de datos numéricos
        
    Returns:
    --------
    int : Número de bins recomendado
    """
    serie_clean = serie.dropna()
    if len(serie_clean) == 0:
        return 10
    
    q75, q25 = np.percentile(serie_clean, [75, 25])
    iqr = q75 - q25
    h = 2 * iqr * (len(serie_clean) ** (-1/3))
    
    if h <= 0:
        return 10
    
    num_bins = int(np.ceil((serie_clean.max() - serie_clean.min()) / h))
    return max(1, min(num_bins, 50))  # Limitar entre 1 y 50 bins

def crear_reporte_missing_values(df):
    """
    Crea un reporte detallado de valores faltantes.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame a analizar
        
    Returns:
    --------
    pd.DataFrame : Reporte de valores faltantes
    """
    missing_stats = []
    
    for col in df.columns:
        n_missing = df[col].isnull().sum()
        pct_missing = (n_missing / len(df)) * 100
        dtype = str(df[col].dtype)
        n_unique = df[col].nunique(dropna=False)
        
        missing_stats.append({
            'Columna': col,
            'Tipo': dtype,
            'N_Missing': n_missing,
            'Pct_Missing': pct_missing,
            'N_Unique': n_unique,
            'N_Valid': len(df) - n_missing
        })
    
    reporte = pd.DataFrame(missing_stats)
    reporte = reporte.sort_values('Pct_Missing', ascending=False)
    
    return reporte.round(2)

def detectar_outliers_multiple(df, columnas=None, metodos=['iqr', 'zscore', 'isolation']):
    """
    Detección de outliers con múltiples métodos.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos
    columnas : list, optional
        Lista de columnas numéricas a analizar
    metodos : list
        Métodos de detección ('iqr', 'zscore', 'isolation')
        
    Returns:
    --------
    dict : Resultados por método y columna
    """
    if columnas is None:
        columnas = df.select_dtypes(include=[np.number]).columns.tolist()
    
    resultados = {}
    
    for col in columnas:
        serie = df[col].dropna()
        if len(serie) == 0:
            continue
            
        resultados[col] = {}
        
        if 'iqr' in metodos:
            # Método IQR
            q1, q3 = serie.quantile([0.25, 0.75])
            iqr = q3 - q1
            limite_inf = q1 - 1.5 * iqr
            limite_sup = q3 + 1.5 * iqr
            outliers_iqr = ((serie < limite_inf) | (serie > limite_sup))
            
            resultados[col]['iqr'] = {
                'n_outliers': outliers_iqr.sum(),
                'pct_outliers': (outliers_iqr.sum() / len(serie)) * 100,
                'limite_inferior': limite_inf,
                'limite_superior': limite_sup,
                'outliers_mask': outliers_iqr
            }
        
        if 'zscore' in metodos:
            # Método Z-score
            z_scores = np.abs(stats.zscore(serie))
            outliers_z = z_scores > 3
            
            resultados[col]['zscore'] = {
                'n_outliers': outliers_z.sum(),
                'pct_outliers': (outliers_z.sum() / len(serie)) * 100,
                'threshold': 3,
                'outliers_mask': outliers_z
            }
        
        if 'isolation' in metodos:
            # Isolation Forest (requiere sklearn)
            try:
                from sklearn.ensemble import IsolationForest
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                outliers_iso = iso_forest.fit_predict(serie.values.reshape(-1, 1)) == -1
                
                resultados[col]['isolation'] = {
                    'n_outliers': outliers_iso.sum(),
                    'pct_outliers': (outliers_iso.sum() / len(serie)) * 100,
                    'contamination': 0.1,
                    'outliers_mask': pd.Series(outliers_iso, index=serie.index)
                }
            except ImportError:
                print("Warning: sklearn no disponible para Isolation Forest")
    
    return resultados

# ============================================================================
# 🎨 CONFIGURACIÓN DE ESTILO PARA GRÁFICOS
# ============================================================================

def configurar_estilo_graficos():
    """
    Configura el estilo por defecto para los gráficos.
    """
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Configuración de matplotlib
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 9
    plt.rcParams['grid.alpha'] = 0.3

# ============================================================================
# 📝 FUNCIÓN DE AYUDA Y DOCUMENTACIÓN
# ============================================================================

def mostrar_ayuda():
    """
    Muestra la documentación de uso de las funciones principales.
    """
    help_text = """
🔧 EDA UTILITIES - GUÍA DE USO
==============================

📈 ANÁLISIS UNIVARIADO:
- analisis_cuantitativo_completo(df, columnas=None)
- visualizar_cuantitativas(df, columnas=None) 
- analisis_cualitativo_completo(df, columnas=None)
- visualizar_cualitativas(df, columnas=None)

🔗 ANÁLISIS MULTIVARIADO:
- correlaciones_completas(df, columnas=None, metodos=['spearman'])
- resumen_correlaciones(df, columnas=None, metodo='spearman')
- visualizar_correlaciones(df, columnas=None, metodo='spearman')

⚖️ COMPARACIÓN DE GRUPOS:
- comparar_grupos_cuanti(df, var_continua, var_categorica)
- visualizar_grupos_cuanti(df, var_continua, var_categorica)
- analizar_independencia_categoricas(df, var1, var2)
- visualizar_categoricas(df, var1, var2)

🔄 DISCRETIZACIÓN:
- discretizar_variable(df, columna, metodos=['equal_width', 'quantiles', 'kmeans'])
- visualizar_discretizacion(df, columna, resultados)
- comparar_discretizacion_supervivencia(df, columna, target, resultados)

🎯 PIPELINE COMPLETO:
- pipeline_eda_completo(df, target=None)
- resumen_ejecutivo_eda(resultados_eda)

🔧 UTILIDADES:
- crear_reporte_missing_values(df)
- detectar_outliers_multiple(df, columnas=None)
- calcular_bins_freedman_diaconis(serie)
- configurar_estilo_graficos()

EJEMPLO DE USO:
===============
import utilities as eda

# Pipeline completo
resultados = eda.pipeline_eda_completo(df, target='Survived')

# Resumen ejecutivo
print(eda.resumen_ejecutivo_eda(resultados))

# Análisis específicos
resumen_corr = eda.resumen_correlaciones(df, metodo='spearman')
eda.visualizar_correlaciones(df, metodo='spearman')
"""
    print(help_text)

if __name__ == "__main__":
    print("🔧 EDA Utilities cargadas correctamente!")
    print("Usa mostrar_ayuda() para ver la documentación.")
    configurar_estilo_graficos()