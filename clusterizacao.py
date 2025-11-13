import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# Carregar os dados
df = pd.read_csv('database.csv')

# PRÉ-PROCESSAMENTO COMPLETO
# 1. Transformação da variável alvo
df = df.rename(columns={'Diabetes_012': 'Diabetes_binary'})
df['Diabetes_binary'] = df['Diabetes_binary'].replace({2: 1})

# 2. Tratamento de outliers para BMI
def winsorize_iqr(df, col, k=1.5):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - k * IQR
    upper_bound = Q3 + k * IQR
    return df[col].clip(lower=lower_bound, upper=upper_bound)

df['BMI'] = winsorize_iqr(df, 'BMI')

# 3. Classificação das variáveis
target = 'Diabetes_binary'
binary_col = [col for col in df.columns if df[col].nunique() == 2 and col != target]
num_col = [col for col in df.columns.difference(binary_col) if col != target]

# 4. Preparação para clusterização
# Selecionar features mais relevantes para clusterização
cluster_features = ['BMI', 'Age', 'GenHlth', 'PhysHlth', 'MentHlth', 'HighBP', 'HighChol']
X_cluster = df[cluster_features]

# Padronizar os dados para clusterização
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# Aplicar PCA para visualização
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Encontrar número ótimo de clusters
wcss = []
silhouette_scores = []
k_range = range(2, 8)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# Clusterização final com K ótimo
optimal_k = 4  # Baseado na análise do cotovelo e silhouette
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['Cluster'] = kmeans_final.fit_predict(X_scaled)

# Adicionar coordenadas PCA ao dataframe
df['PCA1'] = X_pca[:, 0]
df['PCA2'] = X_pca[:, 1]

# Inicializar app Dash
app = dash.Dash(__name__)
server = app.server

# Layout do app expandido
app.layout = html.Div([
    html.H1("Análise Exploratória de Dados - Indicadores de Saúde para Diabetes", 
            style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 30}),
    
    # Seção 1: Visão Geral dos Dados
    html.Div([
        html.H2("1. Visão Geral dos Dados", style={'color': '#3498db', 'borderBottom': '2px solid #3498db', 'paddingBottom': 10}),
        
        html.Div([
            html.Div([
                html.H3("Estatísticas Descritivas"),
                html.P(f"Total de registros: {df.shape[0]:,}"),
                html.P(f"Total de variáveis: {df.shape[1]}"),
                html.P(f"Variáveis binárias: {len(binary_col)}"),
                html.P(f"Variáveis numéricas: {len(num_col)}"),
                html.P(f"Clusters identificados: {optimal_k}"),
            ], className="six columns", style={'padding': 20, 'backgroundColor': '#f8f9fa', 'borderRadius': 10}),
            
            html.Div([
                html.H3("Distribuição da Variável Alvo"),
                dcc.Graph(id='target-distribution')
            ], className="six columns", style={'padding': 20}),
        ], className="row"),
    ], style={'padding': 20, 'backgroundColor': 'white', 'borderRadius': 10, 'marginBottom': 20}),
    
    # Seção 2: Análise de Clusterização
    html.Div([
        html.H2("2. Análise de Clusterização", style={'color': '#3498db', 'borderBottom': '2px solid #3498db', 'paddingBottom': 10}),
        
        html.Div([
            html.Div([
                html.H4("Visualização dos Clusters (PCA)"),
                dcc.Graph(id='cluster-plot')
            ], className="six columns"),
            
            html.Div([
                html.H4("Análise do Número Ótimo de Clusters"),
                dcc.Graph(id='elbow-plot')
            ], className="six columns"),
        ], className="row"),
        
        html.Div([
            html.Div([
                html.H4("Características dos Clusters"),
                dcc.Dropdown(
                    id='cluster-feature-dropdown',
                    options=[{'label': col, 'value': col} for col in cluster_features],
                    value='BMI'
                ),
                dcc.Graph(id='cluster-feature-plot')
            ], className="six columns"),
            
            html.Div([
                html.H4("Composição dos Clusters por Diabetes"),
                dcc.Graph(id='cluster-diabetes-plot')
            ], className="six columns"),
        ], className="row"),
        
        html.Div([
            html.H4("Perfil dos Clusters - Estatísticas Médias"),
            html.Div(id='cluster-stats-table')
        ], style={'marginTop': 20})
    ], style={'padding': 20, 'backgroundColor': 'white', 'borderRadius': 10, 'marginBottom': 20}),
    
    # Seção 3: Análise de Variáveis Binárias
    html.Div([
        html.H2("3. Análise de Variáveis Binárias", style={'color': '#3498db', 'borderBottom': '2px solid #3498db', 'paddingBottom': 10}),
        
        html.Div([
            html.Label("Selecione uma variável binária:"),
            dcc.Dropdown(
                id='binary-var-dropdown',
                options=[{'label': col, 'value': col} for col in binary_col],
                value=binary_col[0] if binary_col else None
            )
        ], style={'marginBottom': 20}),
        
        dcc.Graph(id='binary-histogram')
    ], style={'padding': 20, 'backgroundColor': 'white', 'borderRadius': 10, 'marginBottom': 20}),
    
    # Seção 4: Análise de Variáveis Numéricas
    html.Div([
        html.H2("4. Análise de Variáveis Numéricas", style={'color': '#3498db', 'borderBottom': '2px solid #3498db', 'paddingBottom': 10}),
        
        html.Div([
            html.Label("Selecione uma variável numérica:"),
            dcc.Dropdown(
                id='numeric-var-dropdown',
                options=[{'label': col, 'value': col} for col in num_col],
                value=num_col[0] if num_col else None
            )
        ], style={'marginBottom': 20}),
        
        dcc.Graph(id='numeric-histogram')
    ], style={'padding': 20, 'backgroundColor': 'white', 'borderRadius': 10, 'marginBottom': 20}),
    
    # Seção 5: Análise de Correlação
    html.Div([
        html.H2("5. Análise de Correlação", style={'color': '#3498db', 'borderBottom': '2px solid #3498db', 'paddingBottom': 10}),
        
        html.Div([
            html.Div([
                dcc.Graph(id='correlation-heatmap')
            ], className="six columns"),
            
            html.Div([
                dcc.Graph(id='correlation-with-target')
            ], className="six columns"),
        ], className="row")
    ], style={'padding': 20, 'backgroundColor': 'white', 'borderRadius': 10, 'marginBottom': 20}),
    
    # Seção 6: Análise de Outliers
    html.Div([
        html.H2("6. Análise de Outliers", style={'color': '#3498db', 'borderBottom': '2px solid #3498db', 'paddingBottom': 10}),
        
        html.Div([
            html.Label("Selecione uma variável para análise de outliers:"),
            dcc.Dropdown(
                id='outlier-var-dropdown',
                options=[{'label': col, 'value': col} for col in num_col],
                value='BMI' if 'BMI' in num_col else (num_col[0] if num_col else None)
            )
        ], style={'marginBottom': 20}),
        
        html.Div([
            html.Div([
                dcc.Graph(id='outlier-boxplot')
            ], className="six columns"),
            
            html.Div([
                dcc.Graph(id='outlier-analysis')
            ], className="six columns"),
        ], className="row"),
        
        html.Div(id='outlier-stats', style={'marginTop': 20})
    ], style={'padding': 20, 'backgroundColor': 'white', 'borderRadius': 10, 'marginBottom': 20}),
    
    # Seção 7: Modelo Preditivo
    html.Div([
        html.H2("7. Modelo Preditivo", style={'color': '#3498db', 'borderBottom': '2px solid #3498db', 'paddingBottom': 10}),
        
        html.Div([
            html.Div([
                html.H4("Desempenho do Modelo Gradient Boosting"),
                html.Div(id='model-metrics')
            ], className="six columns"),
            
            html.Div([
                dcc.Graph(id='confusion-matrix')
            ], className="six columns"),
        ], className="row")
    ], style={'padding': 20, 'backgroundColor': 'white', 'borderRadius': 10, 'marginBottom': 20}),
    
], style={'padding': 20, 'backgroundColor': '#ecf0f1', 'fontFamily': 'Arial, sans-serif'})

# Callbacks para Clusterização

@app.callback(
    Output('cluster-plot', 'figure'),
    Input('cluster-plot', 'id')
)
def update_cluster_plot(_):
    fig = px.scatter(
        df, x='PCA1', y='PCA2', color='Cluster',
        title='Visualização dos Clusters (PCA)',
        labels={'PCA1': f'Componente Principal 1 ({pca.explained_variance_ratio_[0]:.1%})',
                'PCA2': f'Componente Principal 2 ({pca.explained_variance_ratio_[1]:.1%})'},
        color_continuous_scale='viridis'
    )
    fig.update_layout(template='plotly_white')
    return fig

@app.callback(
    Output('elbow-plot', 'figure'),
    Input('elbow-plot', 'id')
)
def update_elbow_plot(_):
    fig = go.Figure()
    
    # Plot Elbow Method
    fig.add_trace(go.Scatter(
        x=list(k_range),
        y=wcss,
        mode='lines+markers',
        name='WCSS',
        line=dict(color='#3498db')
    ))
    
    # Plot Silhouette Scores
    fig.add_trace(go.Scatter(
        x=list(k_range),
        y=silhouette_scores,
        mode='lines+markers',
        name='Silhouette Score',
        yaxis='y2',
        line=dict(color='#e74c3c')
    ))
    
    fig.update_layout(
        title='Método do Cotovelo e Silhouette Score',
        xaxis_title='Número de Clusters (K)',
        yaxis_title='WCSS',
        yaxis2=dict(
            title='Silhouette Score',
            overlaying='y',
            side='right'
        ),
        template='plotly_white'
    )
    
    return fig

@app.callback(
    Output('cluster-feature-plot', 'figure'),
    Input('cluster-feature-dropdown', 'value')
)
def update_cluster_feature_plot(selected_feature):
    fig = px.box(
        df, x='Cluster', y=selected_feature,
        title=f'Distribuição de {selected_feature} por Cluster',
        color='Cluster',
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig.update_layout(template='plotly_white')
    return fig

@app.callback(
    Output('cluster-diabetes-plot', 'figure'),
    Input('cluster-diabetes-plot', 'id')
)
def update_cluster_diabetes_plot(_):
    cluster_diabetes = pd.crosstab(df['Cluster'], df['Diabetes_binary'], normalize='index') * 100
    cluster_diabetes.columns = ['Não Diabético', 'Diabético']
    
    fig = px.bar(
        cluster_diabetes, 
        x=cluster_diabetes.index,
        y=['Não Diabético', 'Diabético'],
        title='Distribuição de Diabetes por Cluster (%)',
        barmode='stack',
        color_discrete_sequence=['#3498db', '#e74c3c']
    )
    
    fig.update_layout(
        xaxis_title='Cluster',
        yaxis_title='Proporção (%)',
        template='plotly_white'
    )
    
    return fig

@app.callback(
    Output('cluster-stats-table', 'children'),
    Input('cluster-stats-table', 'id')
)
def update_cluster_stats_table(_):
    # Calcular estatísticas médias por cluster
    stats_cols = cluster_features + ['Diabetes_binary']
    cluster_stats = df[stats_cols + ['Cluster']].groupby('Cluster').mean().round(2)
    
    # Criar tabela
    table_header = [html.Tr([html.Th('Cluster')] + [html.Th(col) for col in cluster_stats.columns])]
    
    table_rows = []
    for cluster in cluster_stats.index:
        row = [html.Td(f'Cluster {cluster}')]
        for col in cluster_stats.columns:
            row.append(html.Td(cluster_stats.loc[cluster, col]))
        table_rows.append(html.Tr(row))
    
    table = html.Table(
        table_header + table_rows,
        style={'width': '100%', 'borderCollapse': 'collapse', 'marginTop': '10px'}
    )
    
    return [
        html.P("Esta tabela mostra as características médias de cada cluster:"),
        table
    ]

# Callbacks existentes (mantidos da versão anterior)...

# 1. Distribuição da variável alvo
@app.callback(
    Output('target-distribution', 'figure'),
    Input('target-distribution', 'id')
)
def update_target_distribution(_):
    counts = df[target].value_counts().sort_index()
    percentages = df[target].value_counts(normalize=True).sort_index().values * 100
    
    fig = go.Figure(data=[
        go.Bar(
            x=['Não Diabético', 'Diabético'],
            y=counts.values,
            text=[f'{p:.1f}%' for p in percentages],
            textposition='auto',
            marker_color=['#3498db', '#e74c3c']
        )
    ])
    
    fig.update_layout(
        title='Distribuição de Diabetes na População',
        xaxis_title='Status de Diabetes',
        yaxis_title='Número de Indivíduos',
        template='plotly_white'
    )
    
    return fig

# 2. Histograma para variáveis binárias
@app.callback(
    Output('binary-histogram', 'figure'),
    Input('binary-var-dropdown', 'value')
)
def update_binary_histogram(selected_var):
    if selected_var is None:
        return go.Figure()
    
    grouped = df.groupby([target, selected_var]).size().unstack(fill_value=0)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Sem Diabetes',
        x=['Não', 'Sim'],
        y=[grouped.iloc[0].get(0, 0), grouped.iloc[0].get(1, 0)],
        marker_color='#3498db',
        text=[grouped.iloc[0].get(0, 0), grouped.iloc[0].get(1, 0)],
        textposition='auto'
    ))
    
    fig.add_trace(go.Bar(
        name='Com Diabetes',
        x=['Não', 'Sim'],
        y=[grouped.iloc[1].get(0, 0), grouped.iloc[1].get(1, 0)],
        marker_color='#e74c3c',
        text=[grouped.iloc[1].get(0, 0), grouped.iloc[1].get(1, 0)],
        textposition='auto'
    ))
    
    fig.update_layout(
        title=f'Distribuição de {selected_var} por Status de Diabetes',
        xaxis_title=selected_var,
        yaxis_title='Número de Indivíduos',
        barmode='group',
        template='plotly_white'
    )
    
    return fig

# 3. Histograma para variáveis numéricas
@app.callback(
    Output('numeric-histogram', 'figure'),
    Input('numeric-var-dropdown', 'value')
)
def update_numeric_histogram(selected_var):
    if selected_var is None:
        return go.Figure()
    
    no_diabetes = df[df[target] == 0][selected_var]
    diabetes = df[df[target] == 1][selected_var]
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=no_diabetes,
        name='Sem Diabetes',
        opacity=0.7,
        nbinsx=min(30, df[selected_var].nunique()),
        marker_color='#3498db'
    ))
    
    fig.add_trace(go.Histogram(
        x=diabetes,
        name='Com Diabetes',
        opacity=0.7,
        nbinsx=min(30, df[selected_var].nunique()),
        marker_color='#e74c3c'
    ))
    
    fig.update_layout(
        title=f'Distribuição de {selected_var} por Status de Diabetes',
        xaxis_title=selected_var,
        yaxis_title='Frequência',
        barmode='overlay',
        template='plotly_white'
    )
    
    return fig

# 4. Heatmap de correlação
@app.callback(
    Output('correlation-heatmap', 'figure'),
    Input('correlation-heatmap', 'id')
)
def update_correlation_heatmap(_):
    corr = df[cluster_features + [target]].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        colorscale='RdBu',
        zmin=-1,
        zmax=1,
        hoverongaps=False
    ))
    
    fig.update_layout(
        title='Matriz de Correlação entre Variáveis',
        xaxis_title='Variáveis',
        yaxis_title='Variáveis',
        height=500
    )
    
    return fig

# 5. Correlação com a variável alvo
@app.callback(
    Output('correlation-with-target', 'figure'),
    Input('correlation-with-target', 'id')
)
def update_correlation_with_target(_):
    corr = df.corr()
    corr_target = corr[target].drop(target).sort_values(ascending=True)
    
    fig = go.Figure(data=[
        go.Bar(
            y=corr_target.index,
            x=corr_target.values,
            orientation='h',
            marker_color=['#e74c3c' if x > 0 else '#3498db' for x in corr_target.values]
        )
    ])
    
    fig.update_layout(
        title='Correlação com Diabetes',
        xaxis_title='Coeficiente de Correlação',
        yaxis_title='Variáveis',
        height=500
    )
    
    return fig

# 6. Boxplot para análise de outliers
@app.callback(
    Output('outlier-boxplot', 'figure'),
    Input('outlier-var-dropdown', 'value')
)
def update_outlier_boxplot(selected_var):
    if selected_var is None:
        return go.Figure()
    
    fig = go.Figure()
    
    fig.add_trace(go.Box(
        y=df[selected_var],
        name=selected_var,
        marker_color='#3498db',
        boxpoints='outliers'
    ))
    
    fig.update_layout(
        title=f'Boxplot de {selected_var}',
        yaxis_title=selected_var,
        template='plotly_white'
    )
    
    return fig

# 7. Análise de outliers
@app.callback(
    [Output('outlier-analysis', 'figure'),
     Output('outlier-stats', 'children')],
    [Input('outlier-var-dropdown', 'value')]
)
def update_outlier_analysis(selected_var):
    if selected_var is None:
        return go.Figure(), ""
    
    Q1 = df[selected_var].quantile(0.25)
    Q3 = df[selected_var].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[selected_var] < lower_bound) | (df[selected_var] > upper_bound)]
    outlier_count = len(outliers)
    outlier_percentage = (outlier_count / len(df)) * 100
    
    fig = go.Figure()
    
    normal_data = df[(df[selected_var] >= lower_bound) & (df[selected_var] <= upper_bound)][selected_var]
    fig.add_trace(go.Histogram(
        x=normal_data,
        name='Dados Normais',
        marker_color='#3498db',
        opacity=0.7
    ))
    
    if outlier_count > 0:
        fig.add_trace(go.Histogram(
            x=outliers[selected_var],
            name='Outliers',
            marker_color='#e74c3c',
            opacity=0.7
        ))
    
    fig.update_layout(
        title=f'Distribuição de {selected_var} com Outliers Destacados',
        xaxis_title=selected_var,
        yaxis_title='Frequência',
        barmode='overlay',
        template='plotly_white'
    )
    
    stats_text = [
        html.H4("Estatísticas de Outliers"),
        html.P(f"Limite inferior: {lower_bound:.2f}"),
        html.P(f"Limite superior: {upper_bound:.2f}"),
        html.P(f"Número de outliers: {outlier_count} ({outlier_percentage:.2f}% dos dados)"),
        html.P(f"Valor mínimo: {df[selected_var].min():.2f}"),
        html.P(f"Valor máximo: {df[selected_var].max():.2f}"),
        html.P(f"Mediana: {df[selected_var].median():.2f}")
    ]
    
    return fig, stats_text

# 8. Métricas do modelo e matriz de confusão
@app.callback(
    [Output('model-metrics', 'children'),
     Output('confusion-matrix', 'figure')],
    Input('model-metrics', 'id')
)
def update_model_metrics(_):
    X = df.drop(target, axis=1)
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=25)
    
    model = GradientBoostingClassifier(random_state=52)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig = px.imshow(
        cm_normalized,
        text_auto=True,
        color_continuous_scale='Blues',
        labels=dict(x="Previsto", y="Real", color="Proporção"),
        x=['Não Diabético', 'Diabético'],
        y=['Não Diabético', 'Diabético']
    )
    
    fig.update_layout(
        title='Matriz de Confusão (Normalizada)',
        xaxis_title='Rótulo Previsto',
        yaxis_title='Rótulo Real'
    )
    
    metrics = [
        html.H4("Métricas do Modelo Gradient Boosting"),
        html.P(f"Acurácia: {accuracy:.2%}"),
        html.P(f"Precisão: {precision:.2%}"),
        html.P(f"Recall: {recall:.2%}"),
        html.P(f"F1-Score: {f1:.2%}"),
        html.P(f"ROC AUC: {roc_auc:.2%}"),
        html.Hr(),
        html.P("O modelo foi treinado com 90% dos dados e testado com 10%."),
        html.P("A matriz de confusão mostra a proporção de classificações corretas e incorretas.")
    ]
    
    return metrics, fig

if __name__ == '__main__':
    app.run(debug=True)