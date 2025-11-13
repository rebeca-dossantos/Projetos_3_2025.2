import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Carregar os dados
df = pd.read_csv('database.csv')

# Pré-processamento, aglutina pré diabetes e diabetes como TRUE
df = df.rename(columns={'Diabetes_012': 'Diabetes_binary'})
df['Diabetes_binary'] = df['Diabetes_binary'].replace({2: 1})

# Definir variáveis
target = 'Diabetes_binary'
binary_col = [col for col in df.columns if df[col].nunique() == 2 and col != target]
num_col = [col for col in df.columns.difference(binary_col) if col != target]

# Inicializar app Dash
app = dash.Dash(__name__)
server = app.server

# Layout do app
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
            ], className="six columns", style={'padding': 20, 'backgroundColor': '#f8f9fa', 'borderRadius': 10}),
            
            html.Div([
                html.H3("Distribuição da Variável Alvo"),
                dcc.Graph(id='target-distribution')
            ], className="six columns", style={'padding': 20}),
        ], className="row"),
        
        html.Div([
            html.H3("Amostra dos Dados"),
            html.Div(
                dash.dash_table.DataTable(
                    data=df.head(10).to_dict('records'),
                    columns=[{'name': i, 'id': i} for i in df.columns],
                    page_size=10,
                    style_table={'overflowX': 'auto'},
                    style_cell={
                        'textAlign': 'left',
                        'padding': '5px',
                        'fontSize': '12px'
                    },
                    style_header={
                        'backgroundColor': '#3498db',
                        'color': 'white',
                        'fontWeight': 'bold'
                    }
                )
            )
        ], style={'marginTop': 20})
    ], style={'padding': 20, 'backgroundColor': 'white', 'borderRadius': 10, 'marginBottom': 20}),
    
    # Seção 2: Análise de Variáveis Binárias (AGORA COM HISTOGRAMA)
    html.Div([
        html.H2("2. Análise de Variáveis Binárias", style={'color': '#3498db', 'borderBottom': '2px solid #3498db', 'paddingBottom': 10}),
        
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
    
    # Seção 3: Análise de Variáveis Numéricas
    html.Div([
        html.H2("3. Análise de Variáveis Numéricas", style={'color': '#3498db', 'borderBottom': '2px solid #3498db', 'paddingBottom': 10}),
        
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
    
    # Seção 4: Análise de Correlação
    html.Div([
        html.H2("4. Análise de Correlação", style={'color': '#3498db', 'borderBottom': '2px solid #3498db', 'paddingBottom': 10}),
        
        html.Div([
            html.Div([
                dcc.Graph(id='correlation-heatmap')
            ], className="six columns"),
            
            html.Div([
                dcc.Graph(id='correlation-with-target')
            ], className="six columns"),
        ], className="row")
    ], style={'padding': 20, 'backgroundColor': 'white', 'borderRadius': 10, 'marginBottom': 20}),
    
    # Seção 5: Análise de Outliers
    html.Div([
        html.H2("5. Análise de Outliers", style={'color': '#3498db', 'borderBottom': '2px solid #3498db', 'paddingBottom': 10}),
        
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
    
    # Seção 6: Modelo Preditivo
    html.Div([
        html.H2("6. Modelo Preditivo", style={'color': '#3498db', 'borderBottom': '2px solid #3498db', 'paddingBottom': 10}),
        
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

# Callbacks

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

# 2. HISTOGRAMA para variáveis binárias (SUBSTITUINDO O GRÁFICO DE PIZZA)
@app.callback(
    Output('binary-histogram', 'figure'),
    Input('binary-var-dropdown', 'value')
)
def update_binary_histogram(selected_var):
    if selected_var is None:
        return go.Figure()
    
    # Agrupar dados por diabetes e pela variável selecionada
    grouped = df.groupby([target, selected_var]).size().unstack(fill_value=0)
    
    # Criar dados para o histograma agrupado
    categories = ['Não', 'Sim']  # Para a variável binária selecionada
    
    fig = go.Figure()
    
    # Barra para não diabéticos
    fig.add_trace(go.Bar(
        name='Sem Diabetes',
        x=categories,
        y=[grouped.iloc[0].get(0, 0), grouped.iloc[0].get(1, 0)],
        marker_color='#3498db',
        text=[grouped.iloc[0].get(0, 0), grouped.iloc[0].get(1, 0)],
        textposition='auto'
    ))
    
    # Barra para diabéticos
    fig.add_trace(go.Bar(
        name='Com Diabetes',
        x=categories,
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
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
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
    
    # Separar dados por status de diabetes
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
    # Calcular matriz de correlação
    corr = df.corr()
    
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
        height=600
    )
    
    return fig

# 5. Correlação com a variável alvo
@app.callback(
    Output('correlation-with-target', 'figure'),
    Input('correlation-with-target', 'id')
)
def update_correlation_with_target(_):
    # Calcular correlação com a variável alvo
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
        height=600
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
    
    # Identificar outliers usando IQR
    Q1 = df[selected_var].quantile(0.25)
    Q3 = df[selected_var].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[selected_var] < lower_bound) | (df[selected_var] > upper_bound)]
    outlier_count = len(outliers)
    outlier_percentage = (outlier_count / len(df)) * 100
    
    # Criar gráfico de distribuição com outliers destacados
    fig = go.Figure()
    
    # Dados normais
    normal_data = df[(df[selected_var] >= lower_bound) & (df[selected_var] <= upper_bound)][selected_var]
    fig.add_trace(go.Histogram(
        x=normal_data,
        name='Dados Normais',
        marker_color='#3498db',
        opacity=0.7
    ))
    
    # Outliers
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
    
    # Estatísticas de outliers
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
    # Preparar dados
    X = df.drop(target, axis=1)
    y = df[target]
    
    # Dividir em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=25)
    
    # Treinar modelo Gradient Boosting
    model = GradientBoostingClassifier(random_state=52)
    model.fit(X_train, y_train)
    
    # Fazer previsões
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calcular métricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Criar matriz de confusão
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    
    # Normalizar matriz de confusão
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
    
    # Métricas do modelo
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