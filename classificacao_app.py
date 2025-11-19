import dash
from dash import dcc, html, Input, Output, State, ALL, dash_table
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import joblib
import shap
import warnings
import io
import base64
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

warnings.filterwarnings('ignore')

# --- 1. CARREGAR MODELO E DADOS ---
print(">>> Carregando modelo 'modelo_classificacao.joblib'...")
pipeline = joblib.load('assets/modelo_classificacao.joblib')
model = pipeline.named_steps['model']
scaler = pipeline.named_steps['scaler']
print(">>> Modelo carregado.")

# --- CARREGAR MÉTRICAS DE COMPARAÇÃO ---
print(">>> Carregando métricas de 'model_comparison_metrics.csv'...")
try:
    df_metrics = pd.read_csv('assets/model_comparison_metrics.csv')
    model_options = [{'label': model, 'value': model} for model in df_metrics['Modelo'].unique()]
except Exception as e:
    print(f"Aviso: Não foi possível carregar 'model_comparison_metrics.csv'. {e}")
    df_metrics = pd.DataFrame(columns=['Modelo', 'Acurácia', 'TN', 'FP', 'FN', 'TP'])
    model_options = []


# --- 2. PREPARAR DADOS PARA O SHAP GLOBAL ---
print(">>> Preparando dados para o SHAP Global...")
df = pd.read_csv('database.csv')
df = df.rename(columns={'Diabetes_012': 'Diabetes_binary'})
df['Diabetes_binary'] = df['Diabetes_binary'].replace({2: 1})
def winsorize_iqr(df, col, k=1.5):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - k * IQR
    upper_bound = Q3 + k * IQR
    return df[col].clip(lower=lower_bound, upper=upper_bound)
df['BMI'] = winsorize_iqr(df, 'BMI')
features = [col for col in df.columns if col not in ['Diabetes_binary', 'Cluster', 'PCA1', 'PCA2']]
target = 'Diabetes_binary'
X = df[features]
y = df[target]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_test_scaled = scaler.transform(X_test)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=features)

# --- NOVO: Listas de colunas e Dicionário de Tradução ---
# Separação correta dos tipos de input
float_cols = ['BMI']
integer_cols = ['GenHlth', 'MentHlth', 'PhysHlth', 'Age', 'Education', 'Income']
binary_cols = [col for col in features if col not in float_cols and col not in integer_cols]

# Dicionário para traduzir os nomes
feature_name_map = {
    'BMI': 'IMC (ex: 28.5)',
    'GenHlth': 'Saúde Geral (1=Excelente, 5=Ruim)',
    'MentHlth': 'Saúde Mental (0-30 dias)',
    'PhysHlth': 'Saúde Física (0-30 dias)',
    'Age': 'Idade (Categoria 1-13)',
    'Education': 'Educação (Categoria 1-6)',
    'Income': 'Renda (Categoria 1-8)',
    'HighBP': 'Pressão Alta',
    'HighChol': 'Colesterol Alto',
    'CholCheck': 'Verificou Colesterol (últ. 5 anos)',
    'Smoker': 'Fumante (mais de 100 cigarros)',
    'Stroke': 'Derrame (AVC)',
    'HeartDiseaseorAttack': 'Doença Cardíaca ou Ataque',
    'PhysActivity': 'Atividade Física (últ. 30 dias)',
    'Fruits': 'Consome Frutas (1+ por dia)',
    'Veggies': 'Consome Vegetais (1+ por dia)',
    'HvyAlcoholConsump': 'Consumo Alto de Álcool',
    'AnyHealthcare': 'Possui Plano de Saúde',
    'NoDocbcCost': 'Evitou Médico por Custo (últ. 12m)',
    'DiffWalk': 'Dificuldade de Andar',
    'Sex': 'Sexo (0=F, 1=M)',
}
# --- FIM DA NOVIDADE ---


print(">>> Calculando SHAP values globais...")
explainer = shap.TreeExplainer(model)
shap_values_global_raw = explainer.shap_values(X_test_scaled_df)
if isinstance(shap_values_global_raw, list) and len(shap_values_global_raw) == 2:
    shap_values_global_class1 = shap_values_global_raw[1]
else:
    shap_values_global_class1 = shap_values_global_raw
print(">>> SHAP Global calculado.")


# --- 3. FUNÇÃO DE LAYOUT (MODIFICADA) ---
def create_input_fields(features_list):
    campos = []
    
    binary_options = [
        {'label': 'Não', 'value': 0},
        {'label': 'Sim', 'value': 1}
    ]
    
    # --- NOVO: Propriedades dos campos de Inteiro ---
    int_field_props = {
        'GenHlth': {'min': 1, 'max': 5, 'step': 1},
        'MentHlth': {'min': 0, 'max': 30, 'step': 1},
        'PhysHlth': {'min': 0, 'max': 30, 'step': 1},
        'Age': {'min': 1, 'max': 13, 'step': 1},
        'Education': {'min': 1, 'max': 6, 'step': 1},
        'Income': {'min': 1, 'max': 8, 'step': 1},
    }
    
    for col in features_list:
        default_value = X_test[col].mean()
        label_traduzida = feature_name_map.get(col, col)
        input_component = None
        
        if col in float_cols:
            # --- TIPO 1: FLOAT (só o IMC) ---
            input_component = dcc.Input(
                id={'type': 'input-local', 'index': col},
                type='number',
                value=round(default_value, 2), # Aceita decimal
                style={'width': '100px'}
            )
            
        elif col in integer_cols:
            # --- TIPO 2: INTEIRO (Idade, Renda, etc.) ---
            props = int_field_props[col] # Pega min/max/step
            input_component = dcc.Input(
                id={'type': 'input-local', 'index': col},
                type='number',
                value=int(round(default_value)), # Valor inicial é inteiro
                min=props['min'],
                max=props['max'],
                step=props['step'], # step=1 força a ser inteiro
                style={'width': '100px'}
            )
            
        else:
            # --- TIPO 3: BINÁRIO (Sim/Não) ---
            default_binary_value = int(round(default_value)) 
            input_component = dcc.Dropdown(
                id={'type': 'input-local', 'index': col},
                options=binary_options,
                value=default_binary_value,
                clearable=False,
                style={'width': '100px', 'display': 'inline-block', 'verticalAlign': 'middle'}
            )
        
        campos.append(
            html.Div([
                html.Label(f"{label_traduzida}:", style={'marginRight': '10px', 'width': '250px', 'display': 'inline-block'}),
                input_component
            ], style={'display': 'inline-block', 'width': '380px', 'padding': '5px', 'border': '1px solid #ccc', 'borderRadius': '5px', 'margin': '5px'})
        )
        
    return html.Div(campos)
# --- FIM DA FUNÇÃO MODIFICADA ---


# --- 4. INICIALIZAR O APP DASH ---
app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
server = app.server

# --- 5. LAYOUT DO APP ---
app.layout = html.Div([
    html.H1("Classificação", style={'textAlign': 'center'}),
    
    dcc.Tabs(id="tabs-main", value='tab-1', children=[
        
        # --- ABA 1: PREDIÇÃO LOCAL ---
        dcc.Tab(label='Predição e Explicação Local', value='tab-1', children=[
            html.Div([
                html.H3("Preencha os dados do paciente:", style={'marginTop': '20px'}),
                html.P("Os valores estão pré-preenchidos com a média do dataset de teste."),
                create_input_fields(features), # Chama a nova função
                html.Button('Fazer Previsão', id='btn-prever', n_clicks=0, 
                            style={'marginTop': '20px', 'fontSize': '16px', 'padding': '10px'}),
                html.Hr(),
                html.H2("Resultado da Predição:", style={'color': '#007BFF'}),
                html.Div(id='output-previsao', style={'fontSize': '24px', 'fontWeight': 'bold'}),
                html.Hr(),
                html.H2("Explicação Local (Gráfico de Força SHAP)", style={'color': 'black'}),
                html.P("Este gráfico mostra quais fatores estão 'empurrando' a previsão para 'Risco de Diabetes' (vermelho) ou 'Não Diabético' (azul)."),
                html.Iframe(id='shap-local-plot', 
                            style={'width': '100%', 'height': '200px', 'border': '1px solid #ddd', 'borderRadius': '5px'})
            ], style={'padding': '20px'})
        ]),
        
        # --- ABA 2: EXPLICAÇÕES GLOBAIS ---
        dcc.Tab(label='Explicações Globais', value='tab-2', children=[
            html.Div([
                html.H3("Importância Geral das Features (SHAP - Gráfico de Barras)", style={'marginTop': '20px'}),
                html.P("Gráfico de barras mostrando o impacto médio de cada feature nas previsões (para a classe 'Risco de Diabetes')."),
                dcc.Graph(id='shap-global-bar-plot'),
                html.Hr(style={'margin': '40px 0'}),
                html.H3("Distribuição do Impacto (SHAP - Beeswarm Plot)", style={'marginTop': '20px'}),
                html.P("Mostra o impacto de cada feature para cada ponto do dataset de teste."),
                html.P("Pontos vermelhos = Valor alto da feature (ex: Idade alta). Pontos azuis = Valor baixo."),
                html.P("Eixo X = Impacto no modelo (positivo aumenta o risco, negativo diminui)."),
                html.Img(id='shap-global-beeswarm-plot', style={'width': '60%', 'maxWidth': '700px', 'margin': 'auto', 'display': 'block'})
            ], style={'padding': '20px'})
        ]),
        
        # --- ABA 3: COMPARAÇÃO DE MODELOS ---
        dcc.Tab(label='Comparação de Modelos', value='tab-3', children=[
            html.Div([
                html.H3("Matriz de Confusão por Modelo", style={'marginTop': '20px'}),
                html.P("Veja quantas vezes cada modelo acertou (diagonal principal) e errou (fora da diagonal)."),
                html.Div([
                    html.Label("Selecione um Modelo:", style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='model-cm-dropdown',
                        options=model_options,
                        value=model_options[0]['value'] if model_options else None,
                        style={'marginTop': '10px'}
                    ),
                    dcc.Graph(id='confusion-matrix-graph')
                ], style={'width': '80%', 'margin': '30px auto', 'padding': '20px', 'border': '1px solid #ddd', 'borderRadius': '5px'}),
                html.Hr(),
                html.H4("Tabela de Métricas (Completa)"),
                dash_table.DataTable(
                    id='metrics-table',
                    columns=[{"name": i, "id": i} for i in df_metrics.columns],
                    data=df_metrics.to_dict('records'),
                    style_cell={'textAlign': 'left', 'padding': '10px', 'fontFamily': 'Arial'},
                    style_header={
                        'backgroundColor': '#007BFF',
                        'color': 'white',
                        'fontWeight': 'bold',
                        'textAlign': 'center'
                    },
                    style_data_conditional=[
                        {'if': {'row_index': 'odd'}, 'backgroundColor': 'rgb(248, 248, 248)'}
                    ],
                    sort_action="native",
                )
            ], style={'padding': '20px'})
        ]),
        
    ])
], style={'width': '90%', 'margin': 'auto'})


# --- 6. CALLBACKS ---

# Callback para a PREDIÇÃO LOCAL e SHAP LOCAL
# (NÃO PRECISA MUDAR)
@app.callback(
    [Output('output-previsao', 'children'),
     Output('shap-local-plot', 'srcDoc')],
    Input('btn-prever', 'n_clicks'),
    State({'type': 'input-local', 'index': ALL}, 'value'),
    State({'type': 'input-local', 'index': ALL}, 'id'),
    prevent_initial_call=True
)
def update_local_prediction(n_clicks, values, ids):
    if n_clicks == 0:
        return "", ""
        
    # Checar se algum valor é nulo (pode acontecer se o usuário apagar um campo)
    if any(v is None for v in values):
        return html.Span("Erro: Todos os campos devem ser preenchidos.", style={'color': 'red'}), ""
        
    input_data = {id['index']: val for id, val in zip(ids, values)}
    input_df = pd.DataFrame([input_data], columns=features)
            
    input_scaled = scaler.transform(input_df)
    
    prob_diabetes = pipeline.predict_proba(input_scaled)[0][1]
    
    if prob_diabetes > 0.5:
        prob_nao_diabetico = (1 - prob_diabetes) * 100
        resultado_texto = [
            html.Span("Resultado: ", style={'color': 'black'}),
            html.Span(f"Risco de Diabetes (Prob: {prob_diabetes * 100:.1f}%)", style={'color': 'red'})
        ]
    else:
        prob_nao_diabetico = (1 - prob_diabetes) * 100
        resultado_texto = [
            html.Span("Resultado: ", style={'color': 'black'}),
            html.Span(f"Não Diabético (Prob: {prob_nao_diabetico:.1f}%)", style={'color': 'green'})
        ]

    
    shap_values_array = explainer.shap_values(input_scaled)
    if isinstance(shap_values_array, list):
        shap_values_local = shap_values_array[1][0]
        expected_value_local = explainer.expected_value[1]
    else:
        shap_values_local = shap_values_array[0]
        expected_value_local = explainer.expected_value
    
    try:
        # Gerar o force plot
        force_plot = shap.force_plot(
            expected_value_local,
            shap_values_local,
            input_df.iloc[0],
            matplotlib=False
        )
        
        # Converter para HTML com o JavaScript necessário
        shap_html = f"""
        <html>
        <head>
            <script src="https://cdn.jsdelivr.net/npm/d3@5"></script>
            {shap.getjs()}
        </head>
        <body style="margin: 0; padding: 0;">
            {force_plot.html()}
        </body>
        </html>
        """
    except Exception as e:
        print(f"Erro ao gerar force plot: {e}")
        shap_html = f"<html><body><p style='color: red;'>Erro ao gerar gráfico: {str(e)}</p></body></html>"
    
    return resultado_texto, shap_html


# Callback para o SHAP GLOBAL (Corrigido)
@app.callback(
    [Output('shap-global-bar-plot', 'figure'),
     Output('shap-global-beeswarm-plot', 'src')],
    Input('tabs-main', 'value')
)
def update_global_shap_plot(tab_value):
    if tab_value == 'tab-2':
        
        mean_abs_shap = np.abs(shap_values_global_class1).mean(0)
        df_importance = pd.DataFrame({
            'Feature': features,
            'Importância Média (SHAP)': mean_abs_shap
        }).sort_values(by='Importância Média (SHAP)', ascending=True)
        
        fig_bar = px.bar(
            df_importance.tail(20),
            x='Importância Média (SHAP)',
            y='Feature',
            orientation='h',
            title='Top 20 Features que mais impactam a predição'
        )
        fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
        
        buf = io.BytesIO()
        plt.figure(figsize=(6, 5))
        
        shap.summary_plot(
            shap_values_global_class1, 
            X_test, # Correção: Usar X_test original
            show=False,
            max_display=15 
        )
        
        plt.title("SHAP Beeswarm Plot (Classe: Risco de Diabetes)", fontsize=16)
        plt.tight_layout()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close() 
        
        data = base64.b64encode(buf.getbuffer()).decode('utf-8')
        src_beeswarm = f"data:image/png;base64,{data}"
        
        return fig_bar, src_beeswarm
    
    return go.Figure(), ""


# --- NOVO CALLBACK: GRÁFICO DE MATRIZ DE CONFUSÃO ---
@app.callback(
    Output('confusion-matrix-graph', 'figure'),
    [Input('tabs-main', 'value'),
     Input('model-cm-dropdown', 'value')]
)
def update_confusion_matrix_graph(tab_value, selected_model):
    if tab_value == 'tab-3' and selected_model:
        model_data = df_metrics[df_metrics['Modelo'] == selected_model].iloc[0]
        
        tn = model_data['TN']
        fp = model_data['FP']
        fn = model_data['FN']
        tp = model_data['TP']
        
        z = [[tn, fp], [fn, tp]]
        x_labels = ['Previsto: Não Diabético (0)', 'Previsto: Risco (1)']
        y_labels = ['Real: Não Diabético (0)', 'Real: Risco (1)']
        
        fig = px.imshow(
            z,
            x=x_labels,
            y=y_labels,
            text_auto=True,
            color_continuous_scale='Blues',
            title=f"Matriz de Confusão - {selected_model}"
        )
        
        fig.update_layout(
            xaxis_title="Valores Previstos pelo Modelo",
            yaxis_title="Valores Reais (Verdadeiros)"
        )
        return fig
    
    return go.Figure()

# --- 7. RODAR O APP ---
if __name__ == '__main__':
    app.run(debug=True, port=8052)