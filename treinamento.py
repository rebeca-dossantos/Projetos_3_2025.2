import pandas as pd
import numpy as np
import warnings
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# --- Importar Métricas e a Matriz de Confusão ---
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score, 
    recall_score, f1_score, roc_auc_score, confusion_matrix
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

warnings.filterwarnings('ignore')

print("Iniciando o script de treinamento (v3)...")

# --- 1. Carregar e Pré-processar Dados ---
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
print("Dados carregados e pré-processados.")

# --- 2. Definir Features (X) e Alvo (y) ---
features = [col for col in df.columns if col not in ['Diabetes_binary', 'Cluster', 'PCA1', 'PCA2']]
target = 'Diabetes_binary'
X = df[features]
y = df[target]

# --- 3. Separar em Treino e Teste ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Dados divididos: {len(X_train)} para treino, {len(X_test)} para teste.")

# --- 4. COMPARAÇÃO DE MODELOS ---
print("\n--- Iniciando Comparação de Modelos ---")
models = {
    'Regressão Logística': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}
results = []

for model_name, model in models.items():
    print(f"Treinando {model_name}...")
    pipeline_comp = Pipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42)),
        ('model', model)
    ])
    pipeline_comp.fit(X_train, y_train)
    y_pred = pipeline_comp.predict(X_test)
    y_pred_proba = pipeline_comp.predict_proba(X_test)[:, 1]
    
    # --- NOVO: Calcular a Matriz de Confusão ---
    cm = confusion_matrix(y_test, y_pred)
    # cm[0, 0] = TN (Verdadeiro Negativo)
    # cm[0, 1] = FP (Falso Positivo)
    # cm[1, 0] = FN (Falso Negativo)
    # cm[1, 1] = TP (Verdadeiro Positivo)
    
    metrics = {
        'Modelo': model_name,
        'Acurácia': accuracy_score(y_test, y_pred),
        'Precisão': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'ROC AUC': roc_auc_score(y_test, y_pred_proba),
        # --- NOVO: Salvar os 4 valores ---
        'TN': int(cm[0, 0]),
        'FP': int(cm[0, 1]),
        'FN': int(cm[1, 0]),
        'TP': int(cm[1, 1])
    }
    results.append(metrics)

df_metrics = pd.DataFrame(results)
# Exportar os resultados (sem arredondar os inteiros)
df_metrics.to_csv('assets/model_comparison_metrics.csv', index=False, float_format='%.3f')
print("--- Comparação concluída e salva em 'model_comparison_metrics.csv' ---")


# --- 5. TUNING DO MELHOR MODELO (Gradient Boosting) ---
# (Esta parte continua igual)
print("\n--- Iniciando Tuning (GridSearch) do Gradient Boosting ---")
pipeline_gb = Pipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('model', GradientBoostingClassifier(random_state=42))
])
param_grid = {
    'model__n_estimators': [100, 200],
    'model__learning_rate': [0.1],
    'model__max_depth': [3, 5]
}
grid_search = GridSearchCV(estimator=pipeline_gb, param_grid=param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)
best_model_pipeline = grid_search.best_estimator_
print("Tuning concluído!")

# --- 6. Exportar o Modelo Otimizado ---
joblib.dump(best_model_pipeline, 'assets/modelo_classificacao.joblib')
print("Modelo otimizado salvo com sucesso como 'modelo_classificacao.joblib'")
print("Script de treinamento concluído.")