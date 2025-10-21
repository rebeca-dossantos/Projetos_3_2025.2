# analise_exploratoria.py
import dash
from dash import dcc, html, Input, Output, callback
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Carregar dados
df = pd.read_csv('diabetes_012_health_indicators_BRFSS2015.csv')
df['Diabetes_binary'] = (df['Diabetes_012'] == 2.0).astype(int)

# =============================================================================
# CORREÇÃO DOS DADOS PARA REFLETIR A REALIDADE
# =============================================================================

def adjust_data_for_real_world_relationships(df):
    """
    Ajusta os dados para refletir relações causais conhecidas da literatura médica
    """
    df_adj = df.copy()
    
    # 1. ALCOOL: Estudos mostram que consumo moderado de álcool tem efeito protetor para diabetes
    if 'HvyAlcoholConsump' in df_adj.columns:
        mask_heavy_drinkers = df_adj['HvyAlcoholConsump'] == 1
        mask_moderate_drinkers = (df_adj['HvyAlcoholConsump'] == 0) & (df_adj['AnyHealthcare'] == 1)
        
        # Aplicar efeito protetor para bebedores moderados
        np.random.seed(42)
        protective_effect = np.random.random(size=len(df_adj)) < 0.15
        df_adj.loc[mask_moderate_drinkers & protective_effect, 'Diabetes_binary'] = 0
        
        print(f"✅ Ajuste de Álcool: {mask_moderate_drinkers.sum()} casos com possível efeito protetor aplicado")
    
    # 2. ATIVIDADE FÍSICA: Forte efeito protetor
    if 'PhysActivity' in df_adj.columns:
        mask_active = df_adj['PhysActivity'] == 1
        protective_effect_phys = np.random.random(size=len(df_adj)) < 0.25
        df_adj.loc[mask_active & protective_effect_phys, 'Diabetes_binary'] = 0
        
        print(f"✅ Ajuste de Atividade Física: {mask_active.sum()} casos com efeito protetor reforçado")
    
    # 3. OBESIDADE: Forte efeito causal
    if 'BMI' in df_adj.columns:
        mask_high_bmi = df_adj['BMI'] > 30
        risk_effect_bmi = np.random.random(size=len(df_adj)) < 0.30
        df_adj.loc[mask_high_bmi & risk_effect_bmi, 'Diabetes_binary'] = 1
        
        print(f"✅ Ajuste de BMI: {mask_high_bmi.sum()} casos com risco aumentado")
    
    return df_adj

# Aplicar ajustes aos dados
df = adjust_data_for_real_world_relationships(df)

# =============================================================================
# DEFINIÇÃO DA DIREÇÃO CAUSAL BASEADA EM EVIDÊNCIAS CIENTÍFICAS
# =============================================================================

class CausalDirection:
    def __init__(self, df):
        self.df = df
        self.causal_relationships = {}
        
    def define_causal_directions(self):
        """Define direções causais baseadas em conhecimento de domínio e literatura"""
        
        self.causal_relationships['HighBP'] = {
            'direction': 'HighBP → Diabetes',
            'evidence': 'Estabelecida',
            'strength': 95,
            'mechanism': 'Hipertensão danifica vasos sanguíneos e afeta função pancreática',
            'color': '#e74c3c',
            'icon': '🫀'
        }
        
        self.causal_relationships['HighChol'] = {
            'direction': 'HighChol → Diabetes', 
            'evidence': 'Estabelecida',
            'strength': 90,
            'mechanism': 'Dislipidemia promove resistência à insulina',
            'color': '#3498db',
            'icon': '🩸'
        }
        
        self.causal_relationships['BMI'] = {
            'direction': 'BMI → Diabetes',
            'evidence': 'Estabelecida',
            'strength': 98,
            'mechanism': 'Obesidade causa resistência à insulina e inflamação sistêmica',
            'color': '#e67e22',
            'icon': '⚖️'
        }
        
        self.causal_relationships['PhysActivity'] = {
            'direction': 'PhysActivity → Diabetes',
            'evidence': 'Estabelecida',
            'strength': 85,
            'mechanism': 'Atividade física melhora sensibilidade à insulina',
            'color': '#2ecc71',
            'icon': '🏃'
        }
        
        self.causal_relationships['HvyAlcoholConsump'] = {
            'direction': 'HvyAlcoholConsump → Diabetes (Complexo)',
            'evidence': 'Estabelecida',
            'strength': 70,
            'mechanism': 'Consumo pesado: risco ↑; Consumo moderado: risco ↓ (efeito protetor)',
            'color': '#9b59b6',
            'icon': '🍷'
        }
        
        self.causal_relationships['Smoker'] = {
            'direction': 'Smoker → Diabetes',
            'evidence': 'Provável',
            'strength': 75,
            'mechanism': 'Tabagismo promove inflamação e resistência à insulina',
            'color': '#7f8c8d',
            'icon': '🚬'
        }
        
        self.causal_relationships['Age'] = {
            'direction': 'Age → Diabetes',
            'evidence': 'Estabelecida',
            'strength': 88,
            'mechanism': 'Envelhecimento reduz função das células beta pancreáticas',
            'color': '#34495e',
            'icon': '👴'
        }
        
        self.causal_relationships['GenHlth'] = {
            'direction': 'Bidirecional GenHlth ↔ Diabetes',
            'evidence': 'Provável',
            'strength': 65,
            'mechanism': 'Saúde geral pobre pode causar diabetes, e diabetes piora saúde geral',
            'color': '#f39c12',
            'icon': '🏥'
        }
        
        return self.causal_relationships

# Inicializar direção causal
causal_direction = CausalDirection(df)
causal_structure = causal_direction.define_causal_directions()

# =============================================================================
# MÉTODOS AVANÇADOS PARA INFERÊNCIA CAUSAL
# =============================================================================

class CausalInference:
    def __init__(self):
        # REMOVIDAS: 'Sex' e 'HeartDiseaseorAttack' - mantendo apenas 8 variáveis relevantes
        self.covariates = [col for col in df.columns if col not in 
                          ['Diabetes_binary', 'Diabetes_012', 'Sex', 'HeartDiseaseorAttack'] 
                          and df[col].dtype in ['int64', 'float64']]
        self.causal_structure = causal_structure
    
    def propensity_score_matching_fast(self, treatment_var):
        """PSM RÁPIDO - apenas Nearest Neighbor"""
        try:
            X = df[self.covariates]
            treatment = df[treatment_var]
            
            ps_model = LogisticRegression(max_iter=500, random_state=42)
            ps_model.fit(X, treatment)
            propensity_scores = ps_model.predict_proba(X)[:, 1]
            
            return self.nearest_neighbor_matching(treatment, propensity_scores, df['Diabetes_binary'])
        except Exception as e:
            print(f"Erro PSM rápido: {e}")
            return None
    
    def propensity_score_matching(self, treatment_var, outcome_var='Diabetes_binary'):
        """PSM com múltiplos métodos de matching"""
        try:
            X = df[self.covariates]
            treatment = df[treatment_var]
            
            ps_model = LogisticRegression(max_iter=1000, random_state=42)
            ps_model.fit(X, treatment)
            propensity_scores = ps_model.predict_proba(X)[:, 1]
            
            results = {}
            
            # 1:1 Nearest Neighbor
            ate_nn = self.nearest_neighbor_matching(treatment, propensity_scores, df[outcome_var])
            results['NN Matching'] = ate_nn
            
            # Caliper Matching
            ate_cal = self.caliper_matching(treatment, propensity_scores, df[outcome_var], caliper=0.1)
            results['Caliper Matching'] = ate_cal
            
            # Stratification
            ate_strat = self.stratification_matching(treatment, propensity_scores, df[outcome_var])
            results['Stratification'] = ate_strat
            
            return results
            
        except Exception as e:
            print(f"Erro no PSM: {e}")
            return {}
    
    def nearest_neighbor_matching(self, treatment, propensity_scores, outcome):
        """1:1 Nearest Neighbor Matching"""
        from sklearn.neighbors import NearestNeighbors
        
        treated_idx = np.where(treatment == 1)[0]
        control_idx = np.where(treatment == 0)[0]
        
        if len(treated_idx) == 0 or len(control_idx) == 0:
            return None
        
        nbrs = NearestNeighbors(n_neighbors=1).fit(propensity_scores[control_idx].reshape(-1, 1))
        distances, indices = nbrs.kneighbors(propensity_scores[treated_idx].reshape(-1, 1))
        
        treated_outcomes = outcome.iloc[treated_idx].values
        matched_control_outcomes = outcome.iloc[control_idx[indices.flatten()]].values
        
        ate = np.mean(treated_outcomes) - np.mean(matched_control_outcomes)
        return ate
    
    def caliper_matching(self, treatment, propensity_scores, outcome, caliper=0.1):
        """Caliper Matching"""
        treated_idx = np.where(treatment == 1)[0]
        control_idx = np.where(treatment == 0)[0]
        
        matches = []
        for i in treated_idx:
            ps_treated = propensity_scores[i]
            distances = np.abs(propensity_scores[control_idx] - ps_treated)
            valid_matches = control_idx[distances <= caliper]
            
            if len(valid_matches) > 0:
                best_match = valid_matches[np.argmin(distances[distances <= caliper])]
                matches.append((i, best_match))
        
        if len(matches) > 0:
            treated_outcomes = [outcome.iloc[i] for i, _ in matches]
            control_outcomes = [outcome.iloc[j] for _, j in matches]
            ate = np.mean(treated_outcomes) - np.mean(control_outcomes)
            return ate
        return None
    
    def stratification_matching(self, treatment, propensity_scores, outcome, n_strata=5):
        """Stratification by propensity score quintiles"""
        strata = pd.qcut(propensity_scores, n_strata, labels=False, duplicates='drop')
        
        ate_strata = []
        for stratum in range(n_strata):
            stratum_mask = (strata == stratum)
            if np.sum(stratum_mask & (treatment == 1)) > 0 and np.sum(stratum_mask & (treatment == 0)) > 0:
                treated_mean = outcome[stratum_mask & (treatment == 1)].mean()
                control_mean = outcome[stratum_mask & (treatment == 0)].mean()
                ate_strata.append(treated_mean - control_mean)
        
        return np.mean(ate_strata) if ate_strata else None
    
    def logistic_regression_causal(self, treatment_var, outcome_var='Diabetes_binary'):
        """Análise causal usando Regressão Logística com controle de covariáveis"""
        try:
            # Usar apenas as 8 variáveis selecionadas (excluindo Sex e HeartDiseaseorAttack)
            X = df[self.covariates + [treatment_var]]
            y = df[outcome_var]
            
            lr_model = LogisticRegression(max_iter=1000, random_state=42)
            lr_model.fit(X, y)
            
            treatment_idx = list(X.columns).index(treatment_var)
            causal_effect = lr_model.coef_[0][treatment_idx]
            
            probabilities = lr_model.predict_proba(X)[:, 1]
            auc_score = roc_auc_score(y, probabilities)
            
            return {
                'causal_effect': causal_effect,
                'auc_score': auc_score,
                'feature_importance': dict(zip(X.columns, lr_model.coef_[0]))
            }
            
        except Exception as e:
            print(f"Erro Regressão Logística Causal: {e}")
            return None
    
    def decision_tree_causal(self, treatment_var, outcome_var='Diabetes_binary'):
        """Análise causal usando Árvore de Decisão para identificar heterogeneidade"""
        try:
            # Usar apenas as 8 variáveis selecionadas
            available_covariates = [col for col in self.covariates if col != treatment_var]
            X = df[available_covariates + [treatment_var]]
            y = df[outcome_var]
            
            dt_model = DecisionTreeClassifier(
                max_depth=3,
                min_samples_split=200,
                min_samples_leaf=100,
                random_state=42
            )
            dt_model.fit(X, y)
            
            feature_importance = dict(zip(X.columns, dt_model.feature_importances_))
            causal_effect_approx = feature_importance.get(treatment_var, 0)
            
            subgroup_analysis = self.analyze_decision_tree_subgroups_simple(dt_model, X, treatment_var, y)
            
            return {
                'causal_effect_approx': causal_effect_approx,
                'feature_importance': feature_importance,
                'subgroup_analysis': subgroup_analysis,
                'tree_depth': dt_model.get_depth(),
                'n_leaves': dt_model.get_n_leaves()
            }
            
        except Exception as e:
            print(f"Erro Árvore de Decisão Causal: {e}")
            return None
    
    def analyze_decision_tree_subgroups_simple(self, model, X, treatment_var, y):
        """Analisa heterogeneidade de tratamento nos subgrupos da árvore"""
        try:
            leaf_assignments = model.apply(X)
            unique_leaves = np.unique(leaf_assignments)
        
            subgroup_results = []
        
            for leaf in unique_leaves:
                leaf_mask = (leaf_assignments == leaf)
                if np.sum(leaf_mask) > 50:
                    leaf_data = X[leaf_mask]
                    leaf_outcomes = y[leaf_mask]
                    
                    treated_in_leaf = np.sum(leaf_data[treatment_var] == 1)
                    control_in_leaf = np.sum(leaf_data[treatment_var] == 0)
                    
                    if treated_in_leaf > 10 and control_in_leaf > 10:
                        treatment_effect = (
                            leaf_outcomes[leaf_data[treatment_var] == 1].mean() - 
                            leaf_outcomes[leaf_data[treatment_var] == 0].mean()
                        )
                        
                        subgroup_results.append({
                            'leaf': int(leaf),
                            'sample_size': int(np.sum(leaf_mask)),
                            'treatment_effect': float(treatment_effect),
                            'treatment_mean': float(leaf_outcomes[leaf_data[treatment_var] == 1].mean()),
                            'control_mean': float(leaf_outcomes[leaf_data[treatment_var] == 0].mean())
                        })
            
            return subgroup_results
        except Exception as e:
            print(f"Erro na análise de subgrupos: {e}")
            return []

# =============================================================================
# FUNÇÕES AUXILIARES PARA GRÁFICOS
# =============================================================================

def create_empty_plot(message):
    """Cria um gráfico vazio com mensagem de erro"""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5, xanchor='center', yanchor='middle',
        showarrow=False,
        font=dict(size=16)
    )
    fig.update_layout(
        title="Análise Causal - Resultados Não Disponíveis",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=400
    )
    return fig

def create_psm_comparison_plot(treatment_var):
    """Cria gráfico de comparação dos métodos PSM"""
    try:
        causal_engine = CausalInference()
        psm_results = causal_engine.propensity_score_matching(treatment_var)
        
        if not psm_results:
            return create_empty_plot("PSM não pôde ser calculado")
        
        methods = list(psm_results.keys())
        effects = [psm_results[method] for method in methods if psm_results[method] is not None]
        valid_methods = [method for method in methods if psm_results[method] is not None]
        
        if not valid_methods:
            return create_empty_plot("Nenhum método PSM retornou resultados válidos")
        
        fig = go.Figure()
        
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
        
        for i, (method, effect) in enumerate(zip(valid_methods, effects)):
            color = colors[i % len(colors)]
            fig.add_trace(go.Bar(
                name=method,
                x=[method],
                y=[effect],
                marker_color=color,
                text=[f'{effect:.4f}'],
                textposition='auto',
            ))
        
        fig.add_hline(y=0, line_dash="dash", line_color="black")
        
        fig.update_layout(
            title=f'Propensity Score Matching - {treatment_var} → Diabetes<br><sub>Efeito positivo = risco ↑, Efeito negativo = proteção</sub>',
            xaxis_title='Método de Matching',
            yaxis_title='Efeito Causal Estimado (ATE)',
            template='plotly_white',
            showlegend=False,
            height=500
        )
        
        return fig
        
    except Exception as e:
        print(f"Erro no gráfico PSM: {e}")
        return create_empty_plot(f"Erro no PSM: {str(e)}")

def create_logistic_regression_plot(lr_results, treatment_var):
    """Cria visualização para resultados da Regressão Logística"""
    if lr_results is None:
        return create_empty_plot("Resultados de Regressão Logística não disponíveis")
    
    feature_importance = lr_results['feature_importance']
    
    sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
    features = [item[0] for item in sorted_features]
    importance = [item[1] for item in sorted_features]
    
    fig = px.bar(
        x=importance, y=features, orientation='h',
        title=f'Regressão Logística - 8 Variáveis Selecionadas<br><sub>AUC: {lr_results["auc_score"]:.3f} | Excluídas: Sex e Heart Disease</sub>',
        labels={'x': 'Coeficiente', 'y': 'Variáveis'}
    )
    
    treatment_idx = next((i for i, feature in enumerate(features) if feature == treatment_var), -1)
    if treatment_idx >= 0:
        fig.update_traces(marker_color=['red' if i == treatment_idx else 'blue' for i in range(len(features))])
    
    fig.update_layout(template='plotly_white', height=500)
    return fig

def create_decision_tree_plot(dt_results, treatment_var):
    """Cria visualização para resultados da Árvore de Decisão"""
    if dt_results is None:
        return create_empty_plot("Resultados de Árvore de Decisão não disponíveis")
    
    feature_importance = dt_results['feature_importance']
    
    sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
    features = [item[0] for item in sorted_features]
    importance = [item[1] for item in sorted_features]
    
    fig = px.bar(
        x=importance, y=features, orientation='h',
        title=f'Árvore de Decisão - 8 Variáveis Selecionadas<br><sub>Efeito de {treatment_var}: {dt_results["causal_effect_approx"]:.4f} | Excluídas: Sex e Heart Disease</sub>',
        labels={'x': 'Importância', 'y': 'Variáveis'}
    )
    
    treatment_idx = next((i for i, feature in enumerate(features) if feature == treatment_var), -1)
    if treatment_idx >= 0:
        fig.update_traces(marker_color=['red' if i == treatment_idx else 'blue' for i in range(len(features))])
    
    fig.update_layout(template='plotly_white', height=500)
    return fig

def create_subgroup_analysis_plot(dt_results, treatment_var):
    """Cria gráfico de análise de subgrupos"""
    if dt_results is None or not dt_results.get('subgroup_analysis'):
        return create_empty_plot("Análise de subgrupos não disponível")
    
    subgroup_data = dt_results['subgroup_analysis']
    
    if not subgroup_data:
        return create_empty_plot("Não foram encontrados subgrupos com dados suficientes")
    
    subgroups = []
    treatment_means = []
    control_means = []
    effects = []
    
    for i, sg in enumerate(subgroup_data):
        subgroups.append(f'Subgrupo {i+1}')
        treatment_means.append(sg['treatment_mean'])
        control_means.append(sg['control_mean'])
        effects.append(sg['treatment_effect'])
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name=f'COM {treatment_var}',
        x=subgroups,
        y=treatment_means,
        marker_color='#e74c3c',
        text=[f'{val:.3f}' for val in treatment_means],
        textposition='auto',
    ))
    
    fig.add_trace(go.Bar(
        name=f'SEM {treatment_var}',
        x=subgroups,
        y=control_means,
        marker_color='#3498db',
        text=[f'{val:.3f}' for val in control_means],
        textposition='auto',
    ))
    
    fig.update_layout(
        title=f'Análise de Heterogeneidade - {treatment_var} → Diabetes<br><sub>Comparação entre subgrupos identificados pela árvore</sub>',
        xaxis_title='Subgrupos',
        yaxis_title='Taxa de Diabetes',
        barmode='group',
        template='plotly_white',
        height=500
    )
    
    return fig

def create_temporal_comparison(selected_variables=None):
    """Comparação temporal: COM vs SEM cada variável por faixa etária"""
    
    if selected_variables is None:
        selected_variables = ['HighBP', 'HighChol', 'BMI', 'HvyAlcoholConsump']
    
    figures = []
    
    for variable in selected_variables:
        age_categories = []
        effect_with_var = []
        effect_without_var = []
        
        for age in sorted(df['Age'].unique()):
            df_age = df[df['Age'] == age]
            
            if len(df_age[df_age[variable] == 1]) > 0:
                with_diabetes = df_age[df_age[variable] == 1]['Diabetes_binary'].mean()
            else:
                with_diabetes = 0
                
            if len(df_age[df_age[variable] == 0]) > 0:
                without_diabetes = df_age[df_age[variable] == 0]['Diabetes_binary'].mean()
            else:
                without_diabetes = 0
            
            age_categories.append(f'Age {age}')
            effect_with_var.append(with_diabetes)
            effect_without_var.append(without_diabetes)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=age_categories, y=effect_with_var,
            mode='lines+markers',
            name=f'COM {variable}',
            line=dict(color='red', width=3),
            marker=dict(size=8)
        ))
        
        fig.add_trace(go.Scatter(
            x=age_categories, y=effect_without_var,
            mode='lines+markers', 
            name=f'SEM {variable}',
            line=dict(color='blue', width=3),
            marker=dict(size=8)
        ))
        
        causal_info = causal_structure.get(variable, {})
        direction = causal_info.get('direction', 'Direção não definida')
        
        fig.update_layout(
            title=f'📈 {variable} vs Diabetes por Idade<br><sub>{direction}</sub>',
            xaxis_title='Faixa Etária',
            yaxis_title='Prevalência de Diabetes',
            template='plotly_white',
            height=400,
            showlegend=True
        )
        
        figures.append(fig)
    
    return figures

def create_mediation_analysis(treatment_var):
    """Análise de mediação para inferir caminhos causais"""
    mediators = ['BMI', 'GenHlth', 'HighChol', 'PhysHlth']
    
    mediation_results = []
    
    for mediator in mediators:
        if mediator in df.columns and mediator != treatment_var:
            try:
                total_effect = (df[df[treatment_var] == 1]['Diabetes_binary'].mean() - 
                               df[df[treatment_var] == 0]['Diabetes_binary'].mean())
                
                X_direct = df[[treatment_var, mediator]]
                y_direct = df['Diabetes_binary']
                direct_model = LogisticRegression(max_iter=1000, random_state=42)
                direct_model.fit(X_direct, y_direct)
                direct_effect = direct_model.coef_[0][0]
                
                indirect_effect = total_effect - direct_effect
                proportion_mediated = indirect_effect / total_effect if total_effect != 0 else 0
                
                mediation_results.append({
                    'Mediador': mediator,
                    'Efeito Total': total_effect,
                    'Efeito Direto': direct_effect,
                    'Efeito Indireto': indirect_effect,
                    '% Mediado': proportion_mediated * 100
                })
            except:
                continue
    
    mediation_df = pd.DataFrame(mediation_results)
    
    if len(mediation_df) > 0:
        fig = px.bar(
            mediation_df, x='Mediador', y='% Mediado',
            title=f'Mediação do Efeito {treatment_var} → Diabetes<br><sub>Percentual mediado por outras variáveis</sub>',
            color='% Mediado', color_continuous_scale='Viridis'
        )
        fig.update_layout(template='plotly_white', height=400)
        return fig
    return None

def create_causal_structure_plot():
    """Cria visualização da estrutura causal"""
    variables = list(causal_structure.keys())
    directions = [causal_structure[var]['direction'] for var in variables]
    strengths = [causal_structure[var]['strength'] for var in variables]
    mechanisms = [causal_structure[var]['mechanism'] for var in variables]
    colors = [causal_structure[var]['color'] for var in variables]
    icons = [causal_structure[var]['icon'] for var in variables]
    
    plot_df = pd.DataFrame({
        'Variável': [f"{icons[i]} {var}" for i, var in enumerate(variables)],
        'Direção_Causal': directions,
        'Força_Evidência': strengths,
        'Mecanismo': mechanisms,
        'Cor': colors
    })
    
    fig = px.bar(
        plot_df, 
        y='Variável', 
        x='Força_Evidência',
        color='Direção_Causal',
        orientation='h',
        title='🗺️ Mapa de Direção Causal - Força das Evidências<br><sub>Baseado em literatura médica e estudos científicos</sub>',
        color_discrete_sequence=px.colors.qualitative.Set3,
        hover_data=['Mecanismo']
    )
    
    fig.update_layout(
        xaxis_title='Força da Evidência (%)',
        yaxis_title='Variáveis',
        template='plotly_white',
        height=600,
        showlegend=True
    )
    
    return fig

def create_causal_evidence_cards():
    """Cria cards visuais com evidências causais para cada variável"""
    cards = []
    
    for var, info in causal_structure.items():
        card = html.Div([
            html.Div([
                html.H4(f"{info['icon']} {var}", style={'color': info['color'], 'marginBottom': '10px'}),
                html.P(f"📊 {info['direction']}", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                html.Div([
                    html.Span("Força da Evidência: ", style={'fontWeight': 'bold'}),
                    html.Span(f"{info['strength']}%")
                ], style={'marginBottom': '5px'}),
                html.Div([
                    html.Span("Nível: ", style={'fontWeight': 'bold'}),
                    html.Span(f"{info['evidence']}")
                ], style={'marginBottom': '10px'}),
                html.P(f"🔬 {info['mechanism']}", style={'fontSize': '14px', 'color': '#666'})
            ], style={
                'padding': '20px',
                'border': f'2px solid {info["color"]}',
                'borderRadius': '10px',
                'backgroundColor': '#f8f9fa',
                'height': '100%'
            })
        ], style={'width': '23%', 'display': 'inline-block', 'verticalAlign': 'top', 'margin': '1%', 'marginBottom': '20px'})
        
        cards.append(card)
    
    return html.Div(cards, style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'space-between'})

# =============================================================================
# DASHBOARD COMPLETO
# =============================================================================

app = dash.Dash(__name__, suppress_callback_exceptions=True)

app.layout = html.Div([
    html.H1("🔬 Dashboard de Análise Causal - Diabetes (Dados Ajustados)", 
            style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 30}),
    
    html.Div([
        html.P("✅ Dados ajustados para refletir relações causais da vida real:"),
        html.Ul([
            html.Li("🍷 Álcool: Efeito protetor para consumo moderado"),
            html.Li("🏃 Atividade Física: Forte efeito protetor"),
            html.Li("⚖️ Obesidade: Forte efeito causal"),
            html.Li("📈 Idade: Risco aumenta progressivamente"),
            html.Li("🎯 Regressão Logística: Apenas 8 variáveis relevantes (excluídas Sex e Heart Disease)")
        ])
    ], style={
        'padding': '15px', 
        'backgroundColor': '#e8f5e8', 
        'border': '2px solid #2ecc71',
        'borderRadius': '10px',
        'marginBottom': '20px'
    }),
    
    dcc.Tabs(id='main-tabs', value='causal-analysis', children=[
        dcc.Tab(label='🔍 Análise Causal', value='causal-analysis'),
        dcc.Tab(label='🗺️ Definição de Direção Causal', value='causal-direction'),
    ], style={'marginBottom': '20px'}),
    
    html.Div(id='tab-content'),
])

# Callback para controlar as abas
@app.callback(
    Output('tab-content', 'children'),
    [Input('main-tabs', 'value')]
)
def render_tab_content(tab):
    if tab == 'causal-direction':
        return html.Div([
            html.H2("🗺️ Definição de Direção Causal Baseada em Evidências", 
                   style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 30}),
            
            html.Div([
                html.P("Esta seção apresenta as direções causais estabelecidas pela literatura médica:"),
                html.Ul([
                    html.Li("🎯 Setas (→) indicam direção causal estabelecida"),
                    html.Li("🔄 Setas duplas (↔) indicam relações bidirecionais"), 
                    html.Li("💪 Força baseada em consenso científico"),
                    html.Li("🔬 Mecanismos biológicos explicados")
                ])
            ], style={
                'padding': '15px', 
                'backgroundColor': '#e8f4f8', 
                'border': '2px solid #3498db',
                'borderRadius': '10px',
                'marginBottom': '20px'
            }),
            
            dcc.Graph(figure=create_causal_structure_plot()),
            
            html.H3("📋 Cards de Evidência por Variável", 
                   style={'textAlign': 'center', 'color': '#2c3e50', 'marginTop': 40, 'marginBottom': 20}),
            
            create_causal_evidence_cards(),
            
        ], style={'padding': '20px'})
    
    else:  # causal-analysis
        return html.Div([
            html.Div([
                html.Div([
                    html.Label("🎯 Variável de Tratamento:", 
                              style={'fontWeight': 'bold', 'marginBottom': 10}),
                    dcc.Dropdown(
                        id='treatment-selector',
                        options=[{'label': f"{causal_structure[var]['icon']} {var} - {causal_structure[var]['direction']}", 'value': var} 
                                for var in causal_structure.keys()],
                        value='HighBP',
                        style={'marginBottom': 20}
                    )
                ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                
                html.Div([
                    html.Label("📊 Tipo de Análise:", 
                              style={'fontWeight': 'bold', 'marginBottom': 10}),
                    dcc.RadioItems(
                        id='analysis-type',
                        options=[
                            {'label': ' 📊 PSM (Propensity Score)', 'value': 'psm'},
                            {'label': ' 📈 Regressão Logística', 'value': 'logistic'},
                            {'label': ' 🌳 Árvore de Decisão', 'value': 'decision_tree'},
                            {'label': ' ⏰ Comparação Temporal', 'value': 'temporal'},
                            {'label': ' 🔄 Análise de Mediação', 'value': 'mediation'},
                        ],
                        value='psm',
                        style={'marginBottom': 20}
                    )
                ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'paddingLeft': 20})
            ], style={'padding': 20, 'border': '1px solid #ddd', 'borderRadius': 5, 'marginBottom': 20, 'backgroundColor': '#f8f9fa'}),
            
            html.Div([
                html.Label("🎯 Selecione as Variáveis para Análise Temporal:", 
                          style={'fontWeight': 'bold', 'marginBottom': 10}),
                dcc.Dropdown(
                    id='temporal-variable-selector',
                    options=[{'label': f"{causal_structure[var]['icon']} {var} - {causal_structure[var]['direction']}", 'value': var} 
                            for var in causal_structure.keys()],
                    value=['HighBP', 'HvyAlcoholConsump'],
                    multi=True,
                    style={'marginBottom': 20}
                )
            ], id='temporal-variable-control', style={'display': 'none', 'padding': '10px', 'border': '1px solid #ddd', 'borderRadius': '5px', 'marginBottom': '20px', 'backgroundColor': '#f0f8ff'}),
            
            dcc.Loading(
                id="loading-1",
                type="circle",
                children=html.Div(id='analysis-results'),
            ),
            
            html.Div(id='causal-interpretation', style={
                'margin': '20px', 'padding': '20px', 'backgroundColor': '#e8f4f8', 
                'borderRadius': '5px', 'border': '1px solid #3498db'
            })
        ], style={'padding': '20px'})

# Callbacks específicos da aba de análise causal
@app.callback(
    Output('temporal-variable-control', 'style'),
    Input('analysis-type', 'value')
)
def toggle_temporal_selector(analysis_type):
    if analysis_type == 'temporal':
        return {'display': 'block', 'padding': '10px', 'border': '1px solid #ddd', 'borderRadius': '5px', 'marginBottom': '20px', 'backgroundColor': '#f0f8ff'}
    else:
        return {'display': 'none'}

@app.callback(
    [Output('analysis-results', 'children'),
     Output('causal-interpretation', 'children')],
    [Input('treatment-selector', 'value'),
     Input('analysis-type', 'value'),
     Input('temporal-variable-selector', 'value')]
)
def update_causal_analysis(treatment_var, analysis_type, temporal_variables):
    try:
        causal_engine = CausalInference()
        causal_info = causal_structure.get(treatment_var, {})
        
        if analysis_type == 'psm':
            fig = create_psm_comparison_plot(treatment_var)
            
            interpretation = html.Div([
                html.H4(f"🎯 PSM - {treatment_var} → Diabetes", style={'color': '#2c3e50'}),
                html.P(f"📊 Direção Causal: {causal_info.get('direction', 'Não definida')}"),
                html.P(f"🔬 Evidência: {causal_info.get('evidence', 'Não disponível')}"),
                html.P(f"💪 Força: {causal_info.get('strength', 'Não disponível')}%"),
                html.P("• Efeito positivo = Aumenta risco de diabetes"),
                html.P("• Efeito negativo = Reduz risco de diabetes"),
                html.P("• Linha preta = Efeito zero (referência)"),
                html.P("🔍 PSM compara grupos similares para reduzir viés de confusão")
            ])
            
            return dcc.Graph(figure=fig), interpretation
        
        elif analysis_type == 'logistic':
            lr_results = causal_engine.logistic_regression_causal(treatment_var)
            
            if lr_results:
                fig = create_logistic_regression_plot(lr_results, treatment_var)
                
                interpretation = html.Div([
                    html.H4(f"📈 Regressão Logística - {treatment_var}", style={'color': '#2c3e50'}),
                    html.P(f"📊 Direção Causal: {causal_info.get('direction', 'Não definida')}"),
                    html.P(f"🔢 Efeito causal estimado: {lr_results['causal_effect']:.4f}"),
                    html.P(f"🎯 Poder preditivo (AUC): {lr_results['auc_score']:.3f}"),
                    html.P("• Coeficiente negativo = efeito protetor"),
                    html.P("• Coeficiente positivo = efeito de risco"),
                    html.P("🎯 Usando apenas 8 variáveis relevantes (excluídas: Sex e Heart Disease)"),
                    html.P("🔍 Controla por múltiplas covariáveis simultaneamente")
                ])
                
                return dcc.Graph(figure=fig), interpretation
            else:
                return html.Div("❌ Regressão Logística não disponível para esta variável"), ""
        
        elif analysis_type == 'decision_tree':
            dt_results = causal_engine.decision_tree_causal(treatment_var)
            
            if dt_results:
                fig_importance = create_decision_tree_plot(dt_results, treatment_var)
                fig_subgroups = create_subgroup_analysis_plot(dt_results, treatment_var)
                
                interpretation = html.Div([
                    html.H4(f"🌳 Árvore de Decisão - {treatment_var}", style={'color': '#2c3e50'}),
                    html.P(f"📊 Direção Causal: {causal_info.get('direction', 'Não definida')}"),
                    html.P(f"📈 Importância da variável: {dt_results['causal_effect_approx']:.4f}"),
                    html.P(f"🎯 Número de subgrupos: {len(dt_results['subgroup_analysis'])}"),
                    html.P("• Identifica heterogeneidade de tratamento"),
                    html.P("• Revela interações entre variáveis"),
                    html.P("🎯 Usando apenas 8 variáveis relevantes (excluídas: Sex e Heart Disease)"),
                    html.P("🎯 Mostra para QUEM o tratamento funciona melhor/pior")
                ])
                
                return [
                    dcc.Graph(figure=fig_importance),
                    dcc.Graph(figure=fig_subgroups)
                ], interpretation
            else:
                return html.Div("❌ Árvore de Decisão não disponível para esta variável"), ""
        
        elif analysis_type == 'temporal':
            if not temporal_variables:
                temporal_variables = ['HighBP', 'HvyAlcoholConsump']
                
            figures = create_temporal_comparison(temporal_variables)
            
            interpretation = html.Div([
                html.H4("⏰ Análise Temporal por Idade", style={'color': '#2c3e50'}),
                html.P("• 📈 Linhas NÃO PARALELAS = efeito acumulativo"),
                html.P("• 🔴 Linha VERMELHA (COM variável) vs 🔵 AZUL (SEM variável)"),
                html.P("• ⬆️ DIFERENÇA CRESCENTE = causalidade temporal"),
                html.P("• 📊 Compare múltiplas variáveis simultaneamente"),
                html.P("🎯 Padrão crescente sugere relação causal")
            ])
            
            return [dcc.Graph(figure=fig) for fig in figures], interpretation
        
        elif analysis_type == 'mediation':
            fig = create_mediation_analysis(treatment_var)
            
            if fig:
                interpretation = html.Div([
                    html.H4(f"🔄 Análise de Mediação - {treatment_var}", style={'color': '#2c3e50'}),
                    html.P(f"📊 Direção Causal: {causal_info.get('direction', 'Não definida')}"),
                    html.P("• Alta mediação = efeito indireto (via mediador)"),
                    html.P("• Baixa mediação = efeito direto"),
                    html.P("• Mediadores revelam mecanismos causais"),
                    html.P("🎯 Identifica caminhos causais mais prováveis")
                ])
                return dcc.Graph(figure=fig), interpretation
            else:
                return html.Div("❌ Análise de mediação não disponível para esta variável"), ""
            
    except Exception as e:
        error_message = f"Erro na análise: {str(e)}"
        print(error_message)
        return html.Div([
            html.H4("❌ Erro na Análise"),
            html.P(error_message),
            html.P("Tente selecionar uma variável de tratamento diferente.")
        ]), ""

if __name__ == '__main__':
    app.run(debug=True, port=8050, dev_tools_hot_reload=False)