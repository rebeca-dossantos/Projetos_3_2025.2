# ==============================
# VERSÃO CORRIGIDA - CLUSTERIZAÇÃO PARA 250K AMOSTRAS
# ==============================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import glob
from sklearn.utils import shuffle

print("🔄 Carregando dataset de 250k amostras...")

file_path = glob.glob("diabetes*_BRFSS2015.csv")[0]
df = pd.read_csv(file_path)
print(f"✅ Dataset carregado: {df.shape}")

# ------------------------------
# ESTRATÉGIA: Amostragem para análise
# ------------------------------
# Para datasets grandes, usamos uma amostra para encontrar o K ideal
SAMPLE_SIZE = 100000  # Amostra de 100k para análise

df_clean = df.dropna().select_dtypes(include=[np.number])
df_sampled = shuffle(df_clean, random_state=42).iloc[:SAMPLE_SIZE]

print(f"📊 Usando amostra de {len(df_sampled)} para análise de clusters...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_sampled)

# ------------------------------
# Método RÁPIDO do Cotovelo e Silhueta
# ------------------------------
print("📊 Calculando método do cotovelo e silhueta (versão acelerada)...")

inertia = []
silhouette_scores = []
calinski_scores = []
K_range = range(2, 10)  # Menos valores de K para testar

for k in K_range:
    print(f"  Processando K={k}...")
    
    # Usa MiniBatchKMeans que é muito mais rápido
    kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=1000, n_init=3)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    inertia.append(kmeans.inertia_)
    
    # Calcula ambas as métricas para todos os K
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    silhouette_scores.append(silhouette_avg)
    
    calinski = calinski_harabasz_score(X_scaled, cluster_labels)
    calinski_scores.append(calinski)
    
    print(f"  K={k}: Silhueta = {silhouette_avg:.4f}, Calinski = {calinski:.0f}")

print(f"K_range: {list(K_range)}")
print(f"silhouette_scores: {[f'{x:.4f}' for x in silhouette_scores]}")
print(f"calinski_scores: {[f'{x:.0f}' for x in calinski_scores]}")

# Gráfico comparativo - AGORA COM DIMENSÕES CORRETAS
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

# Cotovelo (todos os K)
ax1.plot(K_range, inertia, 'o-', color='royalblue', linewidth=2, markersize=8)
ax1.set_title('Método do Cotovelo', fontsize=14, fontweight='bold')
ax1.set_xlabel('Número de Clusters (k)')
ax1.set_ylabel('Inércia')
ax1.grid(True, alpha=0.3)

# Silhueta (todos os K)
ax2.plot(K_range, silhouette_scores, 'o-', color='crimson', linewidth=2, markersize=8)
ax2.set_title('Método da Silhueta', fontsize=14, fontweight='bold')
ax2.set_xlabel('Número de Clusters (k)')
ax2.set_ylabel('Score de Silhueta')
ax2.grid(True, alpha=0.3)

# Destaca o melhor K pela silhueta
best_k_silhouette = list(K_range)[np.argmax(silhouette_scores)]
best_silhouette_score = max(silhouette_scores)
ax2.axvline(x=best_k_silhouette, color='green', linestyle='--', 
            label=f'Melhor K: {best_k_silhouette}\nScore: {best_silhouette_score:.3f}')
ax2.legend()

# Calinski-Harabasz (todos os K)
ax3.plot(K_range, calinski_scores, 'o-', color='green', linewidth=2, markersize=8)
ax3.set_title('Calinski-Harabasz', fontsize=14, fontweight='bold')
ax3.set_xlabel('Número de Clusters (k)')
ax3.set_ylabel('Score Calinski')
ax3.grid(True, alpha=0.3)

# Destaca o melhor K pelo Calinski
best_k_calinski = list(K_range)[np.argmax(calinski_scores)]
best_calinski_score = max(calinski_scores)
ax3.axvline(x=best_k_calinski, color='blue', linestyle='--', 
            label=f'Melhor K: {best_k_calinski}\nScore: {best_calinski_score:.0f}')
ax3.legend()

plt.tight_layout()
plt.show(block=True)

# Análise dos resultados
print(f"\n📊 ANÁLISE DOS RESULTADOS:")
print(f"🎯 Melhor K pela silhueta: {best_k_silhouette} (score: {best_silhouette_score:.4f})")
print(f"🎯 Melhor K pelo Calinski: {best_k_calinski} (score: {best_calinski_score:.0f})")

# Interpretação da qualidade da silhueta
print(f"\n📈 INTERPRETAÇÃO DO SCORE DE SILHUETA ({best_silhouette_score:.4f}):")
if best_silhouette_score > 0.7:
    print("✅ ESTRUTURA FORTE de clusters")
elif best_silhouette_score > 0.5:
    print("✅ Estrutura razoável de clusters")
elif best_silhouette_score > 0.25:
    print("⚠️  Estrutura fraca de clusters")
else:
    print("❌ Não há estrutura significativa de clusters")

# Decisão final do K
if best_k_silhouette == best_k_calinski:
    final_k = best_k_silhouette
    print(f"\n🎯 K FINAL ESCOLHIDO: {final_k} (concordância entre métodos)")
else:
    # Prefere a silhueta por ser mais robusta
    final_k = best_k_silhouette
    print(f"\n🎯 K FINAL ESCOLHIDO: {final_k} (baseado na silhueta)")

# ------------------------------
# Clusterização FINAL com dataset completo
# ------------------------------
print(f"\n🚀 Aplicando K-Means final com k={final_k} no dataset completo...")

# Para o dataset completo, usa MiniBatchKMeans
final_kmeans = MiniBatchKMeans(
    n_clusters=final_k, 
    random_state=42, 
    batch_size=1000,
    n_init=3
)

# Aplica no dataset completo (já limpo)
X_full_scaled = scaler.fit_transform(df_clean)
df_clean['Cluster'] = final_kmeans.fit_predict(X_full_scaled)

print("✅ Clusterização final concluída!")

# ------------------------------
# Visualização com PCA (em amostra)
# ------------------------------
print("📈 Gerando visualizações...")

# Usa amostra para visualização
pca = PCA(n_components=2)
components = pca.fit_transform(X_scaled)
df_viz = df_sampled.copy()
df_viz['PCA1'] = components[:, 0]
df_viz['PCA2'] = components[:, 1]

# Aplica os clusters na amostra de visualização
df_viz['Cluster'] = final_kmeans.predict(X_scaled)

plt.figure(figsize=(10, 8))
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=df_viz, 
                palette='viridis', s=30, alpha=0.6)
plt.title(f'Visualização dos Clusters (K={final_k}) - Amostra de 100k\nSilhueta: {best_silhouette_score:.3f}')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
plt.grid(True, alpha=0.3)
plt.show(block=True)

# ------------------------------
# Análise dos resultados
# ------------------------------
cluster_profile = df_clean.groupby('Cluster').mean()
print(f"\n📊 PERFIL DOS CLUSTERS (dataset completo - {len(df_clean):,} amostras):")

# Tamanho dos clusters
cluster_sizes = df_clean['Cluster'].value_counts().sort_index()
print(f"\n👥 DISTRIBUIÇÃO DOS CLUSTERS:")
for cluster, size in cluster_sizes.items():
    percentage = (size / len(df_clean)) * 100
    print(f"  Cluster {cluster}: {size:,} amostras ({percentage:.1f}%)")

# Análise de diabetes se disponível
if 'Diabetes_012' in df_clean.columns:
    diabetes_stats = df_clean.groupby('Cluster')['Diabetes_012'].mean()
    print(f"\n🎯 PREVALÊNCIA MÉDIA DE DIABETES POR CLUSTER:")
    for cluster, prevalence in diabetes_stats.items():
        print(f"  Cluster {cluster}: {prevalence:.3f} ({prevalence*100:.1f}%)")
    
    # Gráfico de distribuição de diabetes
    plt.figure(figsize=(10, 6))
    diabetes_by_cluster = df_clean.groupby(['Cluster', 'Diabetes_012']).size().unstack()
    diabetes_by_cluster.plot(kind='bar', stacked=True, colormap='Set2')
    plt.title(f'Distribuição de Diabetes por Cluster (K={final_k})')
    plt.xlabel('Cluster')
    plt.ylabel('Número de Amostras')
    plt.legend(title='Diabetes (0=Não, 1=Pré, 2=Sim)')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show(block=True)

# Heatmap do perfil dos clusters
plt.figure(figsize=(12, 8))
sns.heatmap(cluster_profile.T, cmap='coolwarm', annot=True, fmt='.2f', 
            cbar_kws={'label': 'Valor Médio Padronizado'})
plt.title(f'Perfil Médio dos Clusters (K={final_k}) - Dataset Completo')
plt.tight_layout()
plt.show(block=True)

print(f"\n✅ ANÁLISE CONCLUÍDA!")
print(f"📈 Score de silhueta: {best_silhouette_score:.4f}")
print(f"🎯 Número de clusters: {final_k}")
print(f"📊 Total de amostras clusterizadas: {len(df_clean):,}")