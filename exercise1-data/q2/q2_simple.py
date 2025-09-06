from io import StringIO
import numpy as np
import matplotlib.pyplot as plt

# Definir seed para reprodutibilidade
np.random.seed(37)

# =============================================================================
# IMPLEMENTAÇÃO MANUAL DO PCA
# =============================================================================

def manual_pca(X, n_components=2):
    """
    Implementação manual do PCA (Principal Component Analysis).
    
    Parâmetros:
    -----------
    X : array-like, shape (n_samples, n_features)
        Dados de entrada
    n_components : int
        Número de componentes principais a manter
        
    Retorna:
    --------
    X_transformed : array-like, shape (n_samples, n_components)
        Dados transformados
    explained_variance_ratio : array-like, shape (n_components,)
        Razão da variância explicada por cada componente
    """
    
    # Passo 1: Centralizar os dados (subtrair a média)
    X_centered = X - np.mean(X, axis=0)
    
    # Passo 2: Calcular a matriz de covariância
    n_samples = X.shape[0]
    cov_matrix = np.dot(X_centered.T, X_centered) / (n_samples - 1)
    
    # Passo 3: Calcular autovalores e autovetores
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Passo 4: Ordenar por autovalores decrescentes
    # np.linalg.eigh retorna em ordem crescente, então invertemos
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    
    # Passo 5: Selecionar os n_components primeiros
    selected_eigenvectors = eigenvectors[:, :n_components]
    selected_eigenvalues = eigenvalues[:n_components]
    
    # Passo 6: Projetar os dados nos componentes principais
    X_transformed = np.dot(X_centered, selected_eigenvectors)
    
    # Calcular variância explicada
    total_variance = np.sum(eigenvalues)
    explained_variance_ratio = selected_eigenvalues / total_variance
    
    return X_transformed, explained_variance_ratio, eigenvalues, eigenvectors

def print_pca_details(eigenvalues, eigenvectors, explained_variance_ratio):
    """
    Imprime detalhes sobre o PCA calculado.
    """
    print("\n" + "="*50)
    print("🔬 DETALHES DA IMPLEMENTAÇÃO MANUAL DO PCA")
    print("="*50)
    
    print(f"\n📊 Autovalores (variâncias dos componentes):")
    for i, eigenval in enumerate(eigenvalues):
        print(f"   Componente {i+1}: {eigenval:.4f}")
    
    print(f"\n📈 Variância explicada por componente:")
    for i, ratio in enumerate(explained_variance_ratio):
        print(f"   PC{i+1}: {ratio:.3f} ({ratio*100:.1f}%)")
    
    print(f"\n🎯 Variância total explicada pelos 2 primeiros PCs: {np.sum(explained_variance_ratio):.3f} ({np.sum(explained_variance_ratio)*100:.1f}%)")
    
    print(f"\n🔍 Direções dos componentes principais:")
    print(f"   PC1 (direção de máxima variância): {eigenvectors[:, 0]}")
    print(f"   PC2 (segunda direção): {eigenvectors[:, 1]}")

def generate_5d_dataset_simple():
    """
    Gera dataset 5D com duas classes usando distribuições normais multivariadas.
    Versão simplificada para visualização no mkdocs.
    """
    
    # Parâmetros para Classe A
    mean_A = np.array([0, 0, 0, 0, 0])
    cov_A = np.array([
        [1.0, 0.8, 0.1, 0.0, 0.0],
        [0.8, 1.0, 0.3, 0.0, 0.0],
        [0.1, 0.3, 1.0, 0.5, 0.0],
        [0.0, 0.0, 0.5, 1.0, 0.2],
        [0.0, 0.0, 0.0, 0.2, 1.0]
    ])
    
    # Parâmetros para Classe B
    mean_B = np.array([1.5, 1.5, 1.5, 1.5, 1.5])
    cov_B = np.array([
        [1.5, -0.7, 0.2, 0.0, 0.0],
        [-0.7, 1.5, 0.4, 0.0, 0.0],
        [0.2, 0.4, 1.5, 0.6, 0.0],
        [0.0, 0.0, 0.6, 1.5, 0.3],
        [0.0, 0.0, 0.0, 0.3, 1.5]
    ])
    
    # Gerar amostras para cada classe
    samples_A = np.random.multivariate_normal(mean_A, cov_A, 500)
    samples_B = np.random.multivariate_normal(mean_B, cov_B, 500)
    
    # Combinar dados
    X = np.vstack([samples_A, samples_B])
    y = np.hstack([np.zeros(500, dtype=int), np.ones(500, dtype=int)])
    
    return X, y

def apply_pca_and_visualize():
    """
    Aplica PCA manual para reduzir a 2D e cria visualização para mkdocs.
    """
    
    # Gerar dataset 5D
    X, y = generate_5d_dataset_simple()
    
    # print(f"📊 Dataset original: {X.shape[0]} amostras, {X.shape[1]} dimensões")
    
    # Aplicar PCA manual para reduzir a 2D
    X_2d, explained_variance_ratio, eigenvalues, eigenvectors = manual_pca(X, n_components=2)
    
    # Imprimir detalhes do PCA
    # print_pca_details(eigenvalues, eigenvectors, explained_variance_ratio)
    
    # Configurar cores e nomes das classes
    colors = ['red', 'blue']
    class_names = ['Classe A', 'Classe B']
    
    # Criar figura
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Plotar dados de cada classe
    for class_id in range(2):
        mask = y == class_id
        class_data = X_2d[mask]
        ax.scatter(class_data[:, 0], class_data[:, 1], 
                  c=colors[class_id], label=class_names[class_id], 
                  alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
    
    # Configurar gráfico
    ax.set_xlabel('Primeira Componente Principal (PC1)')
    ax.set_ylabel('Segunda Componente Principal (PC2)')
    ax.set_title('Dataset 5D Projetado em 2D usando PCA \n(1000 amostras, 2 classes)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # # Adicionar informações sobre variância explicada
    # variance_text = f"Variância explicada: PC1={explained_variance_ratio[0]:.1%}, PC2={explained_variance_ratio[1]:.1%}"
    # ax.text(0.02, 0.98, variance_text, transform=ax.transAxes, 
    #         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # # Adicionar texto sobre implementação manual
    # manual_text = "🔧 Implementação Manual do PCA"
    # ax.text(0.02, 0.02, manual_text, transform=ax.transAxes, 
    #         verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    return fig

# Gerar e exibir o gráfico em formato SVG para mkdocs
fig = apply_pca_and_visualize()
buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True, bbox_inches='tight')
print(buffer.getvalue())
