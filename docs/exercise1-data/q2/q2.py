import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os

# Configurar seed para reprodutibilidade
np.random.seed(42)

def generate_5d_dataset():
    """
    Gera um dataset 5D com duas classes usando distribuições normais multivariadas.
    
    Especificações:
    - 500 amostras para Classe A
    - 500 amostras para Classe B
    - 5 dimensões por amostra
    - Total: 1000 amostras
    """
    
    print("🔧 Gerando dataset 5D com duas classes...")
    
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
    
    # Gerar amostras para Classe A
    samples_A = np.random.multivariate_normal(mean_A, cov_A, 500)
    labels_A = np.zeros(500, dtype=int)  # Classe A = 0
    
    # Gerar amostras para Classe B
    samples_B = np.random.multivariate_normal(mean_B, cov_B, 500)
    labels_B = np.ones(500, dtype=int)   # Classe B = 1
    
    # Combinar dados
    X = np.vstack([samples_A, samples_B])
    y = np.hstack([labels_A, labels_B])
    
    print(f"✅ Dataset 5D gerado com sucesso!")
    print(f"📊 Forma dos dados: {X.shape}")
    print(f"🏷️  Classes: {np.unique(y)}")
    print(f"📈 Amostras por classe: {np.bincount(y)}")
    
    # Mostrar estatísticas das classes
    print(f"\n📋 Estatísticas Classe A:")
    print(f"   Média real: {np.mean(samples_A, axis=0)}")
    print(f"   Desvio padrão: {np.std(samples_A, axis=0)}")
    
    print(f"\n📋 Estatísticas Classe B:")
    print(f"   Média real: {np.mean(samples_B, axis=0)}")
    print(f"   Desvio padrão: {np.std(samples_B, axis=0)}")
    
    return X, y, samples_A, samples_B

def apply_pca_analysis(X, y):
    """
    Aplica PCA para reduzir dimensionalidade de 5D para 2D e analisa a variância.
    """
    
    print("\n" + "="*60)
    print("🔍 ANÁLISE DE COMPONENTES PRINCIPAIS (PCA)")
    print("="*60)
    
    # Aplicar PCA
    pca = PCA()
    X_pca = pca.fit_transform(X)
    
    # Mostrar variância explicada
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    print("📊 Variância explicada por componente:")
    for i, (var, cum_var) in enumerate(zip(explained_variance_ratio, cumulative_variance)):
        print(f"   PC{i+1}: {var:.3f} ({var*100:.1f}%) - Cumulativa: {cum_var:.3f} ({cum_var*100:.1f}%)")
    
    # Reduzir para 2D
    pca_2d = PCA(n_components=2)
    X_2d = pca_2d.fit_transform(X)
    
    print(f"\n🎯 Redução para 2D:")
    print(f"   Variância explicada pelos 2 primeiros PCs: {pca_2d.explained_variance_ratio_.sum():.3f} ({pca_2d.explained_variance_ratio_.sum()*100:.1f}%)")
    print(f"   PC1: {pca_2d.explained_variance_ratio_[0]:.3f} ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)")
    print(f"   PC2: {pca_2d.explained_variance_ratio_[1]:.3f} ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)")
    
    return X_2d, pca_2d, explained_variance_ratio

def visualize_2d_projection(X_2d, y):
    """
    Cria visualização 2D do dataset após redução dimensional.
    """
    
    print("\n" + "="*60)
    print("🎨 VISUALIZAÇÃO 2D DO DATASET")
    print("="*60)
    
    # Configurar cores e nomes
    colors = ['red', 'blue']
    class_names = ['Classe A', 'Classe B']
    
    # Criar gráfico
    plt.figure(figsize=(12, 8))
    
    for class_id in range(2):
        mask = y == class_id
        class_data = X_2d[mask]
        
        plt.scatter(class_data[:, 0], class_data[:, 1], 
                   c=colors[class_id], label=class_names[class_id], 
                   alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
    
    plt.xlabel('Primeira Componente Principal (PC1)')
    plt.ylabel('Segunda Componente Principal (PC2)')
    plt.title('Dataset 5D Projetado em 2D usando PCA\n(1000 amostras, 2 classes)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Salvar gráfico
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, 'dataset_5d_2d_projection.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Gráfico salvo como 'dataset_5d_2d_projection.png'")
    
    plt.show()
    
    return X_2d

def analyze_linear_separability(X, y, X_2d):
    """
    Analisa a separabilidade linear dos dados em 5D e 2D.
    """
    
    print("\n" + "="*60)
    print("🔍 ANÁLISE DE SEPARABILIDADE LINEAR")
    print("="*60)
    
    # Testar separabilidade em 5D
    print("📊 Testando separabilidade em 5D:")
    lr_5d = LogisticRegression(max_iter=1000, random_state=42)
    lr_5d.fit(X, y)
    score_5d = lr_5d.score(X, y)
    print(f"   Acurácia Regressão Logística (5D): {score_5d:.3f}")
    
    # Testar separabilidade em 2D
    print("\n📊 Testando separabilidade em 2D (após PCA):")
    lr_2d = LogisticRegression(max_iter=1000, random_state=42)
    lr_2d.fit(X_2d, y)
    score_2d = lr_2d.score(X_2d, y)
    print(f"   Acurácia Regressão Logística (2D): {score_2d:.3f}")
    
    # Análise da separabilidade
    print(f"\n🎯 Análise da Separabilidade:")
    if score_5d > 0.95:
        print("   ✅ Dados 5D são LINEARMENTE SEPARÁVEIS")
    elif score_5d > 0.8:
        print("   ⚠️  Dados 5D são PARCIALMENTE LINEARMENTE SEPARÁVEIS")
    else:
        print("   ❌ Dados 5D NÃO são linearmente separáveis")
    
    if score_2d > 0.95:
        print("   ✅ Projeção 2D é LINEARMENTE SEPARÁVEL")
    elif score_2d > 0.8:
        print("   ⚠️  Projeção 2D é PARCIALMENTE LINEARMENTE SEPARÁVEL")
    else:
        print("   ❌ Projeção 2D NÃO é linearmente separável")
    
    return score_5d, score_2d

def test_neural_networks(X, y):
    """
    Testa diferentes arquiteturas de redes neurais.
    """
    
    print("\n" + "="*60)
    print("🧠 TESTE DE REDES NEURAIS")
    print("="*60)
    
    # Dividir dados
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Configurações de redes neurais
    nn_configs = [
        (MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=42), "NN Simples (10)"),
        (MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=42), "NN Média (50)"),
        (MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42), "NN Grande (100)"),
        (MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42), "NN Profunda (100-50)"),
        (MLPClassifier(hidden_layer_sizes=(200, 100, 50), max_iter=1000, random_state=42), "NN Muito Profunda (200-100-50)")
    ]
    
    print("🔬 Testando diferentes arquiteturas:")
    results = []
    
    for model, name in nn_configs:
        model.fit(X_train, y_train)
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        results.append((name, train_score, test_score))
        print(f"   {name}:")
        print(f"      Treino: {train_score:.3f}")
        print(f"      Teste:  {test_score:.3f}")
    
    return results

def create_decision_boundary_plot_2d(X_2d, y):
    """
    Cria gráfico com fronteiras de decisão em 2D.
    """
    
    print("\n" + "="*60)
    print("🎨 FRONTEIRAS DE DECISÃO EM 2D")
    print("="*60)
    
    # Configurar a grade
    h = 0.1
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Modelos para testar
    models = [
        LogisticRegression(max_iter=1000, random_state=42),
        MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=42),
        MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    ]
    
    model_names = [
        "Regressão Logística (Linear)",
        "Rede Neural Simples (50)",
        "Rede Neural Profunda (100-50)"
    ]
    
    # Criar subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = ['red', 'blue']
    class_names = ['Classe A', 'Classe B']
    
    for idx, (model, name) in enumerate(zip(models, model_names)):
        ax = axes[idx]
        
        # Treinar modelo
        model.fit(X_2d, y)
        
        # Fazer predições na grade
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Plotar regiões de decisão
        ax.contourf(xx, yy, Z, alpha=0.3, colors=colors)
        
        # Plotar pontos
        for class_id in range(2):
            mask = y == class_id
            class_data = X_2d[mask]
            ax.scatter(class_data[:, 0], class_data[:, 1], 
                      c=colors[class_id], label=class_names[class_id], 
                      alpha=0.7, s=30, edgecolors='black', linewidth=0.5)
        
        # Configurar gráfico
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title(f'{name}\nAcurácia: {model.score(X_2d, y):.3f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Salvar gráfico
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, 'decision_boundaries_2d.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Gráfico de fronteiras salvo como 'decision_boundaries_2d.png'")
    
    plt.show()

def save_dataset_5d(X, y):
    """
    Salva o dataset 5D em arquivo CSV.
    """
    
    # Criar DataFrame
    df = pd.DataFrame(X, columns=[f'feature_{i+1}' for i in range(5)])
    df['class'] = y
    df['class_name'] = df['class'].map({0: 'A', 1: 'B'})
    
    # Salvar CSV
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, 'dataset_5d.csv')
    df.to_csv(output_path, index=False)
    print(f"💾 Dataset 5D salvo como 'dataset_5d.csv'")
    
    return df


def main():
    """
    Função principal que executa todo o pipeline do Exercício 2.
    """
    
    print("🚀 EXERCÍCIO 2: NÃO-LINEARIDADE EM DIMENSÕES ALTAS")
    print("="*60)
    
    # Gerar dataset 5D usando NumPy
    print("\n🔵 Gerando dataset 5D com NumPy...")
    X, y, samples_A, samples_B = generate_5d_dataset()
    X_2d, pca_2d, explained_variance = apply_pca_analysis(X, y)
    
    # Visualizar projeção 2D
    print("\n🎨 Visualizando projeção 2D...")
    visualize_2d_projection(X_2d, y)
    
    # Analisar separabilidade linear
    print("\n🔍 Analisando separabilidade linear...")
    score_5d, score_2d = analyze_linear_separability(X, y, X_2d)
    
    # Testar redes neurais
    print("\n🧠 Testando redes neurais...")
    nn_results = test_neural_networks(X, y)
    
    # Criar gráficos de fronteiras de decisão
    print("\n🎨 Criando fronteiras de decisão...")
    create_decision_boundary_plot_2d(X_2d, y)
    
    # Salvar dataset
    print("\n💾 Salvando dataset...")
    df = save_dataset_5d(X, y)
    
    # Resumo final
    print("\n" + "="*60)
    print("📋 RESUMO DO EXERCÍCIO 2")
    print("="*60)
    
    print("🎯 Dataset gerado:")
    print("   • 1000 amostras (500 por classe)")
    print("   • 5 dimensões por amostra")
    print("   • 2 classes (A e B)")
    
    print(f"\n📊 Análise PCA:")
    print(f"   • Variância explicada pelos 2 primeiros PCs: {pca_2d.explained_variance_ratio_.sum():.1%}")
    print(f"   • PC1: {pca_2d.explained_variance_ratio_[0]:.1%}")
    print(f"   • PC2: {pca_2d.explained_variance_ratio_[1]:.1%}")
    
    print(f"\n🔍 Separabilidade Linear:")
    print(f"   • 5D: {score_5d:.3f}")
    print(f"   • 2D: {score_2d:.3f}")
    
    print(f"\n🧠 Melhor Rede Neural:")
    best_nn = max(nn_results, key=lambda x: x[2])
    print(f"   • {best_nn[0]}: {best_nn[2]:.3f}")
    
    print(f"\n📁 Arquivos gerados:")
    print(f"   • dataset_5d.csv")
    print(f"   • dataset_5d_2d_projection.png")
    print(f"   • decision_boundaries_2d.png")
    
    return X, y, X_2d, df

if __name__ == "__main__":
    X, y, X_2d, df = main()
