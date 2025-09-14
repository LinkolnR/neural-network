import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

def generate_data():
    """
    Gera duas classes de dados 2D a partir de distribuições normais multivariadas.

    Retorna:
        X (np.array): Array com as características dos dados, shape (2000, 2).
        y (np.array): Array com os rótulos dos dados (0 ou 1), shape (2000,).
    """
    print("🔧 Gerando dataset 2D com duas classes...")
    # Define a seed para reprodutibilidade
    np.random.seed(37)

    # Parâmetros para a Classe 0
    mean0 = [1.5, 1.5]
    cov0 = [[0.5, 0], [0, 0.5]]
    class0_samples = 1000

    # Parâmetros para a Classe 1
    mean1 = [5, 5]
    cov1 = [[0.5, 0], [0, 0.5]]
    class1_samples = 1000

    # Gerar os dados
    class0_data = np.random.multivariate_normal(mean0, cov0, class0_samples)
    class1_data = np.random.multivariate_normal(mean1, cov1, class1_samples)

    # Combinar os dados em um único dataset
    X = np.vstack((class0_data, class1_data))
    
    # Criar os rótulos (labels)
    y = np.hstack((np.zeros(class0_samples), np.ones(class1_samples)))
    
    print(f"✅ Dataset 2D gerado com sucesso!")
    print(f"📊 Forma dos dados: {X.shape}")
    print(f"🏷️  Classes: {np.unique(y)}")
    print(f"📈 Amostras por classe: {np.bincount(y.astype(int))}")

    return X, y

def plot_and_save_data(X, y):
    """
    Plota os dados 2D, salva em PNG e exibe.

    Parâmetros:
        X (np.array): Array com as características dos dados.
        y (np.array): Array com os rótulos dos dados.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(X[y == 0, 0], X[y == 0, 1], color='red', label='Classe 0')
    ax.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', label='Classe 1')
    ax.set_title('Visualização dos Dados Gerados')
    ax.set_xlabel('Característica 1')
    ax.set_ylabel('Característica 2')
    ax.legend()
    ax.grid(True)
    
    # Salvar gráfico
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, 'dataset_2d_visualization.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Gráfico salvo como '{os.path.basename(output_path)}'")
    
    plt.show()

def save_dataset_2d(X, y):
    """
    Salva o dataset 2D em arquivo CSV.
    """
    df = pd.DataFrame(X, columns=['feature_1', 'feature_2'])
    df['class'] = y.astype(int)
    df['class_name'] = df['class'].map({0: 'Classe 0', 1: 'Classe 1'})
    
    # Salvar CSV
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, 'dataset_2d.csv')
    df.to_csv(output_path, index=False)
    print(f"💾 Dataset 2D salvo como '{os.path.basename(output_path)}'")
    return df

def main():
    """
    Função principal para gerar, visualizar e salvar os dados.
    """
    print("🚀 Script de Geração de Dados para Perceptron")
    print("="*60)
    
    X, y = generate_data()
    
    print("\n💾 Salvando dataset...")
    save_dataset_2d(X, y)
    
    print("\n🎨 Visualizando e salvando gráfico...")
    plot_and_save_data(X, y)
    
    print("\n✅ Processo concluído.")

if __name__ == "__main__":
    main()
