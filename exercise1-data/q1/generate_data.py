import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import math
import random

def box_muller_transform():
    """
    Algoritmo de Box-Muller para gerar números aleatórios de distribuição normal.
    
    Este é o método clássico para gerar números de distribuição normal padrão
    a partir de números uniformemente distribuídos.
    
    Retorna:
        tuple: (z1, z2) - dois números de distribuição normal padrão
    """
    # Gerar dois números uniformemente distribuídos entre 0 e 1
    u1 = random.random()
    u2 = random.random()
    
    # Evitar log(0) que causaria erro
    if u1 == 0:
        u1 = 1e-10
    
    # Aplicar transformação de Box-Muller
    z1 = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
    z2 = math.sqrt(-2 * math.log(u1)) * math.sin(2 * math.pi * u2)
    
    return z1, z2

def cholesky_decomposition(matrix):
    """
    Decomposição de Cholesky para matriz 2x2.
    
    A decomposição de Cholesky decompõe uma matriz simétrica positiva definida
    em L * L^T, onde L é uma matriz triangular inferior.
    
    Args:
        matrix: Lista de listas representando uma matriz 2x2
        
    Returns:
        list: Matriz L (triangular inferior)
    """
    # Para matriz 2x2: [[a, b], [b, c]]
    a, b, c = matrix[0][0], matrix[0][1], matrix[1][1]
    
    # Calcular elementos da matriz L
    l11 = math.sqrt(a)
    l21 = b / l11
    l22 = math.sqrt(c - l21**2)
    
    return [[l11, 0], [l21, l22]]

def matrix_vector_multiply(matrix, vector):
    """
    Multiplica uma matriz 2x2 por um vetor 2x1.
    
    Args:
        matrix: Lista de listas representando matriz 2x2
        vector: Lista representando vetor 2x1
        
    Returns:
        list: Resultado da multiplicação
    """
    result = [0, 0]
    for i in range(2):
        for j in range(2):
            result[i] += matrix[i][j] * vector[j]
    return result

def vector_add(v1, v2):
    """
    Soma dois vetores elemento por elemento.
    
    Args:
        v1, v2: Listas representando vetores
        
    Returns:
        list: Soma dos vetores
    """
    return [v1[i] + v2[i] for i in range(len(v1))]

def multivariate_normal_manual(mean, cov_matrix, size):
    """
    Implementação manual de distribuição normal multivariada.
    
    Este é o equivalente ao np.random.multivariate_normal() mas implementado
    do zero usando apenas Python puro e matemática.
    
    Algoritmo:
    1. Gerar números de distribuição normal padrão usando Box-Muller
    2. Aplicar decomposição de Cholesky na matriz de covariância
    3. Transformar os números normais padrão usando a matriz L
    4. Adicionar a média para obter a distribuição final
    
    Args:
        mean: Lista com a média [x_mean, y_mean]
        cov_matrix: Lista de listas com a matriz de covariância 2x2
        size: Número de amostras a gerar
        
    Returns:
        list: Lista de listas com as amostras geradas
    """
    print(f"  🔧 Gerando {size} amostras manualmente...")
    print(f"  📊 Média: {mean}")
    print(f"  📈 Matriz de covariância: {cov_matrix}")
    
    # Passo 1: Decomposição de Cholesky
    L = cholesky_decomposition(cov_matrix)
    print(f"  🧮 Matriz L (Cholesky): {L}")
    
    samples = []
    
    for i in range(size):
        # Passo 2: Gerar números de distribuição normal padrão
        z1, z2 = box_muller_transform()
        z = [z1, z2]
        
        # Passo 3: Aplicar transformação L * z
        transformed = matrix_vector_multiply(L, z)
        
        # Passo 4: Adicionar a média
        sample = vector_add(transformed, mean)
        
        samples.append(sample)
    
    print(f"  ✅ {len(samples)} amostras geradas com sucesso!")
    return samples

def generate_synthetic_dataset_manual():
    """
    Versão manual do gerador de dataset sintético (sem NumPy).
    
    Esta função faz exatamente a mesma coisa que generate_synthetic_dataset(),
    mas implementa toda a matemática do zero sem usar NumPy.
    """
    
    # Parâmetros das classes conforme especificado
    class_params = {
        0: {'mean': [2, 3], 'std': [0.8, 2.5]},
        1: {'mean': [5, 6], 'std': [1.2, 1.9]},
        2: {'mean': [8, 1], 'std': [0.9, 0.9]},
        3: {'mean': [15, 4], 'std': [0.5, 2.0]}
    }
    
    # Listas para armazenar os dados
    all_data = []
    all_labels = []
    
    print("=== GERADOR MANUAL (SEM NUMPY) ===\n")
    
    # Gerar dados para cada classe
    for class_id, params in class_params.items():
        print(f"🎯 Processando Classe {class_id}:")
        
        # Criar matriz de covariância diagonal
        std_x, std_y = params['std']
        cov_matrix = [
            [std_x**2, 0],      # Variância em x, covariância xy
            [0, std_y**2]       # Covariância yx, variância em y
        ]
        
        # Gerar amostras manualmente
        samples = multivariate_normal_manual(
            mean=params['mean'],
            cov_matrix=cov_matrix,
            size=100
        )
        
        # Calcular estatísticas reais
        x_values = [s[0] for s in samples]
        y_values = [s[1] for s in samples]
        
        real_mean_x = sum(x_values) / len(x_values)
        real_mean_y = sum(y_values) / len(y_values)
        real_std_x = math.sqrt(sum((x - real_mean_x)**2 for x in x_values) / len(x_values))
        real_std_y = math.sqrt(sum((y - real_mean_y)**2 for y in y_values) / len(y_values))
        
        print(f"  📋 Parâmetros esperados: média={params['mean']}, std={params['std']}")
        print(f"  📊 Estatísticas reais: média=[{real_mean_x:.2f}, {real_mean_y:.2f}], std=[{real_std_x:.2f}, {real_std_y:.2f}]")
        print()
        
        # Adicionar aos dados totais
        all_data.extend(samples)
        all_labels.extend([class_id] * 100)
    
    print(f"🎉 Dataset manual gerado com sucesso!")
    print(f"📊 Total de amostras: {len(all_data)}")
    print(f"🏷️  Classes únicas: {sorted(set(all_labels))}")
    print(f"📈 Amostras por classe: {[all_labels.count(i) for i in range(4)]}")
    
    return all_data, all_labels

def generate_synthetic_dataset():
    """
    Gera um dataset sintético com 4 classes usando distribuições Gaussianas.
    
    Especificações:
    - 400 amostras no total
    - 4 classes (100 amostras por classe)
    - Cada classe gerada com distribuição Gaussiana
    """
    
    # Parâmetros das classes conforme especificado
    class_params = {
        0: {'mean': [2, 3], 'std': [0.8, 2.5]},
        1: {'mean': [5, 6], 'std': [1.2, 1.9]},
        2: {'mean': [8, 1], 'std': [0.9, 0.9]},
        3: {'mean': [15, 4], 'std': [0.5, 2.0]}
    }
    
    # Listas para armazenar os dados
    all_data = []
    all_labels = []
    
    # Gerar dados para cada classe
    for class_id, params in class_params.items():
        # Gerar 100 amostras para esta classe
        samples = np.random.multivariate_normal(
            mean=params['mean'],
            cov=np.diag(np.square(params['std'])),  # Covariância diagonal
            size=100
        )
        
        # Adicionar aos dados totais
        all_data.extend(samples)
        all_labels.extend([class_id] * 100)
        
        print(f"Classe {class_id}: {len(samples)} amostras geradas")
        print(f"  Média: {params['mean']}")
        print(f"  Desvio padrão: {params['std']}")
        print(f"  Média real: {np.mean(samples, axis=0)}")
        print(f"  Desvio padrão real: {np.std(samples, axis=0)}")
        print()
    
    # Converter para arrays numpy
    X = np.array(all_data)
    y = np.array(all_labels)
    
    print(f"Dataset gerado com sucesso!")
    print(f"Total de amostras: {len(X)}")
    print(f"Forma dos dados: {X.shape}")
    print(f"Classes únicas: {np.unique(y)}")
    print(f"Amostras por classe: {np.bincount(y)}")
    
    return X, y

def visualize_dataset(X, y, save_plot=True):
    """
    Visualiza o dataset gerado.
    
    Args:
        X: Array com as features (coordenadas x, y)
        y: Array com os labels das classes
        save_plot: Se True, salva o gráfico como arquivo
    """
    
    # Cores para cada classe
    colors = ['red', 'blue', 'green', 'orange']
    class_names = ['Classe 0', 'Classe 1', 'Classe 2', 'Classe 3']
    
    # Criar o gráfico
    plt.figure(figsize=(12, 8))
    
    for class_id in range(4):
        # Filtrar dados da classe
        mask = y == class_id
        class_data = X[mask]
        
        # Plotar pontos da classe
        plt.scatter(class_data[:, 0], class_data[:, 1], 
                   c=colors[class_id], label=class_names[class_id], 
                   alpha=0.7, s=50)
    
    plt.xlabel('Coordenada X')
    plt.ylabel('Coordenada Y')
    plt.title('Dataset Sintético - 4 Classes com Distribuição Gaussiana')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_plot:
        # Obter o diretório atual do script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(script_dir, 'synthetic_dataset.png')
        
        try:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Gráfico salvo como '{output_path}'")
        except Exception as e:
            print(f"Erro ao salvar gráfico: {e}")
            # Tentar salvar no diretório atual
            plt.savefig('synthetic_dataset.png', dpi=300, bbox_inches='tight')
            print("Gráfico salvo como 'synthetic_dataset.png' no diretório atual")
    
    plt.show()

def save_dataset(X, y, filename=None):
    """
    Salva o dataset em um arquivo CSV.
    
    Args:
        X: Array com as features
        y: Array com os labels
        filename: Nome do arquivo para salvar (opcional)
    """
    
    if filename is None:
        # Usar o diretório do script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        filename = os.path.join(script_dir, 'synthetic_dataset.csv')
    
    # Criar DataFrame
    df = pd.DataFrame({
        'x': X[:, 0],
        'y': X[:, 1],
        'class': y
    })
    
    # Salvar CSV
    try:
        df.to_csv(filename, index=False)
        print(f"Dataset salvo como '{filename}'")
    except Exception as e:
        print(f"Erro ao salvar CSV: {e}")
        # Tentar salvar no diretório atual
        df.to_csv('synthetic_dataset.csv', index=False)
        print("Dataset salvo como 'synthetic_dataset.csv' no diretório atual")
    
    return df

def main():
    """
    Função principal que executa todo o pipeline de geração de dados.
    """
    
    print("=== Gerador de Dataset Sintético ===\n")
    
    # Definir seed para reprodutibilidade
    np.random.seed(37)
    random.seed(37)

    # Escolher qual versão executar
    print("Escolha a versão:")
    print("1. Versão com NumPy (rápida)")
    print("2. Versão manual (sem NumPy, educativa)")
    print("3. Ambas (comparação)")
    
    choice = input("Digite sua escolha (1, 2 ou 3): ").strip()
    
    if choice == "1":
        # Gerar dataset com NumPy
        X, y = generate_synthetic_dataset()
        data_type = "NumPy"
    elif choice == "2":
        # Gerar dataset manual
        all_data, all_labels = generate_synthetic_dataset_manual()
        X = np.array(all_data)
        y = np.array(all_labels)
        data_type = "Manual"
    elif choice == "3":
        # Comparar ambas as versões
        print("\n" + "="*50)
        print("COMPARAÇÃO: NUMPY vs MANUAL")
        print("="*50)
        
        # Versão NumPy
        print("\n🔵 VERSÃO NUMPY:")
        X_numpy, y_numpy = generate_synthetic_dataset()
        
        print("\n🔴 VERSÃO MANUAL:")
        all_data_manual, all_labels_manual = generate_synthetic_dataset_manual()
        X_manual = np.array(all_data_manual)
        y_manual = np.array(all_labels_manual)
        
        # Usar a versão manual para visualização
        X, y = X_manual, y_manual
        data_type = "Comparação (Manual)"
    else:
        print("Escolha inválida. Usando versão NumPy por padrão.")
        X, y = generate_synthetic_dataset()
        data_type = "NumPy"
    
    # Visualizar dataset
    print(f"\n=== Visualizando Dataset ({data_type}) ===")
    visualize_dataset(X, y)
    
    # Salvar dataset
    print(f"\n=== Salvando Dataset ({data_type}) ===")
    df = save_dataset(X, y)
    
    # Mostrar estatísticas finais
    print(f"\n=== Estatísticas Finais ({data_type}) ===")
    print(df.describe())
    print(f"\nDistribuição das classes:")
    print(df['class'].value_counts().sort_index())
    
    return X, y, df

if __name__ == "__main__":
    X, y, df = main()