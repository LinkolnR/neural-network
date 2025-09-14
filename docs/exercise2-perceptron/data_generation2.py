import numpy as np
import matplotlib.pyplot as plt
from io import StringIO

def generate_data():
    """
    Gera duas classes de dados 2D com sobreposição a partir de 
    distribuições normais multivariadas.

    Retorna:
        X (np.array): Array com as características dos dados, shape (2000, 2).
        y (np.array): Array com os rótulos dos dados (0 ou 1), shape (2000,).
    """
    # Define a seed para reprodutibilidade
    np.random.seed(37)
    
    # Parâmetros para a Classe 0
    mean0 = [3, 3]
    cov0 = [[1.5, 0], [0, 1.5]]
    class0_samples = 1000

    # Parâmetros para a Classe 1
    mean1 = [4, 4]
    cov1 = [[1.5, 0], [0, 1.5]]
    class1_samples = 1000

    # Gerar os dados
    class0_data = np.random.multivariate_normal(mean0, cov0, class0_samples)
    class1_data = np.random.multivariate_normal(mean1, cov1, class1_samples)

    # Combinar os dados em um único dataset
    X = np.vstack((class0_data, class1_data))
    
    # Criar os rótulos (labels)
    y = np.hstack((np.zeros(class0_samples), np.ones(class1_samples)))

    return X, y

def plot_data(X, y):
    """
    Plota os dados 2D, colorindo os pontos por classe.

    Parâmetros:
        X (np.array): Array com as características dos dados.
        y (np.array): Array com os rótulos dos dados.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red', label='Classe 0')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', label='Classe 1')
    plt.title('Visualização dos Dados com Sobreposição')
    plt.xlabel('Característica 1')
    plt.ylabel('Característica 2')
    plt.legend()
    plt.grid(True)
    # plt.show() # Removido para não abrir a janela do gráfico

# Gerar os dados
X, y = generate_data()

# Criar o plot para visualizar a sobreposição
plot_data(X, y)

# Salvar o plot em um buffer SVG e imprimir para o mkdocs
buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True, bbox_inches='tight')
print(buffer.getvalue())
# plt.close() # Fecha a figura para liberar memória
