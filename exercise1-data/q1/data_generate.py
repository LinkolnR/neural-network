from io import StringIO
import numpy as np
import matplotlib.pyplot as plt

def generate_synthetic_data():
    """
    Gera dados sintéticos com 4 classes usando distribuições gaussianas.
    
    Returns:
        tuple: (X, y) - dados das features e labels das classes
    """
    # Definir seed para reprodutibilidade
    np.random.seed(37)
    
    # Parâmetros das classes (igual ao generate_data.py)
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
    
    # Converter para arrays numpy
    X = np.array(all_data)
    y = np.array(all_labels)
    
    return X, y


X, y = generate_synthetic_data()

# Parâmetros das classes para referência
class_params = {
    0: {'mean': [2, 3], 'std': [0.8, 2.5]},
    1: {'mean': [5, 6], 'std': [1.2, 1.9]},
    2: {'mean': [8, 1], 'std': [0.9, 0.9]},
    3: {'mean': [15, 4], 'std': [0.5, 2.0]}
}

# Mostrar estatísticas por classe
for class_id in range(4):
    mask = y == class_id
    class_data = X[mask]

# Configurar a figura
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# Cores para cada classe
colors = ['red', 'blue', 'green', 'orange']
class_names = ['Classe 0', 'Classe 1', 'Classe 2', 'Classe 3']

# Plotar dados de cada classe
for class_id in range(4):
    # Filtrar dados da classe
    mask = y == class_id
    class_data = X[mask]
    
    # Plotar pontos da classe
    ax.scatter(class_data[:, 0], class_data[:, 1], 
                c=colors[class_id], label=class_names[class_id], 
                alpha=0.7, s=50)

# uma linha em x = 12 , que vai de y = -3 até y = 12
ax.plot([12, 12], [-3, 12], 'k-', linewidth=2, alpha=0.8 , label='Fronteira de Decisão')
# separação entre a classe 2 e as classes 1 e 0
ax.plot([3, 12], [-3, 12], 'k-', linewidth=2, alpha=0.8)
# Separação da classe 0 e 1 
ax.plot([6,0], [-3,12], 'k-', linewidth=2, alpha=0.8)

# Configurar aparência
ax.set_xlabel('Coordenada X', fontsize=12)
ax.set_ylabel('Coordenada Y', fontsize=12)
ax.set_title('Dataset Sintético - 4 Classes com Distribuição Gaussiana (NumPy)', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

# Ajustar layout
plt.tight_layout()

# Display the plot
buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())
