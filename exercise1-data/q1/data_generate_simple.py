from io import StringIO
import numpy as np
import matplotlib.pyplot as plt

# Definir seed para reprodutibilidade
np.random.seed(37)

# Parâmetros das classes (igual ao generate_data.py)
class_params = {
    0: {'mean': [2, 3], 'std': [0.8, 2.5]},
    1: {'mean': [5, 6], 'std': [1.2, 1.9]},
    2: {'mean': [8, 1], 'std': [0.9, 0.9]},
    3: {'mean': [15, 4], 'std': [0.5, 2.0]}
}

# Gerar dados das 4 classes
all_data = []
all_labels = []

for class_id, params in class_params.items():
    samples = np.random.multivariate_normal(
        mean=params['mean'],
        cov=np.diag(np.square(params['std'])),
        size=100
    )
    all_data.extend(samples)
    all_labels.extend([class_id] * 100)

X = np.array(all_data)
y = np.array(all_labels)

# Cores para cada classe
colors = ['red', 'blue', 'green', 'orange']
class_names = ['Classe 0', 'Classe 1', 'Classe 2', 'Classe 3']

# Criar figura simples
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# Plotar dados de cada classe
for class_id in range(4):
    mask = y == class_id
    class_data = X[mask]
    ax.scatter(class_data[:, 0], class_data[:, 1], 
              c=colors[class_id], label=class_names[class_id], 
              alpha=0.7, s=50)

ax.set_xlabel('Coordenada X')
ax.set_ylabel('Coordenada Y') 
ax.set_title('Dataset Sintético - 4 Classes')
ax.legend()
ax.grid(True, alpha=0.3)

# Display the plot
buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())
