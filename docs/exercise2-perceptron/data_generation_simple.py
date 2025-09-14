import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO

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


fig, ax = plt.subplots(1,1,figsize=(8, 6))
ax.scatter(X[y == 0, 0], X[y == 0, 1], color='red', label='Classe 0')
ax.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', label='Classe 1')
ax.set_title('Visualização dos Dados Gerados')
ax.set_xlabel('Característica 1')
ax.set_ylabel('Característica 2')
ax.legend()
ax.grid(True)



# Salvar o plot em um buffer SVG e imprimir para o mkdocs
buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True, bbox_inches='tight')
print(buffer.getvalue())
