import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from io import StringIO
import sys

# Adiciona o diretório dos scripts ao sys.path para que as importações funcionem no mkdocs
# O mkdocs executa os scripts a partir da raiz do projeto.
sys.path.insert(0, 'docs/exercise2-perceptron')

# Importa as implementações do Perceptron e da GERAÇÃO DE DADOS 2 (com sobreposição)
from data_generation2 import generate_data
from perceptron_lincoln import Perceptron

def accuracy(y_true, y_pred):
    """
    Calcula a acurácia da classificação.
    """
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

# 1. Gerar o conjunto de dados com sobreposição
X, y = generate_data()

# 2. Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123 # Usamos a mesma seed para a divisão ser consistente
)

# 3. Criar e treinar o Perceptron
# Como os dados não são linearmente separáveis, o Perceptron pode não convergir.
# O resultado será a melhor fronteira que ele encontrar dentro do número de iterações.
p = Perceptron(learning_rate=0.01, n_iters=100)
p.fit(X_train, y_train)

# 4. Fazer predições e avaliar o modelo
predictions = p.predict(X_test)
print(f"Acurácia do Perceptron no conjunto de teste com sobreposição: {accuracy(y_test, predictions):.2f}")

# 5. Visualizar a fronteira de decisão
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.scatter(X_train[:, 0], X_train[:, 1], marker="o", c=y_train)

# Calcula os pontos da reta da fronteira de decisão
x0_1 = np.amin(X_train[:, 0])
x0_2 = np.amax(X_train[:, 0])

# Adicionamos uma verificação para evitar divisão por zero se o peso for nulo
if p.weights[1] != 0:
    x1_1 = (-p.weights[0] * x0_1 - p.bias) / p.weights[1]
    x1_2 = (-p.weights[0] * x0_2 - p.bias) / p.weights[1]
    ax.plot([x0_1, x0_2], [x1_1, x1_2], "k")

# Adiciona informações ao gráfico
plt.xlabel("Característica 1")
plt.ylabel("Característica 2")
plt.title("Fronteira de Decisão (Dados com Sobreposição)")
plt.ylim(np.amin(X_train[:, 1]) - 1, np.amax(X_train[:, 1]) + 1)

# Salvar o plot em um buffer SVG e imprimir para o mkdocs
buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True, bbox_inches='tight')
print(buffer.getvalue())
plt.close() # Fecha a figura para liberar memória
