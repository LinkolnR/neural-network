# Perceptron

## Parte 1: Implementação Perceptron

### Estrutura do Código

Criaremos uma classe `Perceptron`

```python
import numpy as np

class Perceptron:
    """
    Implementação do Perceptron

    Parâmetros
    ----------
    learning_rate : float
        A taxa de aprendizado (entre 0.0 e 1.0)
    n_iters : int
        O número de passagens (épocas)

    Atributos
    ---------
    weights
    bias
    """
    def __init__(self, learning_rate=0.01, n_iters=10):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def _activation_function(self, x):
        """
        Função de Ativação Degrau (Heaviside Step Function).
        Retorna 1 se x >= 0, senão retorna 0.
        """
        return np.where(x >= 0, 1, 0)

    def fit(self, X, y):
        """
        Ajusta o modelo aos dados de treinamento.

        Parâmetros
        ----------
        X : array-like, shape = [n_samples, n_features]
            Vetor de treinamento, onde n_samples é o número de amostras
            e n_features é o número de características.
        y : array-like, shape = [n_samples]
            Vetor com os rótulos (labels) alvo.
        """
        n_samples, n_features = X.shape

        # Inicialização dos pesos e bias
        self.weights = np.zeros(n_features)
        self.bias = 0

        y_ = np.array(y)

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self._activation_function(linear_output)
                # Cálculo do Erro e Atualização dos Pesos
                # A atualização só acontece se houver erro
                error = y_[idx] - y_predicted
                update = self.learning_rate * error
                
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self._activation_function(linear_output)
        return y_predicted

```

---
## Parte 2: Exercícios 

### Exercício 1: Dados Linearmente Separáveis

#### 1. Geração dos Dados

Utilizamos o script `data_generation.py` para criar duas classes de pontos que estão distantes uma da outra, garantindo a separabilidade linear. Os parâmetros chave são:

```python
# data_generation.py
mean0 = [1.5, 1.5]
cov0 = [[0.5, 0], [0, 0.5]]
mean1 = [5, 5]
cov1 = [[0.5, 0], [0, 0.5]]
```

Ao executar o script, obtemos a seguinte visualização:

```python exec="true"
--8<-- "docs/exercise2-perceptron/data_generation_simple.py"
```

Como podemos ver no gráfico, os dois grupos (vermelho e azul) não se misturam, representando um caso ideal para o Perceptron.

#### 2. Treinamento

No script `test_perceptron.py`, instanciamos nosso Perceptron e o treinamos com 80% dos dados gerados, utilizando o método `fit`.

```python
# Trecho de test_perceptron.py
# ...
p = Perceptron(learning_rate=0.01, n_iters=10)
p.fit(X_train, y_train)
```

#### 3. Avaliação (Evaluation)

Após o treinamento, usamos os 20% de dados restantes (o conjunto de teste) para verificar o quão bem o modelo generaliza para dados não vistos.

```python
# Trecho de test_perceptron.py
predictions = p.predict(X_test)
print(f"Acurácia do Perceptron no conjunto de teste: {accuracy(y_test, predictions):.2f}")
```

**Resultado Obtido:**
Ao executar o script, a acurácia impressa no terminal é **1.00** (ou 100%).

#### 4. Análise

O resultado de 100% de acurácia confirma o **Teorema da Convergência do Perceptron**, que afirma que o algoritmo *sempre* encontrará uma solução (uma linha de separação) em um número finito de passos, desde que os dados sejam linearmente separáveis. 

Podemos visualizar essa solução plotando a fronteira de decisão que o Perceptron aprendeu:

```python exec="true"
--8<-- "docs/exercise2-perceptron/test_perceptron.py"
```

A linha preta representa a fronteira final encontrada pelo algoritmo. Qualquer ponto de um lado da linha é classificado como uma classe, e qualquer ponto do outro lado é classificado como a outra, separando perfeitamente os dados, que são claramente linearmente separáveis.

### Exercício 2: Dados com Sobreposição (Não Linearmente Separáveis)

#### 1. Geração dos Dados

Utilizamos o script `data_generation2.py` para criar duas classes de pontos com médias mais próximas e maior variância, o que causa uma sobreposição entre elas. Os parâmetros chave são:

```python
# data_generation2.py
mean0 = [3, 3]
cov0 = [[1.5, 0], [0, 1.5]]
mean1 = [4, 4]
cov1 = [[1.5, 0], [0, 1.5]]
```

Ao executar o script, obtemos a seguinte visualização, onde a mistura entre os pontos vermelhos e azuis é clara:

```python exec="true"
--8<-- "docs/exercise2-perceptron/data_generation2.py"
```

#### 2. Treinamento

O treinamento, realizado pelo script `test_perceptron2.py`, segue o mesmo procedimento. O Perceptron tentará encontrar a melhor linha reta possível para separar as duas classes.

```python
# Trecho de test_perceptron2.py
p = Perceptron(learning_rate=0.01, n_iters=10)
p.fit(X_train, y_train)
```

#### 3. Avaliação (Evaluation)

A avaliação é feita da mesma forma, usando o conjunto de teste. No entanto, o resultado esperado é diferente.

```python
# test_perceptron2.py
predictions = p.predict(X_test)
print(f"Acurácia do Perceptron no conjunto de teste com sobreposição: {accuracy(y_test, predictions):.2f}")
```

**Resultado Obtido:**
Ao executar o script, a acurácia impressa fica em torno de **0.66** (ou 66%). Este valor pode variar ligeiramente devido à aleatoriedade na divisão treino-teste, mas nunca chegará a 100%.

#### 4. Análise

A acurácia abaixo de 100% demonstra a principal limitação do Perceptron: ele só pode encontrar soluções perfeitas para problemas linearmente separáveis. Como os dados se sobrepõem, não existe uma única linha reta que consiga dividir as duas classes sem cometer erros.

O algoritmo não "converge" para uma solução sem erros. Em vez disso, a fronteira de decisão pode oscilar um pouco durante o treinamento. O resultado final, após o número definido de iterações, é a melhor tentativa do Perceptron de minimizar os erros com uma única linha.

A fronteira de decisão, visualizada ao executar o teste, mostra essa tentativa:

```python exec="true"
--8<-- "docs/exercise2-perceptron/test_perceptron2.py"
```

A linha preta corta através da área de sobreposição, classificando incorretamente alguns pontos de ambas as classes que estão do lado "errado" da fronteira. Este é o comportamento esperado e ilustra por que modelos mais complexos, como Redes Neurais de Múltiplas Camadas (MLPs), são necessários para resolver problemas não-lineares.
