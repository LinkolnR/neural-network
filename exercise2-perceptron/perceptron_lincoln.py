import numpy as np

class Perceptron:
    """
    Implementação do Perceptron a partir do zero.

    Parâmetros
    ----------
    learning_rate : float
        A taxa de aprendizado (entre 0.0 e 1.0). Controla o tamanho
        do passo na atualização dos pesos.
    n_iters : int
        O número de passagens (épocas) sobre o conjunto de dados de treino.

    Atributos
    ---------
    weights : array-like, shape = [n_features]
        Pesos após o treinamento.
    bias : float
        O termo de viés após o treinamento.
    """
    def __init__(self, learning_rate=0.01, n_iters=1000):
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

        # 1. Inicialização dos pesos e bias
        self.weights = np.zeros(n_features)
        self.bias = 0

        # O `y_` é usado para garantir que o vetor de rótulos está no formato correto
        y_ = np.array(y)

        # 5. Iteração sobre as épocas
        for _ in range(self.n_iters):
            # Itera sobre cada amostra de treino
            for idx, x_i in enumerate(X):
                # 2. Forward Pass
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self._activation_function(linear_output)

                # 3. Cálculo do Erro e 4. Atualização dos Pesos
                # A atualização só acontece se houver erro
                error = y_[idx] - y_predicted
                update = self.learning_rate * error
                
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        """
        Prevê os rótulos de classe para as amostras em X.

        Parâmetros
        ----------
        X : array-like, shape = [n_samples, n_features]
            Vetor de teste.

        Retorna
        -------
        y_pred : array-like, shape = [n_samples]
            Rótulos de classe previstos para cada amostra em X.
        """
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self._activation_function(linear_output)
        return y_predicted
