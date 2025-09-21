import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

try:
    from sklearn.datasets import make_classification
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "scikit-learn is required only for data generation (make_classification)."
    ) from exc


# -----------------------------
# Data generation (3 classes, uneven clusters per class)
# -----------------------------


def generate_uneven_clusters_multiclass(
    total_samples: int = 1500,
    class_cluster_counts: Tuple[int, int, int] = (2, 3, 4),
    n_features: int = 4,
    n_informative: int = 4,
    n_redundant: int = 0,
    class_sep: float = 1.6,
    flip_y: float = 0.01,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a 3-class dataset where class i has class_cluster_counts[i] clusters.

    We call make_classification three times, isolating one class each time via weights,
    and using different n_clusters_per_class for that class. Concatenate and shuffle.
    """
    if total_samples < 3:
        raise ValueError("total_samples must be >= 3")
    if len(class_cluster_counts) != 3:
        raise ValueError("class_cluster_counts must have length 3")

    rng = np.random.default_rng(seed)
    # roughly balance per-class samples
    base = total_samples // 3
    rem = total_samples - base * 3
    per_class = [base, base, base]
    for i in range(rem):
        per_class[i] += 1

    X_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    for cls in range(3):
        n_samples = per_class[cls]
        clusters = class_cluster_counts[cls]
        weights = [0.0, 0.0, 0.0]
        weights[cls] = 1.0
        Xc, yc = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=n_redundant,
            n_repeated=0,
            n_classes=3,
            n_clusters_per_class=clusters,
            weights=weights,
            flip_y=flip_y,
            class_sep=class_sep,
            random_state=seed + cls,
        )
        yc = np.full_like(yc, fill_value=cls)
        X_list.append(Xc)
        y_list.append(yc)

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    idx = rng.permutation(len(y))
    return X[idx].astype(np.float64), y[idx].astype(np.int64)


def train_test_split_np(
    X: np.ndarray, y: np.ndarray, test_size: float = 0.2, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size must be in (0, 1)")
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = rng.permutation(n)
    test_n = int(np.floor(test_size * n))
    test_idx = idx[:test_n]
    train_idx = idx[test_n:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def standardize_fit_transform(
    X_train: np.ndarray, X_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True) + 1e-12
    return (X_train - mean) / std, (X_test - mean) / std, mean, std


# -----------------------------
# MLP from scratch (NumPy only)
# Reuses Exercise 2 structure, adapted for softmax and 3 classes
# -----------------------------


def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, z)


def relu_grad(z: np.ndarray) -> np.ndarray:
    return (z > 0.0).astype(z.dtype)


def softmax(z: np.ndarray) -> np.ndarray:
    # Stable softmax per row
    z_max = np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z - z_max)
    return exp_z / (np.sum(exp_z, axis=1, keepdims=True) + 1e-12)


def categorical_cross_entropy(y_true_onehot: np.ndarray, y_pred: np.ndarray) -> float:
    eps = 1e-12
    y_pred = np.clip(y_pred, eps, 1.0)
    loss = -np.sum(y_true_onehot * np.log(y_pred), axis=1)
    return float(np.mean(loss))


def one_hot_encode(y: np.ndarray, num_classes: int) -> np.ndarray:
    out = np.zeros((y.shape[0], num_classes), dtype=np.float64)
    out[np.arange(y.shape[0]), y.astype(int)] = 1.0
    return out


@dataclass
class LayerParams:
    weights: np.ndarray
    bias: np.ndarray


class MLPClassifier:
    """
    Multiclass MLP with ReLU hidden layers and softmax output.
    Loss: Categorical Cross-Entropy
    Optimizer: Vanilla Gradient Descent
    Initialization: He (hidden), Xavier (output)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layers: List[int],
        output_dim: int,
        learning_rate: float = 0.05,
        seed: int = 42,
    ) -> None:
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if not hidden_layers:
            raise ValueError("hidden_layers must contain at least one layer size")
        if any(h <= 0 for h in hidden_layers):
            raise ValueError("hidden layer sizes must be positive")
        if output_dim <= 1:
            raise ValueError("output_dim must be >= 2 for multiclass")

        self.learning_rate = float(learning_rate)
        self.rng = np.random.default_rng(seed)

        layer_dims = [input_dim] + list(hidden_layers) + [output_dim]
        self.layers: List[LayerParams] = []

        for i in range(len(layer_dims) - 1):
            fan_in, fan_out = layer_dims[i], layer_dims[i + 1]
            if i < len(layer_dims) - 2:
                std = np.sqrt(2.0 / fan_in)  # He for ReLU
            else:
                std = np.sqrt(1.0 / fan_in)  # Xavier for softmax layer
            W = self.rng.normal(0.0, std, size=(fan_in, fan_out)).astype(np.float64)
            b = np.zeros((1, fan_out), dtype=np.float64)
            self.layers.append(LayerParams(weights=W, bias=b))

    def _forward(self, X: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        activations = [X]
        pre_activations = []
        A = X
        # Hidden layers with ReLU
        for layer in self.layers[:-1]:
            Z = A @ layer.weights + layer.bias
            A = relu(Z)
            pre_activations.append(Z)
            activations.append(A)
        # Output layer with softmax
        last = self.layers[-1]
        Z = activations[-1] @ last.weights + last.bias
        A = softmax(Z)
        pre_activations.append(Z)
        activations.append(A)
        return activations, pre_activations

    def _backward(
        self, activations: List[np.ndarray], pre_activations: List[np.ndarray], y_onehot: np.ndarray
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        m = y_onehot.shape[0]
        grads: List[Tuple[np.ndarray, np.ndarray]] = [None] * len(self.layers)  # type: ignore

        # Output layer gradient: (A - Y)/m
        A_last = activations[-1]
        dZ = (A_last - y_onehot) / m
        A_prev = activations[-2]
        dW = A_prev.T @ dZ
        db = np.sum(dZ, axis=0, keepdims=True)
        grads[-1] = (dW, db)

        # Propagate backward through hidden layers
        dA = dZ @ self.layers[-1].weights.T
        for l in range(len(self.layers) - 2, -1, -1):
            Z = pre_activations[l]
            dZ = dA * relu_grad(Z)
            A_prev = activations[l]
            dW = A_prev.T @ dZ
            db = np.sum(dZ, axis=0, keepdims=True)
            grads[l] = (dW, db)
            if l > 0:
                dA = dZ @ self.layers[l].weights.T

        return grads

    def _update(self, grads: List[Tuple[np.ndarray, np.ndarray]]) -> None:
        for layer, (dW, db) in zip(self.layers, grads):
            layer.weights -= self.learning_rate * dW
            layer.bias -= self.learning_rate * db

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 300, num_classes: int = 3) -> List[float]:
        losses: List[float] = []
        y_onehot = one_hot_encode(y, num_classes)
        for _ in range(int(epochs)):
            activations, pre_activations = self._forward(X)
            y_hat = activations[-1]
            loss = categorical_cross_entropy(y_onehot, y_hat)
            losses.append(loss)
            grads = self._backward(activations, pre_activations, y_onehot)
            self._update(grads)
        return losses

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        A, _ = self._forward(X)
        return A[-1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1).astype(np.int64)


# -----------------------------
# Evaluation & visualization
# -----------------------------


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true.astype(np.int64) == y_pred.astype(np.int64))))


def confusion_matrix_counts(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    mat = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true.astype(int), y_pred.astype(int)):
        mat[t, p] += 1
    return mat


def plot_training_loss(losses: List[float], out_path: Path) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(losses, label="Train loss")
    plt.xlabel("Epoch")
    plt.ylabel("Categorical cross-entropy")
    plt.title("Training Loss (Multiclass)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def pca_project_2d(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Center
    mean = X.mean(axis=0, keepdims=True)
    Xc = X - mean
    # Covariance and eigen decomposition
    cov = (Xc.T @ Xc) / (X.shape[0] - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    W = eigvecs[:, order[:2]]
    Z = Xc @ W
    return Z, mean, W


def plot_pca_scatter(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    yhat_train: np.ndarray,
    yhat_test: np.ndarray,
    out_path: Path,
) -> None:
    Z_train, mean, W = pca_project_2d(X_train)
    Z_test = (X_test - mean) @ W

    plt.figure(figsize=(7, 5))
    plt.scatter(Z_train[:, 0], Z_train[:, 1], c=yhat_train, cmap="tab10", edgecolor="k", s=18, label="train (pred)")
    plt.scatter(Z_test[:, 0], Z_test[:, 1], c=yhat_test, cmap="tab10", marker="^", edgecolor="k", s=24, label="test (pred)")
    plt.title("PCA 2D Projection colored by predicted class")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_confusion_matrix(mat: np.ndarray, out_path: Path) -> None:
    num_classes = mat.shape[0]
    plt.figure(figsize=(4.8, 4.4))
    plt.imshow(mat, cmap="Blues")
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, str(int(mat[i, j])), ha="center", va="center")
    plt.xticks(np.arange(num_classes), [f"Pred {i}" for i in range(num_classes)])
    plt.yticks(np.arange(num_classes), [f"True {i}" for i in range(num_classes)])
    plt.title("Confusion Matrix (Multiclass)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


# -----------------------------
# CLI
# -----------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Exercise 3: Multiclass classification with scratch MLP")
    parser.add_argument("--samples", type=int, default=1500, help="Total number of samples")
    parser.add_argument("--epochs", type=int, default=400, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate")
    parser.add_argument("--hidden", type=int, nargs="+", default=[32, 32], help="Hidden layer sizes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--class_sep", type=float, default=1.6, help="Class separability")
    parser.add_argument("--flip_y", type=float, default=0.01, help="Label noise fraction")
    parser.add_argument(
        "--clusters",
        type=int,
        nargs=3,
        default=[2, 3, 4],
        help="Clusters per class for classes 0,1,2",
    )
    args = parser.parse_args()

    out_dir = Path(__file__).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Data
    X, y = generate_uneven_clusters_multiclass(
        total_samples=args.samples,
        class_cluster_counts=tuple(args.clusters),
        n_features=4,
        n_informative=4,
        n_redundant=0,
        class_sep=args.class_sep,
        flip_y=args.flip_y,
        seed=args.seed,
    )

    X_train, X_test, y_train, y_test = train_test_split_np(X, y, test_size=0.2, seed=args.seed)
    X_train_s, X_test_s, mean, std = standardize_fit_transform(X_train, X_test)

    # Model
    model = MLPClassifier(
        input_dim=4, hidden_layers=args.hidden, output_dim=3, learning_rate=args.lr, seed=args.seed
    )

    # Train
    losses = model.fit(X_train_s, y_train, epochs=args.epochs, num_classes=3)

    # Evaluate
    yhat_train = model.predict(X_train_s)
    yhat_test = model.predict(X_test_s)
    acc = accuracy_score(y_test, yhat_test)
    mat = confusion_matrix_counts(y_test, yhat_test, num_classes=3)

    # Save plots
    plot_training_loss(losses, out_dir / "training_loss_ex3.png")
    plot_pca_scatter(
        X_train_s, y_train, X_test_s, y_test, yhat_train, yhat_test, out_dir / "pca_scatter_ex3.png"
    )
    plot_confusion_matrix(mat, out_dir / "confusion_matrix_ex3.png")

    # Save metrics
    metrics_text = (
        f"Accuracy: {acc:.4f}\n"
        f"Confusion Matrix (rows=true, cols=pred):\n{mat}\n"
        f"Final train loss: {losses[-1]:.6f}\n"
        f"Hidden layers: {args.hidden}\n"
        f"Learning rate: {args.lr}\n"
        f"Epochs: {args.epochs}\n"
        f"Seed: {args.seed}\n"
        f"Class sep: {args.class_sep}, flip_y: {args.flip_y}, clusters: {tuple(args.clusters)}\n"
    )
    (out_dir / "metrics_ex3.txt").write_text(metrics_text, encoding="utf-8")
    print(metrics_text)


if __name__ == "__main__":
    main()


