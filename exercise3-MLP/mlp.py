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
# Data generation
# -----------------------------

def generate_uneven_cluster_data(
    total_samples: int = 1000,
    n_features: int = 2,
    n_informative: int = 2,
    n_redundant: int = 0,
    class_sep: float = 1.5,
    flip_y: float = 0.01,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a binary dataset where class 0 has 1 cluster and class 1 has 2 clusters.

    We call make_classification twice, using weights to force single-class outputs per call:
      - Call A: weights=[1.0, 0.0], n_clusters_per_class=1  -> class 0 only
      - Call B: weights=[0.0, 1.0], n_clusters_per_class=2  -> class 1 only

    This achieves uneven clusters across classes while still using make_classification.
    """
    if total_samples < 2:
        raise ValueError("total_samples must be >= 2")

    rng = np.random.default_rng(seed)
    n0 = total_samples // 2
    n1 = total_samples - n0

    X0, y0 = make_classification(
        n_samples=n0,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=1,
        weights=[1.0, 0.0],  # class 0 only
        flip_y=flip_y,
        class_sep=class_sep,
        random_state=seed,
    )
    # Ensure label is 0
    y0 = np.zeros_like(y0)

    X1, y1 = make_classification(
        n_samples=n1,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=2,
        weights=[0.0, 1.0],  # class 1 only
        flip_y=flip_y,
        class_sep=class_sep,
        random_state=seed + 1,
    )
    # Ensure label is 1
    y1 = np.ones_like(y1)

    X = np.vstack([X0, X1])
    y = np.concatenate([y0, y1])

    # Shuffle
    idx = rng.permutation(len(y))
    X = X[idx]
    y = y[idx]
    return X.astype(np.float64), y.astype(np.int64)


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
# MLP from scratch (NumPy only for tensors)
# -----------------------------


def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, z)


def relu_grad(z: np.ndarray) -> np.ndarray:
    return (z > 0.0).astype(z.dtype)


def sigmoid(z: np.ndarray) -> np.ndarray:
    # numerically stable sigmoid
    positive_mask = z >= 0
    negative_mask = ~positive_mask
    out = np.empty_like(z)
    out[positive_mask] = 1.0 / (1.0 + np.exp(-z[positive_mask]))
    exp_z = np.exp(z[negative_mask])
    out[negative_mask] = exp_z / (1.0 + exp_z)
    return out


def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    eps = 1e-12
    y_pred = np.clip(y_pred, eps, 1.0 - eps)
    loss = -(
        y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred)
    )
    return float(np.mean(loss))


@dataclass
class LayerParams:
    weights: np.ndarray
    bias: np.ndarray


class MLPBinaryClassifier:
    """
    A simple MLP for binary classification with ReLU hidden layers and sigmoid output.

    - Loss: Binary Cross Entropy
    - Optimizer: Vanilla Gradient Descent
    - Initialization: He for ReLU layers; Xavier for final sigmoid layer
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layers: List[int],
        learning_rate: float = 0.05,
        seed: int = 42,
    ) -> None:
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if not hidden_layers:
            raise ValueError("hidden_layers must contain at least one layer size")
        if any(h <= 0 for h in hidden_layers):
            raise ValueError("hidden layer sizes must be positive")

        self.learning_rate = float(learning_rate)
        self.rng = np.random.default_rng(seed)

        layer_dims = [input_dim] + list(hidden_layers) + [1]
        self.layers: List[LayerParams] = []

        for i in range(len(layer_dims) - 1):
            fan_in, fan_out = layer_dims[i], layer_dims[i + 1]
            if i < len(layer_dims) - 2:
                # Hidden layer: He init for ReLU
                std = np.sqrt(2.0 / fan_in)
            else:
                # Output layer: Xavier for sigmoid
                std = np.sqrt(1.0 / fan_in)
            W = self.rng.normal(0.0, std, size=(fan_in, fan_out)).astype(np.float64)
            b = np.zeros((1, fan_out), dtype=np.float64)
            self.layers.append(LayerParams(weights=W, bias=b))

    def _forward(self, X: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Return (activations, pre_activations). activations[0] is input X."""
        activations = [X]
        pre_activations = []
        A = X
        # Hidden layers with ReLU
        for i, layer in enumerate(self.layers[:-1]):
            Z = A @ layer.weights + layer.bias
            A = relu(Z)
            pre_activations.append(Z)
            activations.append(A)
        # Output layer with sigmoid
        last = self.layers[-1]
        Z = activations[-1] @ last.weights + last.bias
        A = sigmoid(Z)
        pre_activations.append(Z)
        activations.append(A)
        return activations, pre_activations

    def _backward(
        self, activations: List[np.ndarray], pre_activations: List[np.ndarray], y: np.ndarray
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Compute gradients for each layer (dW, db), ordered like self.layers.
        Uses BCE loss with sigmoid output ⇒ dZ_last = (A_last - y)/m.
        """
        m = y.shape[0]
        grads: List[Tuple[np.ndarray, np.ndarray]] = [None] * len(self.layers)  # type: ignore

        # Output layer
        A_last = activations[-1]  # shape (m, 1)
        dZ = (A_last - y.reshape(-1, 1)) / m
        A_prev = activations[-2]
        dW = A_prev.T @ dZ
        db = np.sum(dZ, axis=0, keepdims=True)
        grads[-1] = (dW, db)

        # Backprop through hidden layers
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

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 200) -> List[float]:
        losses: List[float] = []
        for _ in range(int(epochs)):
            activations, pre_activations = self._forward(X)
            y_hat = activations[-1]
            loss = binary_cross_entropy(y.reshape(-1, 1), y_hat)
            losses.append(loss)
            grads = self._backward(activations, pre_activations, y)
            self._update(grads)
        return losses

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        A, _ = self._forward(X)
        return A[-1].reshape(-1)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(np.int64)


# -----------------------------
# Evaluation & visualization
# -----------------------------


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true.astype(np.int64) == y_pred.astype(np.int64))))


def confusion_matrix_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, tn, fp, fn


def plot_training_loss(losses: List[float], out_path: Path) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(losses, label="Train loss")
    plt.xlabel("Epoch")
    plt.ylabel("Binary cross-entropy")
    plt.title("Training Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_decision_boundary(
    model: MLPBinaryClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    out_path: Path,
) -> None:
    all_X = np.vstack([X_train, X_test])
    x_min, x_max = all_X[:, 0].min() - 0.5, all_X[:, 0].max() + 0.5
    y_min, y_max = all_X[:, 1].min() - 0.5, all_X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = model.predict_proba(grid).reshape(xx.shape)

    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, zz, levels=50, cmap="RdBu", alpha=0.6)
    plt.colorbar(label="P(class=1)")
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap="bwr", edgecolor="k", s=20, label="train")
    plt.scatter(
        X_test[:, 0], X_test[:, 1], c=y_test, cmap="bwr", marker="^", edgecolor="k", s=28, label="test"
    )
    plt.legend()
    plt.title("Decision Boundary")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_confusion_matrix(tp: int, tn: int, fp: int, fn: int, out_path: Path) -> None:
    mat = np.array([[tn, fp], [fn, tp]], dtype=float)
    labels = np.array([["TN", "FP"], ["FN", "TP"]])
    plt.figure(figsize=(4, 4))
    plt.imshow(mat, cmap="Blues")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, f"{labels[i, j]}\n{int(mat[i, j])}", ha="center", va="center")
    plt.xticks([0, 1], ["Pred 0", "Pred 1"])\
        ; plt.yticks([0, 1], ["True 0", "True 1"])  # noqa: E702
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


# -----------------------------
# CLI
# -----------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Binary classification with scratch MLP")
    parser.add_argument("--samples", type=int, default=1000, help="Total number of samples")
    parser.add_argument("--epochs", type=int, default=300, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate")
    parser.add_argument(
        "--hidden", type=int, nargs="+", default=[16, 16], help="Hidden layer sizes"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--class_sep", type=float, default=1.5, help="Class separability")
    parser.add_argument("--flip_y", type=float, default=0.01, help="Label noise fraction")
    args = parser.parse_args()

    out_dir = Path(__file__).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Generate data
    X, y = generate_uneven_cluster_data(
        total_samples=args.samples,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        class_sep=args.class_sep,
        flip_y=args.flip_y,
        seed=args.seed,
    )

    # Split
    X_train, X_test, y_train, y_test = train_test_split_np(X, y, test_size=0.2, seed=args.seed)

    # Standardize using train statistics
    X_train_s, X_test_s, mean, std = standardize_fit_transform(X_train, X_test)

    # Model
    model = MLPBinaryClassifier(
        input_dim=2, hidden_layers=args.hidden, learning_rate=args.lr, seed=args.seed
    )

    # Train
    losses = model.fit(X_train_s, y_train, epochs=args.epochs)

    # Evaluate
    y_test_prob = model.predict_proba(X_test_s)
    y_test_pred = (y_test_prob >= 0.5).astype(np.int64)
    acc = accuracy_score(y_test, y_test_pred)
    tp, tn, fp, fn = confusion_matrix_counts(y_test, y_test_pred)

    # Save plots
    plot_training_loss(losses, out_dir / "training_loss.png")
    plot_decision_boundary(model, X_train_s, y_train, X_test_s, y_test, out_dir / "decision_boundary.png")
    plot_confusion_matrix(tp, tn, fp, fn, out_dir / "confusion_matrix.png")

    # Save metrics
    metrics_text = (
        f"Accuracy: {acc:.4f}\n"
        f"TP={tp}, TN={tn}, FP={fp}, FN={fn}\n"
        f"Final train loss: {losses[-1]:.6f}\n"
        f"Hidden layers: {args.hidden}\n"
        f"Learning rate: {args.lr}\n"
        f"Epochs: {args.epochs}\n"
        f"Seed: {args.seed}\n"
        f"Class sep: {args.class_sep}, flip_y: {args.flip_y}\n"
    )
    (out_dir / "metrics.txt").write_text(metrics_text, encoding="utf-8")

    # Print key results to stdout
    print(metrics_text)


if __name__ == "__main__":
    main()


