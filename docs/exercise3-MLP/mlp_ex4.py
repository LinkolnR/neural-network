from pathlib import Path

import numpy as np

from mlp_ex3 import (
    MLPClassifier,
    generate_uneven_clusters_multiclass,
    train_test_split_np,
    standardize_fit_transform,
    accuracy_score,
    confusion_matrix_counts,
    plot_training_loss,
    plot_confusion_matrix,
    plot_pca_scatter,
)


def main() -> None:
    # Defaults per report
    samples = 1500
    epochs = 450
    lr = 0.05
    hidden = [64, 64, 32]
    seed = 42
    class_sep = 1.6
    flip_y = 0.01
    clusters = (2, 3, 4)

    if len(hidden) < 2:
        raise ValueError("Exercise 4 requires at least two hidden layers")

    out_dir = Path(__file__).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    X, y = generate_uneven_clusters_multiclass(
        total_samples=samples,
        class_cluster_counts=clusters,
        n_features=4,
        n_informative=4,
        n_redundant=0,
        class_sep=class_sep,
        flip_y=flip_y,
        seed=seed,
    )

    X_train, X_test, y_train, y_test = train_test_split_np(X, y, test_size=0.2, seed=seed)
    X_train_s, X_test_s, mean, std = standardize_fit_transform(X_train, X_test)

    model = MLPClassifier(input_dim=4, hidden_layers=hidden, output_dim=3, learning_rate=lr, seed=seed)

    losses = model.fit(X_train_s, y_train, epochs=epochs, num_classes=3)

    yhat_train = model.predict(X_train_s)
    yhat_test = model.predict(X_test_s)
    acc = accuracy_score(y_test, yhat_test)
    mat = confusion_matrix_counts(y_test, yhat_test, num_classes=3)

    plot_training_loss(losses, out_dir / "training_loss_ex4.png")
    plot_pca_scatter(
        X_train_s, y_train, X_test_s, y_test, yhat_train, yhat_test, out_dir / "pca_scatter_ex4.png"
    )
    plot_confusion_matrix(mat, out_dir / "confusion_matrix_ex4.png")

    metrics_text = (
        f"Accuracy: {acc:.4f}\n"
        f"Confusion Matrix (rows=true, cols=pred):\n{mat}\n"
        f"Final train loss: {losses[-1]:.6f}\n"
        f"Hidden layers: {hidden}\n"
        f"Learning rate: {lr}\n"
        f"Epochs: {epochs}\n"
        f"Seed: {seed}\n"
        f"Class sep: {class_sep}, flip_y: {flip_y}, clusters: {clusters}\n"
    )
    (out_dir / "metrics_ex4.txt").write_text(metrics_text, encoding="utf-8")
    print(metrics_text)


if __name__ == "__main__":
    main()


