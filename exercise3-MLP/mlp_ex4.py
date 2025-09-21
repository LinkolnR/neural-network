import argparse
from pathlib import Path

import numpy as np

# Reuse Exercise 3 implementation for the model and utilities
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
    parser = argparse.ArgumentParser(
        description="Exercise 4: Multiclass classification with deeper MLP (>=2 hidden layers)"
    )
    parser.add_argument("--samples", type=int, default=1500, help="Total number of samples")
    parser.add_argument("--epochs", type=int, default=450, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate")
    # Enforce at least 2 hidden layers by default and via validation
    parser.add_argument(
        "--hidden", type=int, nargs="+", default=[64, 64, 32], help="Hidden layer sizes (>=2)"
    )
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

    if len(args.hidden) < 2:
        raise ValueError("Exercise 4 requires at least two hidden layers (pass --hidden with >=2 sizes)")

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

    # Model: reuse Exercise 3 MLP, just deeper via --hidden
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
    plot_training_loss(losses, out_dir / "training_loss_ex4.png")
    plot_pca_scatter(
        X_train_s, y_train, X_test_s, y_test, yhat_train, yhat_test, out_dir / "pca_scatter_ex4.png"
    )
    plot_confusion_matrix(mat, out_dir / "confusion_matrix_ex4.png")

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
    (out_dir / "metrics_ex4.txt").write_text(metrics_text, encoding="utf-8")
    print(metrics_text)


if __name__ == "__main__":
    main()


