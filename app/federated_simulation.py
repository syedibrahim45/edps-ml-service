"""Experimental FedAvg simulation; this does not replace the API model."""

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from app.train_model import FEATURE_NAMES, create_synthetic_dataset


FEATURE_MIN = np.array([1, 0, 40, 20, 5, 20, 50], dtype=float)
FEATURE_MAX = np.array([120, 1, 300, 1000, 100, 250, 1000], dtype=float)


def _scale_features(features: pd.DataFrame) -> np.ndarray:
    return (features.loc[:, FEATURE_NAMES].to_numpy(dtype=float) - FEATURE_MIN) / (
        FEATURE_MAX - FEATURE_MIN
    )


def _partition_clients(
    features: pd.DataFrame,
    labels: pd.Series,
    client_count: int,
    random_state: int,
) -> list[np.ndarray]:
    if client_count < 2:
        raise ValueError("At least two clients are required for this simulation.")

    random_generator = np.random.default_rng(random_state)
    client_indices: list[list[int]] = [[] for _ in range(client_count)]
    for label in sorted(labels.unique()):
        indices = labels.index[labels == label].to_numpy().copy()
        if len(indices) < client_count:
            raise ValueError("Each class must have at least one row per client.")
        random_generator.shuffle(indices)
        for client_index, partition in enumerate(np.array_split(indices, client_count)):
            client_indices[client_index].extend(partition.tolist())

    return [np.asarray(sorted(indices), dtype=int) for indices in client_indices]


def aggregate_client_parameters(
    client_parameters: list[tuple[int, np.ndarray, float]],
) -> tuple[np.ndarray, float]:
    """Aggregate coefficients and intercept by each client's sample count."""
    total_samples = sum(sample_count for sample_count, _, _ in client_parameters)
    if total_samples <= 0:
        raise ValueError("FedAvg requires at least one client training sample.")
    averaged_weights = sum(
        sample_count * weights
        for sample_count, weights, _ in client_parameters
    ) / total_samples
    averaged_intercept = sum(
        sample_count * intercept
        for sample_count, _, intercept in client_parameters
    ) / total_samples
    return averaged_weights, float(averaged_intercept)


def run_federated_simulation(
    client_count: int = 3,
    rounds: int = 10,
    local_epochs: int = 5,
    learning_rate: float = 0.5,
    regularization: float = 0.01,
    random_state: int = 42,
) -> dict:
    """Train local logistic models and aggregate their parameters with FedAvg."""
    if rounds < 1 or local_epochs < 1 or learning_rate <= 0 or regularization < 0:
        raise ValueError("Training settings must be positive (regularization non-negative).")

    dataset = create_synthetic_dataset()
    features = dataset.loc[:, FEATURE_NAMES]
    labels = dataset["disease"]
    train_features, test_features, train_labels, test_labels = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=random_state,
        stratify=labels,
    )
    train_features = train_features.reset_index(drop=True)
    train_labels = train_labels.reset_index(drop=True)
    clients = _partition_clients(
        train_features, train_labels, client_count, random_state
    )
    scaled_training_data = _scale_features(train_features)
    label_array = train_labels.to_numpy(dtype=float)

    global_weights = np.zeros(len(FEATURE_NAMES), dtype=float)
    global_intercept = 0.0

    for _ in range(rounds):
        client_parameters = []
        for client_indices in clients:
            local_weights = global_weights.copy()
            local_intercept = global_intercept
            client_features = scaled_training_data[client_indices]
            client_labels = label_array[client_indices]

            for _ in range(local_epochs):
                probabilities = 1.0 / (
                    1.0 + np.exp(-np.clip(client_features @ local_weights + local_intercept, -35, 35))
                )
                errors = probabilities - client_labels
                weight_gradient = (
                    client_features.T @ errors / len(client_labels)
                    + regularization * local_weights
                )
                intercept_gradient = float(errors.mean())
                local_weights -= learning_rate * weight_gradient
                local_intercept -= learning_rate * intercept_gradient

            client_parameters.append(
                (len(client_indices), local_weights, local_intercept)
            )

        global_weights, global_intercept = aggregate_client_parameters(
            client_parameters
        )

    test_matrix = _scale_features(test_features)
    test_probabilities = 1.0 / (
        1.0 + np.exp(-np.clip(test_matrix @ global_weights + global_intercept, -35, 35))
    )
    test_predictions = (test_probabilities >= 0.5).astype(int)
    test_truth = test_labels.to_numpy(dtype=int)

    return {
        "method": "FedAvg",
        "model": "binary logistic regression",
        "client_count": client_count,
        "rounds": rounds,
        "local_epochs": local_epochs,
        "aggregation": "sample-count-weighted arithmetic mean of local coefficients and intercepts",
        "preprocessing": "fixed public feature bounds; no global data-fitted scaler",
        "training_rows": len(train_features),
        "test_rows": len(test_features),
        "client_rows": [len(indices) for indices in clients],
        "test_class_counts": test_labels.value_counts().sort_index().to_dict(),
        "confusion_matrix_labels_0_1": confusion_matrix(
            test_truth, test_predictions, labels=[0, 1]
        ).tolist(),
        "accuracy": accuracy_score(test_truth, test_predictions),
        "precision_class_1": precision_score(
            test_truth, test_predictions, pos_label=1, zero_division=0
        ),
        "recall_class_1": recall_score(
            test_truth, test_predictions, pos_label=1, zero_division=0
        ),
        "f1_class_1": f1_score(
            test_truth, test_predictions, pos_label=1, zero_division=0
        ),
        "global_coefficients": global_weights.tolist(),
        "global_intercept": global_intercept,
        "interpretation": (
            "Experimental simulation on synthetic data only. Four held-out records "
            "cannot support reliable performance or privacy claims; clients and "
            "communication are simulated in one process."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clients", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--local-epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = run_federated_simulation(
        client_count=args.clients,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        random_state=args.seed,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()