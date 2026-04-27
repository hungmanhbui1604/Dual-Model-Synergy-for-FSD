import numpy as np
from sklearn.metrics import accuracy_score, roc_curve


def compute_pad_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    # Compute ROC
    fpr, tpr, thresholds = roc_curve(labels, probabilities)

    # Convert to PAD metrics
    apcer = 1 - tpr  # FNR
    bpcer = fpr  # FPR
    ace = (apcer + bpcer) / 2

    # Find best threshold (DISCRETE)
    idx = np.argmin(ace)
    threshold = thresholds[idx]

    # Compute accuracy at this threshold
    predictions = (probabilities >= threshold).astype(int)
    accuracy = accuracy_score(labels, predictions)

    return {
        "threshold": threshold,
        "accuracy": accuracy * 100.0,
        "ace": ace[idx] * 100.0,
        "apcer": apcer[idx] * 100.0,
        "bpcer": bpcer[idx] * 100.0,
    }
