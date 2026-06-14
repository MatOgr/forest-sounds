import time

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)


def classification_metrics(y_true, y_pred, average="macro"):
    return {
        "acc": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
    }


def per_class_f1(y_true, y_pred, class_names=None):
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    if class_names is None:
        class_names = [str(i) for i in range(len(f1))]
    return dict(zip(class_names, f1))


def inference_latency(predict_fn, sample, n=50, warmup=5):
    """Return mean / p50 / p95 latency in ms."""
    for _ in range(warmup):
        predict_fn(sample)
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        predict_fn(sample)
        ts.append((time.perf_counter() - t0) * 1000)
    ts = np.array(ts)
    return {
        "mean_ms": ts.mean(),
        "p50_ms": np.percentile(ts, 50),
        "p95_ms": np.percentile(ts, 95),
    }


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def report_text(y_true, y_pred, class_names=None):
    return classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    )
