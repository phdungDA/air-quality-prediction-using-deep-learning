"""
Tính các metric đánh giá cho bài toán phân loại AQI (5 mức).
Đầu vào: y_true, y_pred là mảng nhãn 0..4 (đã trừ 1 từ 1..5).
"""

import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")  # ghi ảnh ra file, không cần cửa sổ GUI
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

# Nhãn 0..4 tương ứng AQI 1..5 của OpenWeatherMap
CLASS_NAMES = ["1-Good", "2-Fair", "3-Moderate", "4-Poor", "5-Very Poor"]


def compute_metrics(y_true, y_pred, num_classes: int = 5) -> dict:
    """
    Trả về dict gồm accuracy, precision/recall/F1 (macro, weighted, từng lớp)
    và confusion matrix. Truyền labels cố định để lớp vắng mặt trong tập test
    vẫn có dòng riêng (giá trị 0) thay vì làm lệch kích thước ma trận.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    labels = list(range(num_classes))

    p_cls, r_cls, f_cls, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    p_mac, r_mac, f_mac, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    p_wei, r_wei, f_wei, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    names = CLASS_NAMES[:num_classes]
    return {
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro": {"precision": float(p_mac), "recall": float(r_mac), "f1": float(f_mac)},
        "weighted": {"precision": float(p_wei), "recall": float(r_wei), "f1": float(f_wei)},
        "per_class": {
            names[i]: {
                "precision": float(p_cls[i]),
                "recall": float(r_cls[i]),
                "f1": float(f_cls[i]),
                "support": int(support[i]),
            }
            for i in labels
        },
        "confusion_matrix": cm.tolist(),
    }


def plot_confusion_matrix(cm, save_path: str, normalize: bool = False, title: str = "Confusion Matrix"):
    """Vẽ confusion matrix ra file ảnh. normalize=True: chuẩn hoá theo hàng (recall)."""
    cm = np.asarray(cm, dtype=float)
    if normalize:
        row_sum = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sum, out=np.zeros_like(cm), where=row_sum != 0)

    n = cm.shape[0]
    names = CLASS_NAMES[:n]

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_yticklabels(names)
    ax.set_xlabel("Predicted (y_pred)")
    ax.set_ylabel("True (y_true)")
    ax.set_title(title + (" (normalized)" if normalize else ""))

    thresh = cm.max() / 2 if cm.max() > 0 else 0
    for i in range(n):
        for j in range(n):
            text = f"{cm[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
            ax.text(j, i, text, ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def print_report(result: dict):
    """In kết quả gọn ra terminal."""
    print(f"\nSố mẫu test: {result['n_samples']}")
    print(f"Accuracy         : {result['accuracy']:.4f}")
    m, w = result["macro"], result["weighted"]
    print(f"Macro    P/R/F1  : {m['precision']:.4f} / {m['recall']:.4f} / {m['f1']:.4f}")
    print(f"Weighted P/R/F1  : {w['precision']:.4f} / {w['recall']:.4f} / {w['f1']:.4f}")
    print("\nTheo từng lớp:")
    print(f"{'Lớp':<14}{'Precision':>10}{'Recall':>10}{'F1':>10}{'Support':>10}")
    for name, v in result["per_class"].items():
        print(f"{name:<14}{v['precision']:>10.4f}{v['recall']:>10.4f}{v['f1']:>10.4f}{v['support']:>10d}")
    print("\nConfusion matrix (hàng = y_true, cột = y_pred):")
    print(np.array(result["confusion_matrix"]))


def save_json(result: dict, save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
