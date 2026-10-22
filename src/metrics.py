"""Shared evaluation: accuracy, macro precision/recall/F1, confusion matrix."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)


def evaluate_and_save(y_true, y_pred, class_names, name, extra=None, out_dir="results"):
    Path(out_dir).mkdir(exist_ok=True)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    res = {"model": name,
           "accuracy": accuracy_score(y_true, y_pred),
           "precision_macro": p, "recall_macro": r, "f1_macro": f1,
           "report": classification_report(y_true, y_pred, target_names=class_names,
                                           output_dict=True, zero_division=0)}
    if extra:
        res.update(extra)
    json.dump(res, open(f"{out_dir}/{name}_metrics.json", "w"), indent=2)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title(f"{name} confusion matrix")
    plt.tight_layout(); plt.savefig(f"{out_dir}/{name}_confusion.png", dpi=150); plt.close()
    print(f"\n=== {name} ===")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
    return res
