import sys; sys.path.insert(0, ".")
import time
from src.dataset import load_splits
from src import classical_model as cm
from src.metrics import evaluate_and_save

sp = load_splits(); classes = sp["classes"]
print("Extracting features...")
X = {k: cm.build_matrix(sp[k]["paths"]) for k in ("train", "val", "test")}
y = {k: sp[k]["labels"] for k in ("train", "val", "test")}
import numpy as np
X_trval = np.vstack([X["train"], X["val"]]); y_trval = y["train"] + y["val"]

best, best_f1 = None, -1
from sklearn.metrics import f1_score
for kind in ("svm", "rf"):
    model = cm.train_classical(X["train"], y["train"], kind)
    f1 = f1_score(y["val"], model.predict(X["val"]), average="macro")
    print(f"{kind} val macro-F1 = {f1:.3f}")
    if f1 > best_f1: best, best_f1, best_kind = model, f1, kind
    pred = model.predict(X["test"])
    evaluate_and_save(y["test"], pred, classes, f"classical_{kind}")
cm.save(best); print("Saved best classical model:", best_kind)
lat = cm.predict_latency_ms(best, sp["test"]["paths"])
import json; json.dump({"best": best_kind, "latency_ms_cpu": lat}, open("results/classical_info.json", "w"))
print(f"Classical latency (incl. feature extraction): {lat:.1f} ms/img")
