import sys; sys.path.insert(0, ".")
import json, numpy as np, torch
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
from src.dataset import load_splits, AvocadoDataset, train_transforms, eval_transforms
from src import dl_model as dl
from src.metrics import evaluate_and_save

sp = load_splits(); classes = sp["classes"]
mk = lambda k, tf, sh: DataLoader(AvocadoDataset(sp[k]["paths"], sp[k]["labels"], tf),
                                  batch_size=32, shuffle=sh, num_workers=2)
tr, va, te = mk("train", train_transforms(), True), mk("val", eval_transforms(), False), mk("test", eval_transforms(), False)
cw = compute_class_weight("balanced", classes=np.arange(len(classes)), y=sp["train"]["labels"])
model = dl.build_avocado_classifier(len(classes))
model, hist = dl.fit(model, tr, va, class_weights=cw, epochs=25, lr=1e-3, patience=5, finetune_after=8)
torch.save({"state_dict": model.state_dict(), "classes": classes}, "models/mobilenetv2.pth")
json.dump(hist, open("results/dl_history.json", "w"))
y, p = dl.predict_loader(model, te)
extra = {"latency_ms_cpu": dl.latency_ms(model, "cpu")}
if torch.cuda.is_available(): extra["latency_ms_gpu"] = dl.latency_ms(model, "cuda")
evaluate_and_save(y, p, classes, "mobilenetv2", extra)
print(extra)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
h = hist; plt.plot([e["train_loss"] for e in h], label="train"); plt.plot([e["val_loss"] for e in h], label="val")
plt.legend(); plt.xlabel("epoch"); plt.title("MobileNetV2 loss"); plt.savefig("results/dl_loss.png", dpi=150)
