"""EDA: class distribution, image size/brightness stats, sample grid."""
import sys; sys.path.insert(0, ".")
from collections import Counter
import cv2, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.dataset import list_images, make_splits

paths, labels, classes = list_images("data/raw")
counts = Counter(labels)
df = pd.DataFrame({"stage": classes, "count": [counts[i] for i in range(len(classes))]})
print(df); print("Imbalance ratio (max/min):", df["count"].max() / df["count"].min())
plt.figure(figsize=(7, 3.5)); plt.bar(df.stage, df["count"], color="forestgreen")
plt.title("Class distribution"); plt.tight_layout(); plt.savefig("results/class_distribution.png", dpi=150)

rows = []
rng = np.random.default_rng(0)
for i in rng.choice(len(paths), min(400, len(paths)), replace=False):
    im = cv2.imread(paths[i])
    if im is None: continue
    h, w = im.shape[:2]
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    rows.append(dict(cls=classes[labels[i]], h=h, w=w, brightness=g.mean(),
                     glare=(g > 245).mean(), corner_bg=im[:20, :20].mean()))
s = pd.DataFrame(rows)
print(s.groupby("cls")[["h", "w", "brightness", "glare"]].agg(["mean", "std"]).round(2))
s.groupby("cls")[["brightness", "glare"]].mean().plot(kind="bar", subplots=True, figsize=(7, 5), legend=False)
plt.tight_layout(); plt.savefig("results/lighting_stats.png", dpi=150)

fig, ax = plt.subplots(len(classes), 5, figsize=(10, 2 * len(classes)))
for r, c in enumerate(classes):
    idx = [i for i, l in enumerate(labels) if l == r][:5]
    for k in range(5):
        ax[r, k].axis("off")
        if k < len(idx):
            ax[r, k].imshow(cv2.cvtColor(cv2.imread(paths[idx[k]]), cv2.COLOR_BGR2RGB))
        if k == 0: ax[r, k].set_title(c, fontsize=9, loc="left")
plt.tight_layout(); plt.savefig("results/samples.png", dpi=120)
make_splits(); print("Saved splits to data/processed/splits.json")
