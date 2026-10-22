"""Approach 1: HSV + CIELAB colour histograms/stats + LBP texture -> SVM / Random Forest."""
import time

import cv2
import joblib
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def _hist(ch, bins, rng=(0, 256)):
    h = cv2.calcHist([ch], [0], None, [bins], list(rng)).flatten()
    return h / (h.sum() + 1e-8)


def extract_avocado_features(image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Cannot read {image_path}")
    img = cv2.resize(img, (128, 128))
    feats = []
    # HSV (hue 0-180 in OpenCV)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    feats += [_hist(hsv[..., 0], 18, (0, 180)), _hist(hsv[..., 1], 16), _hist(hsv[..., 2], 16)]
    # CIELAB: a* = green<->red axis
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    for c in range(3):
        feats.append(_hist(lab[..., c], 16))
    stats = [f(lab[..., c]) for c in range(3) for f in (np.mean, np.std)]
    stats += [f(hsv[..., c]) for c in range(3) for f in (np.mean, np.std)]
    feats.append(np.array(stats) / 255.0)
    # LBP uniform, P=8 -> 10 codes (0..9)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    h, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11))
    feats.append(h / (h.sum() + 1e-8))
    return np.concatenate(feats).astype(np.float32)


def build_matrix(paths):
    return np.stack([extract_avocado_features(p) for p in paths])


def train_classical(X_tr, y_tr, kind="svm"):
    if kind == "svm":
        pipe = make_pipeline(StandardScaler(), SVC(class_weight="balanced", probability=True))
        grid = {"svc__C": [1, 10, 100], "svc__gamma": ["scale", 0.01, 0.001]}
    else:
        pipe = make_pipeline(RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1))
        grid = {"randomforestclassifier__n_estimators": [200, 400],
                "randomforestclassifier__max_depth": [None, 20]}
    gs = GridSearchCV(pipe, grid, cv=5, scoring="f1_macro", n_jobs=-1)
    gs.fit(X_tr, y_tr)
    print(f"[{kind}] best params: {gs.best_params_}  cv f1_macro={gs.best_score_:.3f}")
    return gs.best_estimator_


def predict_latency_ms(model, paths, n=50):
    paths = paths[:n]
    t = time.perf_counter()
    for p in paths:
        model.predict(extract_avocado_features(p).reshape(1, -1))
    return (time.perf_counter() - t) / len(paths) * 1000


def save(model, path="models/classical.joblib"):
    joblib.dump(model, path)


def load(path="models/classical.joblib"):
    return joblib.load(path)
