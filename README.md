# Avocado Ripening Stage Classification

Two approaches compared on the same stratified 70/15/15 split:
1. **Classical**: HSV + CIELAB colour histograms/stats + LBP texture -> SVM / Random Forest
2. **Deep learning**: MobileNetV2 transfer learning (head training, then fine-tune last blocks)

## Setup
```bash
pip install -r requirements.txt
```
Download the dataset (Drive link in the assignment) and extract so you have
`data/raw/<class_name>/*.jpg` (e.g. underripe, breaking, ripe, overripe).
Class names are auto-detected from folder names.

## Run (from repo root)
```bash
python scripts/01_eda.py              # class distribution, lighting stats, sample grid, creates split
python scripts/02_train_classical.py  # SVM + RF, saves models/classical.joblib
python scripts/03_train_dl.py         # MobileNetV2, saves models/mobilenetv2.pth
python scripts/04_compare.py          # comparison table (results/comparison.md)
python main_demo.py data/raw/ripe/some_image.jpg
```
All plots/metrics land in `results/`.

## Google Colab
```python
!git clone <your-repo-url> && %cd avocado_ripening_classification
!pip install -q -r requirements.txt
# upload/unzip dataset into data/raw, enable GPU runtime, then run the scripts with !python ...
```
Set sharing to "Anyone with the link can view".

## Results
Paste `results/comparison.md` here after running.
