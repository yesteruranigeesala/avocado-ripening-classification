# Avocado Ripening Stage Classification: Report

GreenPod Labs Data Science Assignment 2

## 1. Understanding of the Dataset

The dataset has 4 classes and 1090 images in total (train and test folders merged, then re-split 70/15/15 with stratification).

| Class | Images |
|---|---|
| blackspot | 206 |
| canker | 201 |
| fresh | 314 |
| grenning | 369 |

**Class imbalance:** max/min ratio is 1.84 (grenning is largest, canker is smallest). This is mild, so I used a stratified split and macro-averaged metrics instead of resampling.

**Outliers and biases found in EDA:**

- **Image size bias:** blackspot and canker images are all 800x800. Fresh images vary a lot (mean about 315x347, std about 80-90). Size alone can leak the class.
- **Lighting bias:** mean brightness is much higher for fresh (165) than blackspot (110) and canker (102).
- **Glare bias:** glare appears only in fresh images (mean 0.21), not in blackspot or canker.

These shortcuts mean a model could score high by learning the background or lighting rather than the fruit. To reduce this, all images are resized to one fixed size before modelling.

## 2. Understanding of Model Selection

I compared two approaches on the same split.

**Approach 1: Classical image processing + ML (SVM, Random Forest)**

- Features: HSV and CIELAB colour histograms and statistics, plus LBP texture.
- Why: ripening is mainly a colour and surface-texture change, so hand-crafted colour/texture features fit the problem. The dataset is small (1090 images), which suits SVM and RF. They are fast, need no GPU and are easy to explain.
- SVM handles high-dimensional features well. RF is robust and gives a second classical baseline.

**Approach 2: Deep learning (MobileNetV2 transfer learning)**

- Head training first, then fine-tuning the last blocks.
- Why: a small dataset needs pretrained weights, and MobileNetV2 is lightweight enough for real-time use on CPU or mobile in a sorting line. It can also learn texture patterns (spots, cracks) that colour features miss.

## 3. Understanding of Evaluation Metrics

- **Accuracy:** overall correctness, but it can hide weak classes when classes are imbalanced.
- **Macro precision, recall and F1:** each class counts equally, so smaller classes (canker, blackspot) are not ignored. **Macro-F1 was my main metric.**
- **Recall** matters because missing a defective avocado (canker, blackspot) is costlier than a false alarm.
- **CPU latency (ms/image):** shows whether the model is practical in production.

### Results (test set)

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | CPU ms/img |
|---|---|---|---|---|---|
| MobileNetV2 | 0.9939 | 0.9919 | 0.9919 | 0.9918 | 33.6 |
| SVM | 0.9878 | 0.9848 | 0.9833 | 0.9836 | n/a |
| Random Forest | 0.9756 | 0.9683 | 0.9669 | 0.9671 | n/a |

**Best model by macro-F1: MobileNetV2.** It leads on every metric. The SVM is a close second and is a good fallback if no deep learning runtime is available.

## 4. Example Run

Command (from the repo root):

```
python main_demo.py data/raw/fresh/<some_image>.jpg
```

The script loads the saved model, preprocesses the image and prints the predicted ripening stage with its confidence. (Paste the actual output or a screenshot of one run here.)

## 5. Assignment Questions

**1. In production, how does your model perform?**
MobileNetV2 reaches macro-F1 0.9918 at about 33.6 ms per image on CPU, which is fast enough for real-time sorting. The SVM (0.9836) is a lightweight backup. In real conditions, changes in lighting, camera angle or avocado variety can lower accuracy, so the model needs monitoring and periodic retraining.

**2. What is the drawback of your model?**
The dataset is small (1090 images) and has clear lighting, glare and image-size differences between classes, so the 0.99 scores may be partly inflated by these shortcuts and may not hold on new data. The classical features are weak under changing light. The classes are mildly imbalanced.

**3. If given more time, what would you improve?**
Collect more varied data (different lighting, backgrounds, varieties), use stronger augmentation, add k-fold cross-validation, check the confusion matrix for hard classes, explain predictions with Grad-CAM, and export the model to ONNX or TFLite for mobile deployment.

**4. Where did you spend the most time?**
Setting up the data (downloading from Drive, merging the train/test folders, fixing the class folder structure) and debugging the Colab runtime and training pipeline.

## Links

- GitHub: https://github.com/yesteruranigeesala/avocado-ripening-classification
- Colab: https://colab.research.google.com/drive/1BlBnd8Z9yXzJOK1Nbqaw5nelLd2AXvTv?usp=sharing
