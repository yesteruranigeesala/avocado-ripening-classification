# Avocado Ripening Stage Classification: Report

## 1. Dataset Understanding
(Fill from results/class_distribution.png, lighting_stats.png, samples.png)
- Class counts and imbalance ratio:
- Skin colour shift (green -> purple/black), glare on shiny skin:
- Background / camera distance / lighting variation:

## 2. Model Selection Rationale
- Approach 1: interpretable, tiny, trains in seconds on CPU; CIELAB a* tracks green->red/purple; LBP captures smooth vs pebbled skin.
- Approach 2: MobileNetV2 learns colour and texture jointly, robust to lighting with augmentation, light enough for edge (depthwise separable convs).

## 3. Evaluation Metrics & Comparison
(Paste results/comparison.md; add confusion matrices.)
Primary metric: macro-F1 (class imbalance). Precision matters for not passing hard fruit as ready; recall for not missing ripe fruit.

## 4. Answers to End-of-Assignment Questions
**Q1. In production, how does your model perform?** Use YOUR measured latency (results/*_metrics.json: latency_ms_cpu / latency_ms_gpu). Discuss errors between adjacent stages (breaking vs ripe) under glare/shadows.

**Q2. Drawbacks?** Colour-only ambiguity (some varieties stay green when ripe); single-view camera misses bruises; continuous boundary between late breaking and early ripe causes label noise.

**Q3. What would you improve with more time?** NIR/tactile firmness sensing, multi-view rotating rollers, hyperspectral imaging, INT8 quantisation (ONNX/TensorRT), more varied data.

**Q4. Where did you spend the most time?** Be honest: give your own split (feature/colour-space tuning, data cleaning, benchmarking).
