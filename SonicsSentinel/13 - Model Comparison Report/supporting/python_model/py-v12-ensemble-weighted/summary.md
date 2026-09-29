# Python model py-v12-ensemble-weighted

* Algorithm: **Soft-voting ensemble (HistGradientBoosting + Random Forest + MLP) + calibrated class weighting**
* Trained: 2026-09-28T18:56:17Z with `python -m python_models.train`
* Features: 193 per 2.0 s segment, feature set `ssf-1.0` (hop 1.0 s)
* Classes: 17 — Machinery Fault, Glass Breaking, Alarm or Siren, Vehicle Horn, Animal Sound, Gunshot, Panic Scream, Aggression, Person Asking for Help, Drone Sound, Drilling or Grinder Sound, Fireworks, Vehicle Backfire, Explosion, Door Impact, Graffiti Spray, Background Noise
* Critical classes: Gunshot, Glass Breaking, Panic Scream, Aggression, Person Asking for Help, Explosion
* Training data: 68983 segments from 8405 train-split recordings; validation 1810 and test 1803 recordings
* Cross-validated macro F1: **0.885**

## Ensemble configuration

* Members: hist_gradient_boosting, random_forest, mlp
* Weights: [0.45, 0.35, 0.2] — weighted mean of class probabilities
* Critical-class weights tuned on the validation split:

| Class | Weight |
|---|---:|
| Machinery Fault | 0.75 |
| Glass Breaking | 1.5 |
| Alarm or Siren | 1.15 |
| Vehicle Horn | 2.87 |
| Animal Sound | 0.75 |
| Gunshot | 0.83 |
| Panic Scream | 1.15 |
| Aggression | 0.95 |
| Person Asking for Help | 0.71 |
| Drone Sound | 0.75 |
| Drilling or Grinder Sound | 1.05 |
| Explosion | 1.5 |

Effect of the weighting on the validation split:

| | Min critical recall | Macro F1 |
|---|---:|---:|
| before | 0.822 | 0.874 |
| after | 0.889 | 0.916 |

## Validation set (recording level)

| Accuracy | Precision (macro) | Recall (macro) | Macro F1 | Min critical recall | N |
|---|---|---|---|---|---|
| 0.915 | 0.901 | 0.891 | 0.920 | 0.889 | 1810 |

## Test set (recording level)

| Accuracy | Precision (macro) | Recall (macro) | Macro F1 | Weighted F1 | Min critical recall | N |
|---|---|---|---|---|---|---|
| 0.912 | 0.877 | 0.885 | 0.915 | 0.847 | 0.889 | 1803 |

## Class-wise performance (test)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Machinery Fault | 0.872 | 0.911 | 0.891 | 45 |
| Glass Breaking | 0.962 | 1.000 | 0.981 | 51 |
| Alarm or Siren | 0.922 | 0.910 | 0.916 | 156 |
| Vehicle Horn | 0.836 | 0.890 | 0.862 | 109 |
| Animal Sound | 0.834 | 0.736 | 0.782 | 246 |
| Gunshot | 0.988 | 0.810 | 0.890 | 100 |
| Panic Scream | 0.936 | 0.978 | 0.957 | 45 |
| Aggression | 0.820 | 0.911 | 0.863 | 45 |
| Person Asking for Help | 0.783 | 0.800 | 0.791 | 45 |
| Drone Sound | 0.978 | 1.000 | 0.989 | 45 |
| Drilling or Grinder Sound | 0.841 | 0.556 | 0.670 | 124 |
| Fireworks | 0.943 | 0.980 | 0.962 | 51 |
| Vehicle Backfire | 0.976 | 0.911 | 0.943 | 45 |
| Explosion | 0.642 | 0.956 | 0.768 | 45 |
| Door Impact | 0.793 | 0.821 | 0.807 | 56 |
| Graffiti Spray | 0.978 | 1.000 | 0.989 | 45 |
| Background Noise | 0.799 | 0.867 | 0.832 | 550 |

## Models compared

| Model | CV macro F1 | Val accuracy | Val macro F1 | Test accuracy | Test macro F1 | Min critical recall (test) | Selected |
|---|---|---|---|---|---|---|---|
| Random Forest | 0.825 | 0.842 | 0.848 | 0.835 | 0.841 | 0.778 |  |
| Histogram Gradient Boosting | 0.865 | 0.868 | 0.872 | 0.854 | 0.868 | 0.822 |  |
| Multi-layer Perceptron (neural network) | 0.845 | 0.852 | 0.858 | 0.846 | 0.852 | 0.800 |  |
| Soft-voting ensemble (HistGradientBoosting + Random Forest + MLP) + calibrated class weighting | 0.885 | 0.915 | 0.920 | 0.912 | 0.915 | 0.889 | yes |

## Confusion matrix (test, recording level)

Row = actual class, column = predicted class, in the class order above.

```
[[41, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4], [0, 51, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [3, 0, 142, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 9], [0, 0, 1, 97, 0, 0, 0, 0, 1, 0, 1, 0, 0, 3, 0, 0, 6], [1, 0, 7, 7, 181, 0, 2, 0, 1, 0, 1, 0, 0, 3, 1, 0, 42], [0, 0, 0, 0, 1, 81, 0, 0, 0, 0, 0, 2, 0, 3, 4, 0, 9], [0, 0, 0, 0, 0, 0, 44, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 41, 2, 0, 0, 0, 0, 0, 1, 0, 1], [0, 1, 0, 0, 0, 0, 0, 3, 36, 0, 0, 1, 0, 1, 1, 0, 2], [0, 0, 0, 0, 0, 0, 0, 0, 0, 45, 0, 0, 0, 0, 0, 0, 0], [2, 1, 4, 2, 1, 1, 0, 1, 0, 0, 69, 0, 0, 0, 0, 1, 42], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 50, 0, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 41, 2, 1, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 43, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 5, 46, 0, 4], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 45, 0], [0, 0, 0, 8, 34, 0, 0, 4, 6, 1, 10, 0, 0, 6, 4, 0, 477]]
```

## Files

* `../../../../08 - Python Model/py-v12-ensemble-weighted.joblib` — the saved model the application loads
* `../../../../08 - Python Model/py-v12-ensemble-weighted.json` — the full model card this summary is drawn from
* `metrics.json` — the card and its complete metrics block
* `model_comparison.csv` — every model that was trained and compared for this version
