# Google Teachable Machine — Training Evidence (Deliverable 5)

## 1. What the deployed model is

| | |
|---|---|
| Deployed files | `model/model.json`, `model/weights.bin` (5.85 MB), `model/metadata.json` |
| Model name in metadata | `SonicSentinel_GTM_v10` |
| Training run that produced it | **`gtm-v9-tm-finetuned`** — same training timestamp, `2026-09-28T17:35:28Z` |
| Architecture | Teachable Machine audio network: speech-commands **18w BROWSER_FFT** base (4 × Conv2D/MaxPooling2D, Dense 2000) + `NewHeadDense` softmax |
| Base model | `https://storage.googleapis.com/tfjs-models/tfjs/speech-commands/v0.3/browser_fft/18w/` |
| Labels | 17 — the 10 mandatory categories plus 7 optional ones, in the same order as `classes.json` |
| Export format | TensorFlow.js / Teachable Machine audio format |
| Runs where | **In the browser**, via TensorFlow.js. The server never runs it during normal operation. |

Full per-run evidence for every training run is in [`training_evidence/`](training_evidence/) —
`summary.md`, `metrics.json`, `model_comparison.csv`, `sample_predictions.csv`, `training_history.csv`
and confusion-matrix images for each version.

---

## 2. Training configuration

The model is built and trained by `scripts/train_gtm_model.py`, which reproduces the Teachable Machine
audio pipeline on the project's own dataset.

**Input representation** — identical to what `static/js/gtm.js` computes in the browser, so the training
input and the live input cannot drift apart:

> 43 × 232 spectrogram of the loudest ~1 s of each 2-second segment, 44.1 kHz, FFT 2048, hop 1024,
> 0–5 kHz band, z-score normalised.

**Hyperparameters (run `gtm-v9-tm-finetuned`)**

| Setting | Value |
|---|---|
| Transfer epochs (frozen base) | 50 |
| Fine-tune max epochs (end to end) | 30 |
| Fine-tune learning rate | 3e-4 |
| Early-stopping patience | 8 |
| Random seed | 42 |
| Label smoothing | 0.05 |
| Augmentation | time shift ±10 frames · SpecAugment (1 frequency + 1 time mask) · live-microphone rendering of the training recordings |
| Critical-class weights | none in this run (`{}`) — earlier runs used them, see §5 |

**Environment:** Python 3.14.7, PyTorch 2.14.0+cpu, Windows 11.

---

## 3. Training sample counts

| Split | Recordings | Windows fed to the network |
|---|---:|---:|
| Train | 13,443 | **33,252** |
| Validation | 1,810 | — |
| Test | 1,803 | — |

Train recordings include the augmented copies; validation and test are originals only.
Per-class recording counts for the underlying dataset are in
`07 - Dataset Metadata/dataset_statistics.json` and `dataset_report.md`.

### Background Noise samples

Background Noise is the largest single class in the evaluation data and carries **550 test recordings** —
more than any other class. Teachable Machine requires a Background Noise class, and this project keeps it
as a full first-class category rather than a filler class: it is what stops ambience from being escalated
into an alert. Its behaviour is covered by
`tests/test_alert_engine.py::test_background_noise_escalates_only_above_noise_limit` and
`test_background_second_class_is_not_an_overlap`.

---

## 4. Results of the deployed run (`gtm-v9-tm-finetuned`)

**Test set, recording level, measured on the exported model files**

| Accuracy | Precision (macro) | Recall (macro) | Macro F1 | Min critical recall |
|---|---|---|---|---|
| 0.679 | 0.657 | 0.824 | 0.714 | 0.622 |

**Class-wise performance (test)**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Machinery Fault | 0.644 | 0.844 | 0.731 | 45 |
| Glass Breaking | 0.847 | 0.980 | 0.909 | 51 |
| Alarm or Siren | 0.820 | 0.846 | 0.833 | 156 |
| Vehicle Horn | 0.778 | 0.835 | 0.805 | 109 |
| Animal Sound | 0.796 | 0.699 | 0.745 | 246 |
| Gunshot | 0.669 | 0.830 | 0.741 | 100 |
| Panic Scream | 0.688 | 0.978 | 0.807 | 45 |
| Aggression | 0.452 | 0.844 | 0.589 | 45 |
| Person Asking for Help | 0.368 | 0.867 | 0.517 | 45 |
| Drone Sound | 0.714 | 1.000 | 0.833 | 45 |
| Drilling or Grinder Sound | 0.542 | 0.621 | 0.579 | 124 |
| Fireworks | 0.667 | 0.863 | 0.752 | 51 |
| Vehicle Backfire | 0.606 | 0.956 | 0.741 | 45 |
| Explosion | 0.337 | 0.622 | 0.438 | 45 |
| Door Impact | 0.588 | 0.839 | 0.691 | 56 |
| Graffiti Spray | 0.818 | 1.000 | 0.900 | 45 |
| Background Noise | 0.826 | 0.380 | 0.521 | 550 |

**Models compared in this run** — the two ways a Teachable Machine audio model can be trained:

| Model | Val accuracy | Val macro F1 | Test accuracy | Test macro F1 | Min critical recall | Selected |
|---|---|---|---|---|---|---|
| Teachable Machine transfer (frozen base + new softmax layer — what the TM website trains) | 0.604 | 0.626 | 0.599 | 0.629 | 0.510 | |
| Teachable Machine network fine-tuned end to end | 0.694 | 0.725 | 0.680 | 0.715 | 0.622 | **yes** |

Training the output layer only — exactly what the Teachable Machine website does — reaches 59.9% test
accuracy. Fine-tuning the whole network on the same data reaches 68.0%. The fine-tuned model is the one
shipped.

**Live-microphone path** (raw audio, no noise reduction or trimming):

| Split | Accuracy | Macro F1 |
|---|---|---|
| validation | 0.699 | 0.720 |
| test | 0.691 | 0.725 |

---

## 5. Retraining history and training observations

Six training runs were kept. Every one has its own folder under `training_evidence/`.

| Version | Date | Classes | Train windows | Test acc | Macro F1 | Min critical recall | Class weights used |
|---|---|---:|---:|---:|---:|---:|---|
| gtm-v4-tm-finetuned | 2026-09-26 | 10 | 12,290 | 0.865 | 0.871 | 0.911 | Glass Breaking 1.5, Aggression 1.1 |
| gtm-v5-tm-transfer | 2026-09-27 | 17 | 25,690 | 0.693 | 0.717 | 0.578 | Glass Breaking 1.1, Gunshot 1.1, Panic Scream 1.25, Help 1.1, Explosion 2.5 |
| gtm-v6-tm-finetuned | 2026-09-27 | 17 | 25,690 | 0.801 | 0.821 | 0.760 | Glass Breaking 1.25, Aggression 1.25, Explosion 1.1 |
| gtm-v7-tm-transfer | 2026-09-28 | 37 | 49,227 | 0.574 | 0.497 | 0.420 | Glass Breaking 1.25, Gunshot 1.1, Panic Scream 1.1, Explosion 3.0 |
| gtm-v8-tm-finetuned | 2026-09-28 | 37 | 49,227 | 0.683 | 0.593 | 0.710 | Glass Breaking 2.5, Aggression 2.0, Help 1.5, Explosion 1.5 |
| **gtm-v9-tm-finetuned** | **2026-09-28** | **17** | **33,252** | **0.679** | **0.714** | **0.622** | **none — deployed** |

### What each retraining taught us

1. **v4 → v5/v6: going from 10 to 17 classes cost around 6 points of accuracy.** v4 scored 0.865 on 10
   classes. Adding the 7 optional SRS categories dropped the fine-tuned model to 0.801 (v6). This is
   expected — the optional categories (Fireworks, Vehicle Backfire, Explosion, Door Impact) are all short
   broadband transients that overlap acoustically with Gunshot and Glass Breaking.
2. **Transfer-only training is never competitive.** v5 (transfer, 0.693) vs v6 (fine-tuned, 0.801) on the
   same data; v7 (transfer, 0.574) vs v8 (fine-tuned, 0.683) on the same data. Freezing the speech-commands
   base costs 10–11 points every time. That base was trained on spoken words, not on impacts and alarms.
3. **v7/v8: 37 classes was a mistake and was rolled back.** Expanding to 37 fine-grained classes collapsed
   macro F1 to 0.497 (transfer) / 0.593 (fine-tuned) and dropped minimum critical recall to 0.420. The class
   list was reverted to 17 for v9.
4. **Aggressive class weighting stopped helping once the dataset was rebalanced.** v8 used weights up to
   2.5. v9 uses none and still reaches a better macro F1 (0.714 vs 0.593), because the dataset itself was
   rebalanced and augmented between the two runs.
5. **Background Noise recall is the known weak point (0.380).** The model is precise on it (0.826) but
   under-recalls it — it prefers to guess an event class. In the product this is the safe direction to fail:
   a false event still has to pass the alert rules, which require confidence, top-two margin, model
   agreement and consecutive detections before anything escalates, and Background Noise generated
   **0 alerts** across the whole 100-recording comparison set.

### Incorrect classifications — largest confusions (test set, run v9)

| Actual | Predicted as | Count | Share of that class |
|---|---|---:|---:|
| Background Noise | Drilling or Grinder Sound | 57 | 10% |
| Background Noise | Person Asking for Help | 56 | 10% |
| Background Noise | Animal Sound | 37 | 7% |
| Background Noise | Aggression | 35 | 6% |
| Background Noise | Explosion | 31 | 6% |
| Background Noise | Door Impact | 26 | 5% |
| Background Noise | Gunshot | 23 | 4% |
| Drilling or Grinder Sound | Background Noise | 20 | 16% |
| Background Noise | Vehicle Horn | 18 | 3% |
| Background Noise | Vehicle Backfire | 16 | 3% |
| Animal Sound | Background Noise | 14 | 6% |
| Animal Sound | Alarm or Siren | 13 | 5% |

The dominant error is **Background Noise mistaken for an event class**, and the second is the
**Drilling or Grinder Sound / Background Noise** pair in both directions — both are sustained, broadband
and low-contrast. Confusion-matrix images for every run are in `training_evidence/<version>/`.

Where the two models disagree on the 100-recording comparison set, and what the fused decision did about
it, is listed recording by recording in
`13 - Model Comparison Report/model_prediction_and_confidence_comparison_summary.md`.

---

## 6. Integration evidence

**Export check.** The exported `model.json` / `weights.bin` were re-run with the independent NumPy
implementation in `integration/inference.py`:

> max |probability difference| to the trained network = **1.61e-06** over **4,651** validation windows.

The exported files therefore compute the same thing the trained network did.

**Browser / Python parity.** The browser path and the Python path are held to the same input by tests:

| Test | What it proves |
|---|---|
| `tests/test_gtm.py::test_browser_gtm_path_matches_python_pipeline` | `static/js/gtm.js` and `gtm_model/pipeline.py` produce the same input |
| `tests/test_gtm.py::test_python_and_javascript_spectrograms_match` | the two spectrogram implementations agree |
| `tests/test_gtm.py::test_python_and_javascript_pick_the_same_window` | both pick the same loudest ~1 s window |
| `tests/test_gtm.py::test_numpy_inference_matches_pytorch` | the exported model matches the trained one |
| `tests/test_gtm.py::test_exported_gtm_model_labels_and_shape` | 17 labels, correct output shape |
| `tests/test_gtm.py::test_gtm_label_mismatch_is_rejected` | a model whose labels differ from `classes.json` is refused |
| `tests/test_gtm.py::test_gtm_files_served_and_results_finalise_record` | `/gtm/model/<file>` is served and the browser result finalises the record |
| `tests/test_gtm.py::test_gtm_error_payload_falls_back_to_single_model` | if GTM fails the app degrades to the Python model instead of breaking |
| `tests/test_gtm.py::test_python_prediction_is_not_an_input_to_gtm_scores` | the two models are genuinely independent |

**Model source configuration.** Admin → Models → *Teachable Machine* selects where the model comes from:

* **Local files** (default) — the app serves `model/`. To replace the model, copy the three exported files in.
* **Hosted link** — paste a Teachable Machine share link (`https://teachablemachine.withgoogle.com/models/<id>/`).

Either way the labels must match `classes.json` exactly or the app rejects the model. The choice is written
to the settings and the audit trail.

**Screenshots.** Application-side GTM evidence — Admin → Models showing both models and the label check, and
a record showing both models' predictions side by side — is in
`03 - Complete Source Code/screenshots/`: `17_admin_models.png`, `03_record_result_both_models.png`,
`04_live_monitor.png`, `10_review_disagreement_record.png`.

---

## 7. Scope note

This model was trained with `scripts/train_gtm_model.py`, which rebuilds the Teachable Machine audio
architecture (speech-commands 18w BROWSER_FFT base + `NewHeadDense` head) and exports in Teachable Machine
format, rather than by uploading samples to the teachablemachine.withgoogle.com web interface. The reason is
in §4: the website trains the output layer only, which reaches 59.9% test accuracy on this dataset against
68.0% for end-to-end fine-tuning of the same architecture.

**A web-interface Teachable Machine project has not been created, so browser screenshots of the TM class
list and the TM training screen do not exist.** To produce them, `integration/export_gtm_samples.py` exports
per-class ~1 s training samples ready to feed into the website; the step-by-step procedure is in
`03 - Complete Source Code/gtm_model/README.md`.
