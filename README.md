# 🧠 Multimodal Suicide Detection & Risk Assessment

A research prototype that classifies suicide risk from **text** and **speech audio** in **English and Hindi**. It combines a fine-tuned **XLM-RoBERTa** text encoder with a frozen **OpenAI Whisper** audio encoder through a learned **gated fusion** layer, trained with a multi-task objective and evaluated with **5-fold stratified cross-validation**.

> ⚠️ **Research prototype, not a clinical tool.** This model has been validated only with cross-validation on a small dataset (500 samples). It has **not** been tested on a separate held-out dataset and must not be used to assess a real person's safety or make care decisions. The accompanying research is not yet published.

---

## 📑 Table of Contents
1. [Results at a glance](#-results-at-a-glance)
2. [Architecture](#️-architecture)
3. [Repository structure](#-repository-structure)
4. [Dataset and audio files](#️-dataset-and-audio-files)
5. [Setup and how to run](#-setup-and-how-to-run)
6. [Training configuration](#-training-configuration)
7. [Detailed results](#-detailed-results)
8. [Limitations and honest notes](#️-limitations-and-honest-notes)
9. [Troubleshooting](#-troubleshooting)
10. [Tech stack](#-tech-stack)

---

## 🎯 Results at a glance

5-fold cross-validation, mean ± std across folds:

| Model head | Accuracy | F1 | ROC-AUC |
| --- | --- | --- | --- |
| Text only | 0.938 ± 0.037 | 0.938 ± 0.037 | n/a |
| Audio only | 0.818 ± 0.046 | 0.816 ± 0.043 | n/a |
| **Fusion (text + audio)** | **0.940 ± 0.035** | **0.940 ± 0.035** | **0.970 ± 0.019** |

The fusion head is supervised by, and scored against, the **text label** (see [Limitations](#️-limitations-and-honest-notes) for what this means).

---

## 🏗️ Architecture

```
                   ┌────────────────────────┐
Text Input  ───────► XLM-RoBERTa Encoder    ├──────► Text Logits (Auxiliary)
                   └──────────┬─────────────┘
                              │
                              ▼ [768d]
                     ┌──────────────────┐
                     │ Gated Fusion Head├─────────► Fusion Logits (Primary Target)
                     └──────────────────┘
                              ▲ [768d]
                              │
                   ┌──────────┴─────────────┐
Audio Input ───────► Frozen Whisper Encoder ├──────► Audio Logits (Auxiliary)
                   └────────────────────────┘
```

| Component | Details |
| --- | --- |
| Text encoder | `xlm-roberta-base`, fine-tuned; `[CLS]` representation (768-d), max 128 tokens |
| Audio encoder | `openai/whisper-small` encoder, **frozen**; audio converted to 16 kHz mono; first-token representation |
| Audio projection | Linear layer mapping Whisper features into the 768-d text space |
| Gated fusion | `g = sigmoid(Linear([text ‖ audio]))`; `fused = g·text + (1−g)·audio`; then dropout + LayerNorm + linear classifier |
| Text head | Linear classifier |
| Audio head | MLP: Linear → ReLU → Dropout(0.1) → Linear, hidden size 512 |

**Loss:** `0.5 · (CE_text + CE_audio) + 0.5 · CE_fusion`

---

## 📂 Repository structure

| File | Description |
| --- | --- |
| `final_ai_model_.ipynb` | Complete pipeline: dataset, model, training, cross-validation, plots, PDF report generation |
| `final_suicidal_dataset.csv` | Dataset index (500 rows): text, labels, languages and audio file names |
| `final_suicidal_report.pdf` | Cross-validation report: per-fold and aggregated results |
| `final_suicidal_summary.pdf` | Per-epoch validation metrics and summary figures |

The **audio files (~1 GB)** are not stored in this repository; see below.

---

## 🗂️ Dataset and audio files

### CSV format

`final_suicidal_dataset.csv` has 500 rows and these columns:

| Column | Description |
| --- | --- |
| `text` | Input text (English or Hindi) |
| `text_lang` | `english` or `hindi` (250 each) |
| `text_label` | `0` or `1` (250 each) |
| `audio` | Audio file name, e.g. `audio100.wav` |
| `audio_lang` | `english` or `hindi` (250 each) |
| `audio_label` | `0` or `1` (249 / 251) |

The CSV references **406 unique `.wav` files**; some files are used by more than one row.

### Downloading the audio

📥 **Audio files:** [Google Drive folder](https://drive.google.com/drive/folders/1y-_uXBG6HZI-dOqda5La-panVy952ftz?usp=sharing)

The `audio` column holds **bare file names**, which the notebook resolves relative to the **current working directory**. Put all `.wav` files in the folder the notebook runs from (in Google Colab this is `/content/`).

> ⚠️ **If an audio file cannot be found or read, the notebook silently replaces it with 1 second of silence and keeps running.** The audio results are only valid if every file loads, so run the [audio check](#2-verify-the-audio-files) before training.

### How text and audio relate in this dataset

Each row pairs a text sample with an audio sample, but their labels are independent: `text_label` and `audio_label` match in only **49%** of rows, and `text_lang` and `audio_lang` match in about **50%**. The audio and text in a row should therefore be treated as separately labelled inputs, not as the same utterance. This is why the report includes a "Text vs Audio Agreement" matrix.

---

## 🚀 Setup and how to run

### 1. Install dependencies

Python 3.9+ is recommended. A GPU is used automatically when available; training on CPU works but will be slow.

```bash
git clone https://github.com/pranjal-2218/Multimodal-suicide-detection.git
cd Multimodal-suicide-detection
pip install torch torchaudio transformers scikit-learn pandas numpy matplotlib reportlab jupyter
```

The first run downloads `xlm-roberta-base` and `openai/whisper-small` from Hugging Face, so an internet connection is needed.

### 2. Verify the audio files

Place the downloaded `.wav` files in your working directory, then run this in a notebook cell (or Python) before training:

```python
import os, pandas as pd

df = pd.read_csv("final_suicidal_dataset.csv")
missing = sorted({a for a in df["audio"] if not os.path.exists(a)})
print(f"Missing audio files: {len(missing)}")   # must print 0
print(missing[:10])
```

If this prints anything other than `0`, fix the file locations first.

### 3. Set the dataset path

In the config cell of `final_ai_model_.ipynb`, change:

```python
CSV_PATH = "/content/new synthetic_diaz - Sheet1.csv"   # <-- change this
```

to the location of `final_suicidal_dataset.csv`, for example `CSV_PATH = "final_suicidal_dataset.csv"`.

### 4. Run

Open `final_ai_model_.ipynb` and run all cells in order. The notebook trains and evaluates all 5 folds and writes everything to a `results/` folder:

- `results/checkpoints/fold{N}_best.pt`: best model per fold
- confusion matrices, gate histograms, ROC/PR curves (PNG)
- `results/final_report.pdf`: the generated cross-validation report

---

## ⚙️ Training configuration

| Setting | Value |
| --- | --- |
| Cross-validation | 5 folds, stratified on `text_label`, shuffled |
| Optimizer | AdamW, learning rate 2e-5 |
| Batch size | 8 |
| Max epochs | 15 |
| Early stopping | patience 4, on validation fusion F1; best checkpoint reloaded |
| Multi-task weight (α) | 0.5 |
| Max text length | 128 tokens |
| Audio sample rate | 16 kHz |
| Whisper encoder | frozen |
| Random seed | 42 |

---

## 📊 Detailed results

**Cross-validation, mean ± std across 5 folds**

| Head | Accuracy | Recall | F1 | Precision |
| --- | --- | --- | --- | --- |
| Text | 0.938 ± 0.037 | 0.940 ± 0.051 | 0.938 ± 0.037 | 0.937 ± 0.033 |
| Audio | 0.818 ± 0.046 | 0.805 ± 0.066 | 0.816 ± 0.043 | 0.833 ± 0.066 |
| Fusion | 0.940 ± 0.035 | 0.944 ± 0.046 | 0.940 ± 0.035 | 0.937 ± 0.033 |

**Fusion head**

| Metric | Value |
| --- | --- |
| ROC-AUC | 0.9698 ± 0.0185 |
| PR-AUC | 0.9652 ± 0.0211 |
| Gate mean (0 = audio, 1 = text) | 0.5250 ± 0.0072 |
| Gate variance | 0.0143 ± 0.0011 |
| Accuracy per fold | 0.91 · 0.90 · 0.93 · 0.97 · 0.99 |

**Overall fusion confusion matrix** (summed over folds, n = 500)

|  | Predicted 0 | Predicted 1 |
| --- | --- | --- |
| **Actual 0** | 234 | 16 |
| **Actual 1** | 14 | 236 |

**Text head by language** (summed over folds): English 234/250 correct (93.6%), Hindi 235/250 correct (94.0%).

Per-fold figures and curves are in [`final_suicidal_report.pdf`](final_suicidal_report.pdf), and per-epoch logs are in [`final_suicidal_summary.pdf`](final_suicidal_summary.pdf).

---

## ⚠️ Limitations and honest notes

- **No independent test set.** All reported numbers come from cross-validation on the same 500 samples. Early stopping and checkpoint selection use the validation fold that is then reported, so scores may be optimistic. Performance on unseen data is unknown.
- **Small dataset.** 500 samples, with some repeated audio files (406 unique files across 500 rows) and a few repeated texts, so the same item may appear in both training and validation folds.
- **Fusion mostly reflects the text signal.** The fusion head is trained and scored against the text label, and its accuracy is nearly identical to the text head (0.940 vs 0.938). Because text and audio labels are independent in this dataset, these results do not show that adding audio improves text-based detection.
- **Audio-only performance is lower** (0.818 accuracy) and evaluated against the separate audio label.
- **Not a clinical instrument.** It has not been validated for real-world screening or diagnosis.

---

## 🛠️ Troubleshooting

| Problem | Fix |
| --- | --- |
| `FileNotFoundError` for the CSV | Update `CSV_PATH` in the config cell (step 3) |
| Audio results look poor or odd | Run the audio check; missing files become silence without any error |
| CUDA out of memory | Lower `BATCH_SIZE` in the config cell |
| Very slow training | Use a GPU runtime (e.g., Colab GPU) |
| Model download fails | Check your internet connection; the models download from Hugging Face on first run |

---

## 🧰 Tech stack

`PyTorch` · `torchaudio` · `Hugging Face Transformers` · `scikit-learn` · `pandas` · `NumPy` · `Matplotlib` · `ReportLab`

---

## 👤 Author

**Pranjal** · [@pranjal-2218](https://github.com/pranjal-2218)
