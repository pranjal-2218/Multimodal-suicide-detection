# 🧠 Multimodal Suicide Detection & Risk Assessment System

An end-to-end multimodal deep learning pipeline that combines textual and acoustic cues to perform suicide risk classification. Built on top of **XLM-RoBERTa** and **OpenAI Whisper**, the model leverages dynamic gated feature fusion alongside a multi-task objective to ensure robust prediction across multilingual datasets (e.g., English, Hindi, Hinglish).

---

## 📌 Features

* **Multimodal Architecture**: Fuses textual representations from `xlm-roberta-base` and audio features from `openai/whisper-small`.
* **Gated Fusion Module**: Dynamically balances audio vs. text representations per-dimension using a learned sigmoid gating mechanism.
* **Multi-Task Optimization**: Trains combined auxiliary classification heads for both text and audio alongside the primary fusion head.
* **Stratified $K$-Fold Cross Validation**: Evaluates cross-validation performance with dynamic early stopping and checkpointing.
* **Automated Analytical PDF Reports**: Uses `reportlab` to generate comprehensive evaluation PDFs containing ROC/PR curves, confusion matrices (CMs), and dynamic gate distribution plots.

---

## 🏗️ Architecture Overview

```text
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
