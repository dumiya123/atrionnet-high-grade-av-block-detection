# AtrionNet — High-Grade AV Block Detection from ECG

AtrionNet is a deep learning system built to automatically detect high-grade atrioventricular (AV) blocks from 12-lead ECG recordings. It was developed as a final-year research project to address a real gap: automated, explainable ECG interpretation for conduction disorders that are often missed or misread.

The system goes beyond a simple classifier. It segments individual cardiac waves (P, QRS, T), measures PR intervals, identifies dissociated P-waves, and produces a structured clinical report — all from a single `.npy` ECG file.

---

## The Problem

AV blocks are conduction abnormalities between the atria and ventricles. High-grade variants (Type II Second-Degree and Third-Degree/Complete AV Block) are medical emergencies — they can cause syncope, cardiac arrest, and sudden death if not identified quickly.

The challenge is that:
- **P-wave detection is hard**. P-waves are small, overlapping, and easily buried in noise.
- **PR interval analysis requires precision**. A few milliseconds of drift changes the diagnosis.
- **Existing automated ECG tools** either oversimplify (threshold-based) or are black boxes with no clinical explanation.
- **Most deep learning work** on ECG focuses on AF or ST-elevation — AV block classification with wave-level explainability is largely unexplored.

This project targets exactly that gap.

---

## Architecture

The core model is `AtrionNetHybrid` (v5.0), a U-Net style architecture with:

- **Attentional Inception Encoder** — multi-scale 1D convolutions (kernels: 9, 19, 39) with Squeeze-and-Excitation attention to amplify weak P-wave signals across different temporal scales.
- **Dilated Convolutional Bridge** — replaces the original BiLSTM with stacked dilated convolutions (dilations: 1, 2, 4). This covers a large receptive field across the 625-sample bottleneck without the vanishing gradient instability that caused the BiLSTM to collapse on long sequences.
- **Skip-connected Decoder** — mirrors the encoder, reconstructing full-resolution (5000-sample) outputs with skip connections from each encoder level.
- **Three Output Heads** — heatmap (P-wave probability at each sample), width prediction, and binary segmentation mask.

A simpler `AtrionNetBaseline` (standard U-Net with plain Conv-BN-ReLU blocks) is included for ablation studies.

```
Input (12 leads × 5000 samples @ 500Hz)
  → AttentionalInception Enc1 (→64ch) → Pool
  → AttentionalInception Enc2 (→128ch) → Pool  
  → AttentionalInception Enc3 (→256ch) → Pool
  → Dilated CNN Bridge (256→512, dilations 1,2,4)
  → Decoder + Skip Connections
  → Heatmap / Width / Mask outputs (5000 samples)
```

---

## What It Actually Does

1. **Upload** a `.npy` ECG file (12-lead, 5000 samples at 500 Hz = 10 seconds)
2. **Preview** the Lead II signal before running inference
3. **Analyze** — the model segments waves, classifies AV block type, and scores confidence
4. **Report** — downloads a structured PDF with diagnosis, PR intervals, heart rate, and XAI heatmap overlay

Supported diagnoses:
- Normal Sinus Rhythm
- First-Degree AV Block (prolonged PR)
- Second-Degree AV Block Type I (Wenckebach / progressive PR lengthening)
- Second-Degree AV Block Type II (Mobitz II / sudden dropped QRS)
- Third-Degree AV Block (complete dissociation — P and QRS independent)

---

## Project Structure

```
AtrionNet_Implementation/
├── ml_component/
│   ├── src/
│   │   ├── modeling/         # AtrionNetHybrid + Baseline model definitions
│   │   ├── data_pipeline/    # LUDB dataset loader, augmentations, instance dataset
│   │   ├── inference/        # predictor.py — full inference + XAI + report generation
│   │   ├── engine/           # Training evaluator, mAP + F1 metrics
│   │   └── losses/           # Focal loss for heatmap, combined segmentation loss
│   ├── train.py              # Training entry point
│   ├── outputs/weights/      # Saved model checkpoints (.pth)
│   └── requirements.txt
├── backend/
│   └── main.py               # FastAPI server (endpoints: /preview, /analyze, /report)
├── frontend/
│   └── src/                  # React + Vite dashboard
└── tests/
```

---

## Setup

### Prerequisites
- Python 3.9+
- Node.js 18+
- CUDA-capable GPU (recommended for training; CPU works for inference)

### Install Python dependencies

```bash
cd ml_component
pip install -r requirements.txt
```

### Download the LUDB Dataset

```bash
python download_data.py
```

This fetches the [Lobachevsky University Electrocardiography Database (LUDB)](https://physionet.org/content/ludb/1.0.1/) from PhysioNet and saves it to `ml_component/data/raw/ludb/`.

### Train the model

```bash
cd ml_component
python train.py
```

Training runs for up to 150 epochs with early stopping (patience=25). The best checkpoint is saved to `outputs/weights/atrion_hybrid_best.pth`.

### Start the backend

```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Start the frontend

```bash
cd frontend
npm install
npm run dev
```

Then open `http://localhost:5173` in your browser.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Server + model status |
| `POST` | `/preview` | Preview ECG signal (no inference) |
| `POST` | `/analyze` | Full inference + report generation |
| `GET` | `/report/{id}` | Download generated PDF report |

All endpoints expect `.npy` files in the format `(12, 5000)` or `(5000, 12)`.

---

## Training Configuration

Key hyperparameters (from `train.py`):

| Parameter | Value |
|-----------|-------|
| Epochs | 150 |
| Early stopping patience | 25 |
| Batch size | 16 |
| Learning rate | 1e-4 |
| Heatmap confidence threshold | 0.45 |
| Min inter-peak distance | 60 samples (120ms) |
| Loss weights (heatmap/width/mask) | 2.0 / 1.0 / 1.0 |

---

## Explainability (XAI)

The system includes a gradient-based XAI heatmap that highlights which parts of the ECG signal most influenced the model's prediction. The frontend overlays this directly on the plotted waveform so the output is interpretable — not just a label.

Libraries used: `captum`, `grad-cam`, `shap`.

---

## Dataset

**LUDB — Lobachevsky University Electrocardiography Database**  
200 ECG records, expert-annotated at the wave level (P-onset, P-peak, P-offset, QRS boundaries, T boundaries). Sampling rate: 500 Hz.  
Source: [PhysioNet](https://physionet.org/content/ludb/1.0.1/)

---

## Stack

| Layer | Technology |
|-------|-----------|
| Model | PyTorch 2.0+ |
| ECG processing | wfdb, neurokit2, scipy |
| API | FastAPI + uvicorn |
| Frontend | React + Vite |
| Reports | ReportLab |
| XAI | captum, grad-cam, shap |

---

## License

This project is for academic and research purposes. Dataset usage is subject to PhysioNet's terms of access.
