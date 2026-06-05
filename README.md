# AtrionNet — High-Grade AV Block Detection from 12-Lead ECG

> **Final Year Research Project** | Deep Learning · ECG Signal Processing · Clinical Decision Support

AtrionNet is an end-to-end deep learning system for detecting and classifying atrioventricular (AV) blocks from 12-lead ECG recordings. It goes beyond a simple classifier — it detects individual cardiac waves (P, QRS, T), measures PR/RR intervals, identifies dissociated P-waves, quantifies confidence from the model's own probability outputs, and generates a structured downloadable clinical report.

Built as a final-year research project, the goal was to produce something that could realistically sit alongside a cardiologist's workflow rather than just hitting benchmark numbers.

---

## Table of Contents

1. [The Problem](#the-problem)
2. [Research Gap](#research-gap)
3. [System Overview](#system-overview)
4. [Architecture](#architecture)
5. [AV Block Classification Logic](#av-block-classification-logic)
6. [Explainability (XAI)](#explainability-xai)
7. [Project Structure](#project-structure)
8. [Dataset](#dataset)
9. [Setup & Installation](#setup--installation)
10. [Training](#training)
11. [Running the Full System](#running-the-full-system)
12. [API Reference](#api-reference)
13. [Frontend Dashboard](#frontend-dashboard)
14. [Evaluation Metrics](#evaluation-metrics)
15. [Technology Stack](#technology-stack)

---

## The Problem

Atrioventricular (AV) blocks are electrical conduction abnormalities between the heart's atria and ventricles. When functioning normally, every P-wave (atrial beat) should be followed by a QRS complex (ventricular response) with a PR interval of 120–200 ms. AV blocks disrupt this relationship in different ways — from simple PR prolongation in First-Degree block to complete electrical dissociation in Third-Degree (Complete) block.

High-grade variants are medical emergencies. Third-Degree AV block means the ventricles are beating independently of the atria — the heart is essentially running on two disconnected clocks. Without treatment (typically a pacemaker), this leads to hemodynamic collapse.

The clinical challenge comes from detecting and classifying these blocks automatically:

- **P-waves are small and hard to detect.** They're typically one-tenth the amplitude of a QRS complex and can be buried in baseline noise or overlap with T-waves in fast heart rates.
- **PR interval measurement requires sample-level precision.** At 500 Hz, a 20 ms PR interval change is only 10 samples. Errors at this resolution produce wrong diagnoses.
- **Dissociated P-waves (Third-Degree) look different.** They're not attached to QRS complexes and occur at a different rate, making them harder to find with standard rule-based algorithms.
- **Hospital ECG systems oversimplify.** Most automated ECG machines use threshold-based algorithms from the 1980s. They produce results like "abnormal ECG" with no wave-level breakdown.

---

## Research Gap

There's a fair amount of deep learning work on ECG, but most of it targets atrial fibrillation (AF) detection or myocardial infarction (ST elevation). AV block classification — specifically the kind that involves wave-level segmentation and explicit PR interval analysis — is much less explored.

The specific gaps this project addresses:

1. **Most existing models are classifiers only.** They output a label (e.g., "AV Block") but don't show *which waves* triggered the decision or *what the intervals measured*. That's a problem for clinical trust and regulatory use.

2. **Explainability is bolted on, not built in.** XAI in most ECG papers means running GradCAM post-hoc on a CNN. Here, the model's heatmap output head is a core part of the architecture — it's trained directly to predict P-wave probability at every sample.

3. **Long-sequence BiLSTM instability.** Early versions of this system used a BiLSTM bottleneck over a 625-step sequence. This caused training instability and vanishing gradients. The final architecture replaces this with a Dilated Convolutional Bridge, which achieves equivalent receptive field coverage stably.

4. **No wave-level detection + classification in one pipeline.** This system does both: it detects P, QRS, and T waves and uses their spatial relationships to classify the block type — in a single inference call.

---

## System Overview

The system has three layers:

```
User (Browser)
    ↕  HTTP
FastAPI Backend  (port 8000)
    ↕  PyTorch inference
AtrionNetHybrid Model + Rule-Based Post-Processor
```

**User flow:**
1. User uploads a `.npy` ECG file from the dashboard
2. The frontend previews the Lead II waveform before any analysis
3. User clicks Analyze — backend loads the file, runs inference, classifies the AV block, computes intervals, generates the XAI heatmap, and builds a PDF report
4. Results appear on screen with the waveform annotated with wave boundaries, confidence score, and clinical metrics
5. User can download the PDF clinical report

<!-- 📸 SCREENSHOT: System overview / dashboard homepage -->
> **[Add screenshot: Dashboard home page showing the upload interface]**

---

## Architecture

### AtrionNetHybrid (v5.0)

The core model is a U-Net style encoder-decoder with three major components:

#### 1. Attentional Inception Encoder

Each encoder level uses an `AttentionalInception` block — an Inception module with multi-scale parallel convolutions followed by a Squeeze-and-Excitation (SE) attention gate:

- **Bottleneck (1×1 conv)** reduces channels before the multi-scale convolutions
- **Three parallel 1D convolutions** with kernel sizes 9 (18ms), 19 (38ms), and 39 (78ms) capture P-wave features at different temporal scales
- **Residual connection** preserves original features
- **SE Attention** (`AttentionBlock1D`) performs global average pooling then learns per-channel weights — this helps the model focus on P-wave frequency bands and suppress noise

The SE attention is important for P-wave detection because P-waves occupy a specific frequency range (0.5–15 Hz) and are easily dominated by higher-amplitude QRS features without explicit gating.

```
Input → Bottleneck → [Conv9 | Conv19 | Conv39 | Residual] → Concat → SE Attention → BN → ReLU
```

Three encoder levels progressively increase channel depth (12 → 64 → 128 → 256) while halving the sequence length at each max-pool.

#### 2. Dilated Convolutional Bridge

After the third encoder pool, the sequence is 625 samples (5000 / 2³). Early versions of this model used a BiLSTM here — 625 time steps caused frequent vanishing gradient issues and the model would collapse to predicting zero P-waves after a few epochs.

The Dilated CNN Bridge replaces this with three stacked dilated convolutions:

| Layer | Channels | Dilation | Effective Receptive Field |
|-------|----------|----------|--------------------------|
| Bridge 1 | 256 → 512 | 1 | 3 samples |
| Bridge 2 | 512 → 512 | 2 | 5 samples |
| Bridge 3 | 512 → 512 | 4 | 9 samples |

At 625-sample resolution, a 9-sample receptive field covers ~90 ms at the bottleneck — equivalent to modeling over ~720 ms at the original 5000-sample input scale. This is enough to capture inter-P-wave timing relationships without recurrence.

Each bridge layer uses Batch Normalization and ReLU. A shared dropout (p=0.2) is applied after the bridge stack.

#### 3. Skip-Connected Decoder

The decoder mirrors the encoder. At each level, a `ConvTranspose1d` upsamples by ×2, concatenates the matching encoder skip connection, and passes through another `AttentionalInception` block.

Skip connections are critical here because fine-grained P-wave boundaries are lost at deep encoder levels — the skip connections reintroduce spatial detail from the encoder to guide precise wave boundary reconstruction.

#### 4. Output Heads (Three)

From the final decoder feature map (64 channels × 5000 samples):

| Head | Architecture | Output | Purpose |
|------|-------------|--------|---------|
| Heatmap | Conv64→32→1 + Sigmoid | (1, 5000) float in [0,1] | P-wave probability at each sample |
| Width | Conv64→32→1 | (1, 5000) float (raw logit) | Predicted P-wave width |
| Mask | Conv64→32→1 | (1, 5000) float (raw logit) | Binary segmentation mask |

The heatmap output is the primary signal for inference — peaks above the confidence threshold (0.45) that are separated by at least 60 samples (120 ms) are decoded as P-wave instances.

#### Full Architecture Diagram

```
Input (12 × 5000)
│
├── AttentionalInception → 64ch → MaxPool2 → Dropout(0.2)
│      [Skip e1: 64 × 5000]
├── AttentionalInception → 128ch → MaxPool2 → Dropout(0.2)
│      [Skip e2: 128 × 2500]
├── AttentionalInception → 256ch → MaxPool2 → Dropout(0.2)
│      [Skip e3: 256 × 1250]
│
├── Dilated CNN Bridge (dilation 1, 2, 4) → 512ch × 625
│
├── ConvTranspose×2 → 256ch + Skip e3 → AttInception → 256ch × 1250
├── ConvTranspose×2 → 128ch + Skip e2 → AttInception → 128ch × 2500
└── ConvTranspose×2 → 64ch + Skip e1 → AttInception → 64ch × 5000
         │
         ├── Heatmap Head → (1 × 5000) Sigmoid
         ├── Width Head   → (1 × 5000) Raw logit
         └── Mask Head    → (1 × 5000) Raw logit
```

<!-- 📸 SCREENSHOT: Model architecture diagram or training loss/metrics plot -->
> **[Add screenshot: Training curves (loss, precision, recall, F1) over epochs]**

---

### AtrionNetBaseline

A simpler baseline U-Net (standard Conv-BN-ReLU blocks, no attention, no dilated bridge) is included in `atrion_net.py` for ablation comparisons. It uses the same three output heads and loss function as the hybrid model.

---

## AV Block Classification Logic

After the model produces P-wave detections, a rule-based classifier (`_classify_av_block` in `predictor.py`) maps the measured intervals to a clinical diagnosis:

### Detection → Intervals → Diagnosis

**Step 1: Wave detection**
- **QRS (R-peaks):** Bandpass filtered (5–40 Hz), squared energy, peak detection with 400 ms refractory period
- **P-waves:** Model heatmap peaks (threshold 0.45, min distance 60 samples) → each peak decoded to span using the width head
- **T-waves:** Searched in a 100–450 ms window after each R-peak in the 0.5–15 Hz filtered signal

**Step 2: Interval calculation**
- `RR` — distance between consecutive R-peaks (in samples)
- `PR` — distance from P-wave center to the nearest following R-peak
- `P:QRS ratio` — total P-waves detected / total QRS complexes detected

**Step 3: Classification rules**

```
P:QRS ratio > 1.9  →  3rd Degree AV Block (Complete)      [Severe]
P:QRS ratio > 1.4  →  2nd Degree (Mobitz II)              [Severe]
P:QRS ratio > 1.15
  + progressive PR  →  2nd Degree (Mobitz I / Wenckebach) [Moderate]
  + no progression  →  2nd Degree (Mobitz II)             [Moderate]
avg PR > 200 ms     →  1st Degree AV Block                [Mild]
otherwise           →  Normal Sinus Rhythm                [Normal]
```

The P:QRS ratio is the key discriminator because in complete block, the atria and ventricles beat independently — there are more P-waves than QRS complexes. In Mobitz II some P-waves are blocked (dropped QRS), giving a ratio between 1.15–1.9.

**Confidence score** is derived directly from the model's heatmap: it takes the mean of the peak heatmap values across all detected P-wave spans. This means the confidence reflects how certain the neural network was about each P-wave, not a post-hoc softmax class probability.

### Supported Diagnoses

| Type | Severity | Key Indicator |
|------|----------|---------------|
| Normal Sinus Rhythm | Normal | PR 120–200ms, P:QRS ≈ 1.0 |
| 1st Degree AV Block | Mild | PR > 200ms consistently |
| 2nd Degree Mobitz I (Wenckebach) | Moderate | Progressively lengthening PR, then dropped QRS |
| 2nd Degree Mobitz II | Severe | Sudden dropped QRS, no PR progression |
| 3rd Degree (Complete) AV Block | Severe | P-waves and QRS complexes fully dissociated |

<!-- 📸 SCREENSHOT: Analysis results page showing diagnosis, confidence, and annotated ECG waveform -->
> **[Add screenshot: Analysis results page — annotated waveform with P/QRS/T highlights, diagnosis panel, and confidence score]**

---

## Explainability (XAI)

The heatmap output of the model is itself an explainability tool — it shows, at each sample point, how confident the model is that a P-wave is present. This is rendered in the frontend as an overlay on the ECG waveform.

Beyond the heatmap head, a clinical importance map (`_build_importance_map`) is computed during inference. It combines:

- **P-wave regions** — high importance, intensity scaled by diagnosis severity (AV block cases get higher P-wave weights since P-wave detection was the discriminating factor)
- **QRS complexes** — high importance as ventricular response markers
- **T-waves** — lower importance background context
- **Confidence-adaptive spreading** — low-confidence detections produce wider, more dispersed importance regions; high-confidence detections produce sharp, focused peaks

Libraries used: `captum`, `grad-cam`, `shap` (available for offline analysis; the runtime importance map uses the direct heatmap head output).

<!-- 📸 SCREENSHOT: XAI heatmap overlay on ECG waveform -->
> **[Add screenshot: ECG waveform with XAI heatmap overlay showing high-importance regions highlighted]**

---

## Project Structure

```
AtrionNet_Implementation/
│
├── ml_component/                    # All ML code lives here
│   ├── src/
│   │   ├── modeling/
│   │   │   └── atrion_net.py        # AtrionNetHybrid + AtrionNetBaseline definitions
│   │   │
│   │   ├── data_pipeline/
│   │   │   ├── ludb_loader.py       # LUDB dataset reader (wfdb annotations)
│   │   │   ├── instance_dataset.py  # PyTorch Dataset: converts P-wave spans to heatmap targets
│   │   │   └── augmentations.py     # Signal augmentations (noise, scaling, shifts)
│   │   │
│   │   ├── inference/
│   │   │   └── predictor.py         # AVBlockPredictor class — full inference + XAI + PDF report
│   │   │
│   │   ├── engine/
│   │   │   └── atrion_evaluator.py  # compute_instance_metrics, calculate_mAP, 1D-NMS, IoU
│   │   │
│   │   ├── losses/
│   │   │   └── segmentation_losses.py  # Focal loss (heatmap), combined instance loss
│   │   │
│   │   └── utils/
│   │
│   ├── notebooks/                   # Exploration and ablation notebooks
│   ├── benchmarking_codes/          # Benchmarking scripts for baseline comparison
│   ├── model_testing/               # Test scripts for individual model components
│   ├── outputs/
│   │   ├── weights/
│   │   │   └── atrion_hybrid_best.pth   # Best model checkpoint (saved during training)
│   │   ├── checkpoints/             # Intermediate epoch checkpoints
│   │   └── plots/                   # Training curve plots
│   │
│   ├── train.py                     # Main training entry point
│   ├── download_data.py             # Downloads LUDB from PhysioNet
│   ├── generate_test_case.py        # Generates .npy test cases from LUDB records
│   └── requirements.txt
│
├── backend/
│   ├── main.py                      # FastAPI app — /preview, /analyze, /report endpoints
│   ├── audit_system.py              # System health checker (model, weights, environment)
│   ├── uploads/                     # Temporary storage for uploaded ECG files
│   └── reports/                     # Generated PDF clinical reports
│
├── frontend/
│   └── src/
│       ├── pages/
│       │   ├── HomePage.jsx         # Landing page
│       │   ├── AnalysisPage.jsx     # Main upload + analysis interface
│       │   ├── InstructionsPage.jsx # Usage guide
│       │   └── AboutPage.jsx        # Project background
│       ├── components/              # Reusable UI components
│       ├── services/                # API call functions
│       ├── hooks/                   # Custom React hooks
│       └── index.css                # Global styling
│
├── tests/                           # Integration and unit tests
├── AtrionNet_Documentation/         # Research documentation and reports
│
├── ludb_test_case_104.npy           # Sample ECG test file
└── test_case_34.npy                 # Another sample test file
```

---

## Dataset

**Lobachevsky University Electrocardiography Database (LUDB)**

- **200 records**, each a 10-second 12-lead ECG at 500 Hz (5000 samples)
- **Expert wave annotations** for all 12 leads: P-onset, P-peak, P-offset, QRS boundaries, T-boundaries
- Records cover a range of conditions including normal sinus rhythm, various AV block types, bundle branch blocks, and other arrhythmias
- Annotation format: `.hea` + `.dat` + per-lead `.ii`, `.v1`, `.v5`, etc. annotation files (WFDB format)

**Source:** [PhysioNet LUDB v1.0.1](https://physionet.org/content/ludb/1.0.1/)

The `LUDBLoader` class (`ludb_loader.py`) handles:
- Recursive discovery of `.hea` files in the data directory
- Loading 12-lead signals via `wfdb.rdrecord`
- Parsing P-wave annotations (onset/peak/offset triplets) from lead-specific annotation files
- Cropping/padding to exactly 5000 samples
- Fallback lead order: `['ii', 'v1', 'v5', 'i', 'iii', 'avf']` — tries each lead until one with annotations is found

**Data split used in training:**

| Split | Records | Usage |
|-------|---------|-------|
| Train | ~160 | Model training |
| Validation | ~20 | Early stopping & threshold tuning |
| Test | ~20 | Final evaluation metrics |

<!-- 📸 SCREENSHOT: Sample ECG waveform from the LUDB dataset -->
> **[Add screenshot: A sample LUDB ECG record plotted with annotated P/QRS/T boundaries]**

---

## Setup & Installation

### Prerequisites

- Python 3.9 or higher
- Node.js 18+
- Git
- CUDA-capable GPU recommended for training (CPU works fine for inference)

### 1. Clone the repository

```bash
git clone https://github.com/dumiya123/avscan-high-grade-av-block-detection.git
cd AtrionNet_Implementation
```

### 2. Create a Python virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install Python dependencies

```bash
cd ml_component
pip install -r requirements.txt
```

Key packages installed:

| Package | Version | Purpose |
|---------|---------|---------|
| torch | ≥ 2.0.0 | Model training and inference |
| wfdb | ≥ 4.1.0 | Reading LUDB WFDB format files |
| neurokit2 | ≥ 0.2.7 | ECG preprocessing utilities |
| scipy | ≥ 1.10.0 | Signal filtering, peak detection |
| fastapi | ≥ 0.100.0 | Backend API server |
| reportlab | ≥ 4.0.0 | PDF report generation |
| captum | ≥ 0.6.0 | Gradient-based XAI |
| shap | ≥ 0.42.0 | SHAP value explanations |

### 4. Install frontend dependencies

```bash
cd frontend
npm install
```

### 5. Download the LUDB dataset

```bash
cd ml_component
python download_data.py
```

This downloads and extracts the LUDB dataset into `ml_component/data/raw/ludb/`. You'll need a PhysioNet account — the script will prompt for your credentials.

---

## Training

### Configuration

All training hyperparameters are in the `ATRION_CONFIG` dictionary at the top of `train.py`:

```python
ATRION_CONFIG = {
    "EPOCHS":          150,    # Maximum training epochs
    "PATIENCE":        25,     # Early stopping patience
    "BATCH_SIZE":      16,     # Batch size
    "LEARNING_RATE":   1e-4,   # Adam optimizer learning rate

    # Focal loss parameters
    "FOCAL_ALPHA":     2.0,    # Focal loss alpha
    "FOCAL_BETA":      4.0,    # Focal loss beta

    # Multi-task loss weights
    "LOSS_WEIGHT_HM":  2.0,    # Heatmap loss weight (primary)
    "LOSS_WEIGHT_W":   1.0,    # Width loss weight
    "LOSS_WEIGHT_M":   1.0,    # Mask loss weight

    # Inference thresholds
    "EVAL_CONF":       0.45,   # Heatmap peak confidence threshold
    "EVAL_DIST":       60,     # Min inter-peak distance (120ms at 500Hz)
    "EVAL_PROM":       0.10,   # Min peak prominence (10%)
}
```

### Loss Function

Training uses a composite loss across all three output heads:

```
Total Loss = (2.0 × Heatmap Focal Loss) + (1.0 × Width MSE) + (1.0 × Mask BCE)
```

The heatmap head uses a modified Focal Loss (α=2, β=4) which penalises missed detections more than false positives — important because P-waves are sparse in the signal (only a few per 10 seconds) and the loss would otherwise be dominated by the large number of "no P-wave" negatives.

### Run Training

```bash
cd ml_component
python train.py
```

Training output is saved to:
- `outputs/weights/atrion_hybrid_best.pth` — best checkpoint by validation F1
- `outputs/checkpoints/` — intermediate checkpoints every N epochs
- `outputs/plots/` — loss and metric curves

On a mid-range GPU (e.g., RTX 3060), training takes approximately 2–4 hours for 150 epochs on the full LUDB dataset.

<!-- 📸 SCREENSHOT: Training output terminal or training loss/F1 plot -->
> **[Add screenshot: Training terminal output showing epoch progress, loss values, and validation metrics]**

<!-- 📸 SCREENSHOT: Loss curve and F1 curve plots from outputs/plots/ -->
> **[Add screenshot: Training curves plot — loss over epochs, validation precision/recall/F1]**

---

## Running the Full System

### Start the backend API

Make sure the model checkpoint exists at `ml_component/outputs/weights/atrion_hybrid_best.pth` before starting. If it's missing, the backend runs in **fallback mode** (scipy-based peak detection instead of the neural model).

```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Check health: `http://localhost:8000/health`

```json
{
  "status": "online",
  "model_loaded": true,
  "device": "cuda"
}
```

### Start the frontend

```bash
cd frontend
npm run dev
```

Open `http://localhost:5173` in your browser.

<!-- 📸 SCREENSHOT: Frontend running in browser — upload page -->
> **[Add screenshot: Browser showing the AtrionNet dashboard with the ECG upload interface]**

### Generate a test case

If you don't have a `.npy` file ready:

```bash
cd ml_component
python generate_test_case.py
```

This extracts a sample from the LUDB dataset and saves it as a `.npy` file ready for upload.

---

## API Reference

The backend exposes four endpoints:

### `GET /health`
Returns server and model status.

**Response:**
```json
{
  "status": "online",
  "model_loaded": true,
  "device": "cuda"
}
```

---

### `POST /preview`
Loads the ECG file and returns a downsampled Lead II signal (1000 points) for fast frontend visualization — no model inference involved.

**Input:** `.npy` file (multipart form-data)

**Response:**
```json
{
  "filename": "patient_001.npy",
  "signal": [0.12, 0.08, ...],   // 1000 points, Lead II
  "sample_count": 5000,
  "fs": 500
}
```

---

### `POST /analyze`
Full inference pipeline: loads file → runs AtrionNetHybrid → detects waves → classifies AV block → generates PDF report.

**Input:** `.npy` file (multipart form-data)

**Response:**
```json
{
  "diagnosis": "3rd Degree AV Block",
  "confidence": 0.872,
  "severity": "Severe",
  "clinical_metrics": {
    "mean_pr_ms": 248.4,
    "heart_rate_bpm": 52.3
  },
  "intervals": {
    "pr": [124, 128, 131, ...],
    "rr": [577, 580, ...],
    "avg_pr": 127.8,
    "hr": 52.3
  },
  "waves": {
    "p_associated":   [[45, 72], [623, 651], ...],
    "p_dissociated":  [[312, 338], ...],
    "qrs":            [[84, 124], ...],
    "t":              [[145, 198], ...]
  },
  "heatmap": [0.01, 0.02, ..., 0.89, 0.91, ...],
  "signal":  [0.12, 0.08, ...],
  "report_id": "report_patient_001.pdf",
  "explanation": "Detected 12 P-waves and 7 QRS complexes..."
}
```

---

### `GET /report/{report_id}`
Downloads the generated PDF clinical report.

**Example:** `GET /report/report_patient_001.pdf`

Returns the PDF as a file attachment.

---

## Frontend Dashboard

The frontend is a React + Vite single-page application with four pages:

| Page | Route | Description |
|------|-------|-------------|
| Home | `/` | Landing page with project overview |
| Analysis | `/analysis` | Main upload and analysis interface |
| Instructions | `/instructions` | Step-by-step usage guide |
| About | `/about` | Project background and developer info |

The Analysis page is where most of the work happens:

1. **File upload zone** — drag and drop or click to upload a `.npy` ECG file
2. **Preview panel** — renders Lead II signal using the `/preview` endpoint before analysis
3. **Analyze button** — triggers the full inference pipeline
4. **Results panel** — shows diagnosis, severity badge, confidence bar, and clinical metrics table
5. **Annotated waveform** — ECG signal with P/QRS/T wave boundaries overlaid and XAI heatmap as a colored overlay
6. **Download report** — button to fetch and download the PDF from `/report/{id}`

<!-- 📸 SCREENSHOT: Full analysis results page -->
> **[Add screenshot: Full analysis page showing uploaded ECG, wave annotations, diagnosis result, and the downloadable report button]**

<!-- 📸 SCREENSHOT: PDF clinical report example -->
> **[Add screenshot: Generated PDF clinical report showing patient ECG, diagnosis summary, interval table, and XAI heatmap]**

---

## Evaluation Metrics

Detection performance is measured at the P-wave instance level (not at the ECG record level):

- **Precision** — of all predicted P-wave instances, what fraction are correct
- **Recall** — of all true P-wave instances in the dataset, what fraction were found
- **F1-Score** — harmonic mean of precision and recall
- **mAP (Mean Average Precision)** — area under the precision-recall curve, VOC 2010+ style, across the full test set

A predicted P-wave is counted as a True Positive if its center falls within the annotated onset-to-offset span of a ground truth P-wave (with ±20 ms slack), following AAMI/CSE clinical standards.

**Non-Maximum Suppression (1D-NMS)** is applied before metric computation — overlapping predictions (IoU > 0.5) are suppressed, keeping the highest-confidence detection.

### Threshold Calibration History

| Version | Conf. Threshold | Min Distance | Result |
|---------|----------------|--------------|--------|
| v4.0 | 0.20 | 20 samples | Precision 0.04 (5167 false positives) |
| v4.1 | 0.50 | 100 samples | Recall 0.31 (too many missed detections) |
| v4.2 | 0.35 | 60 samples | Balanced F1 target ≥ 0.70 |
| Final | 0.45 | 60 samples | Tuned to σ=6 Gaussian heatmap targets |

The final threshold (0.45) excludes T-waves and noise (which score ~0.1–0.2 on the heatmap) while capturing weak dissociated P-waves in AV block cases.

<!-- 📸 SCREENSHOT: Precision-Recall curve or confusion matrix -->
> **[Add screenshot: Precision-Recall curve and/or mAP results from evaluation]**

---

## Technology Stack

| Layer | Technology | Version |
|-------|-----------|---------|
| Deep Learning | PyTorch | ≥ 2.0.0 |
| ECG Signal Reading | wfdb | ≥ 4.1.0 |
| ECG Preprocessing | neurokit2, scipy | ≥ 0.2.7 / ≥ 1.10.0 |
| Data Processing | NumPy, Pandas | ≥ 1.24.0 / ≥ 2.0.0 |
| Visualization | Matplotlib, Seaborn | ≥ 3.7.0 |
| Explainability | captum, grad-cam, shap | ≥ 0.6.0 |
| API Server | FastAPI + uvicorn | ≥ 0.100.0 |
| Frontend | React + Vite | — |
| PDF Reports | ReportLab | ≥ 4.0.0 |
| Training Monitor | TensorBoard | ≥ 2.13.0 |

---

## Known Limitations

- Input format is currently restricted to `.npy` files in `(12, 5000)` or `(5000, 12)` shape at 500 Hz. EDF/DICOM support is not implemented.
- The LUDB dataset has 200 records. The model has not been validated on external datasets (e.g., PTBXL or PhysioNet 2020 Challenge data).
- Fallback mode (scipy-based detection without the trained model) is less accurate — it's included for demonstration purposes only.
- The classification logic uses fixed rule-based thresholds after P-wave detection. A learned classifier on top of interval features was not implemented in this version.

---

## Future Work

- Extend to EDF and DICOM input formats for real-world hospital data
- Validate on PTBXL and the PhysioNet 2020 Challenge dataset
- Replace rule-based classifier with a learned MLP on interval features
- Add real-time streaming ECG support via WebSocket
- ONNX export for edge deployment on portable ECG devices

---

## License

This project is developed for academic and research purposes as part of a final-year undergraduate research project.

Dataset (LUDB) is subject to [PhysioNet's terms of use](https://physionet.org/about/licenses/physionet-restricted-health-data-license-150/).

---

## Acknowledgements

- LUDB dataset — Kalyakulina et al., Lobachevsky University
- PhysioNet — for hosting openly accessible ECG datasets
- wfdb Python library — for WFDB format reading
- PyTorch team — for the deep learning framework
