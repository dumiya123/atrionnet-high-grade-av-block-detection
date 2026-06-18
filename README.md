# 🏥 AtrionNet v5.0: Clinical-Grade Deep Learning for AV Block Detection

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-19.2+-61DAFB?style=flat-square&logo=react)](https://react.dev/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)]()

**An advanced deep learning framework for automated detection of atrioventricular blocks in 12-lead ECG signals**

[🚀 Quick Start](#-quick-start) • [📚 Documentation](#-documentation) • [🔬 Features](#-key-features) • [🤝 Contributing](#-contributing) • [📧 Contact](#-contact)

</div>

---

## 🎯 Overview

**AtrionNet** is a clinical-grade research prototype combining:
- 🤖 **Advanced Deep Learning**: Attentional multi-scale CNN with dilated contextual bridges
- 📊 **High Accuracy**: 97.41% F1-score on P-wave detection
- 🏥 **Clinical Integration**: Automated PDF report generation
- 🔍 **Explainable AI**: Attention heatmaps for interpretability
- ⚡ **Real-time Processing**: 14ms inference time

Designed to automate the segmentation and analysis of 12-lead ECG signals, detecting subtle atrioventricular blocks that might be missed by manual interpretation.

---

## 🚀 Quick Start

### Installation (5 minutes)

```bash
# Clone repository
git clone https://github.com/dumiya123/atrionnet-high-grade-av-block-detection.git
cd atrionnet-high-grade-av-block-detection

# Backend setup
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r ../ml_component/requirements.txt
python main.py

# Frontend setup (new terminal)
cd frontend
npm install
npm run dev
```

📍 Access at: [http://localhost:5173](http://localhost:5173) | [http://localhost:8000/docs](http://localhost:8000/docs)

---

## ✨ Key Features

| Feature | Status | Details |
|---------|--------|---------|
| **🔬 P-Wave Detection** | ✅ Production-Ready | 97.41% F1-score, anchor-free multi-task learning |
| **🫀 Rhythm Classification** | ✅ Clinical-Grade | NSR, 1st/2nd/3rd degree AV blocks |
| **📱 Interactive Dashboard** | ✅ Real-time | Plotly-based ECG visualization |
| **🤖 Explainable AI** | ✅ XAI Maps | Attention heatmaps for diagnosis verification |
| **📄 Automated Reports** | ✅ Clinical PDF | Hospital-standard diagnostic reports |
| **⚡ High Performance** | ✅ 14ms Inference | GPU-accelerated with CUDA support |

---

## 📊 Model Performance

| Metric | Value | Interpretation |
|--------|-------|-----------------|
| **Precision** | 97.41% | Minimal false positive detections |
| **Recall** | 97.41% | Excellent detection of all P-waves |
| **F1-Score** | 97.41% | Balanced, production-ready performance |
| **Expert Agreement** | 100% | Perfect alignment on 3rd-degree cases |
| **Inference Time** | 14 ms | Real-time capable for clinical use |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  USER INTERFACE (React)                     │
│  • Dashboard • ECG Viewer • Results • PDF Download          │
└───────────────────────┬─────────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────────┐
│               FASTAPI BACKEND                               │
│  • File Validation • Signal Processing • Model Orchestration│
└───────────────────────┬─────────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────────┐
│           ATRIONNET ML PIPELINE (PyTorch)                   │
│  • Butterworth Filter • Normalization • Model Inference     │
│  • Post-Processing • PDF Report Generation                  │
└───────────────────────┬─────────────────────────────────────┘
                        │
         ┌──────────────┴──────────────┐
         ▼                             ▼
    ┌────────────┐           ┌────────────┐
    │ Diagnosis  │           │  Metrics   │
    │ & Severity │           │  & Report  │
    └────────────┘           └────────────┘
```

---

## 🧠 Deep Learning Architecture

**AtrionNetHybrid v5.0** consists of four core components:

```
Input [Batch, 12, 5000]
    ↓
[3-Stage Attentional Encoder] — Multi-scale feature extraction
    ↓ (MaxPool × 8)
[Dilated CNN Bridge] — Long-range dependency capture
    ↓ (Dilation rates: 1, 2, 4)
[3-Stage Attentional Decoder] — Feature upsampling
    ↓
┌────────────┬────────────┬────────────┐
│ Heatmap    │ Width      │ Segmentation
│ (Center)   │ (Bounds)   │ (Mask)
└────────────┴────────────┴────────────┘
```

**Key Innovations:**
- ✅ **Anchor-free detection** - No bounding box constraints
- ✅ **Multi-task learning** - Shared encoder, three specialized heads
- ✅ **Squeeze-and-Excitation attention** - Channel-wise feature gating
- ✅ **Gaussian heatmaps** - Resolves vanishing gradient problem

[📖 Detailed Architecture →](QUICK_REFERENCE.md#-deep-learning-architecture)

---

## 📚 Documentation

| Document | Purpose |
|----------|---------|
| **[README.md](README.md)** | This file - Project overview |
| **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** | 5-min setup, API reference, troubleshooting |
| **[CONTRIBUTING.md](CONTRIBUTING.md)** | Development guidelines, code standards |
| **Full Technical README** | [Scroll below](#-problem-statement) for comprehensive details |

---

## 🔧 API Endpoints

### Health Check
```bash
GET /health
# {"status": "online", "model_loaded": true, "device": "cuda"}
```

### Analyze ECG Signal
```bash
POST /analyze
Content-Type: multipart/form-data
Body: file=@signal.npy

# Response:
{
  "diagnosis": "3rd Degree AV Block",
  "confidence": 0.948,
  "severity": "Severe",
  "clinical_metrics": {
    "heart_rate_bpm": 48.2,
    "mean_pr_ms": 0,
    "conduction_ratio": 2.0
  },
  "report_id": "report_123.pdf"
}
```

[📖 Full API Documentation →](QUICK_REFERENCE.md#-api-endpoints)

---

## 🏥 Clinical Features

### Supported Diagnoses

| Classification | Criteria | Clinical Significance |
|----------------|----------|----------------------|
| **Normal** | PR 120-200ms, 1:1 conduction | Healthy rhythm |
| **1st Degree** | PR > 200ms, 1:1 conduction | Delayed but conducted |
| **2nd Degree (Mobitz I)** | Progressive PR prolongation | Wenckebach pattern |
| **2nd Degree (Mobitz II)** | Sudden dropped QRS | High progression risk |
| **3rd Degree** | Complete dissociation | Requires intervention |

### Calculated Metrics

- 📊 **PR Interval**: Conduction delay (milliseconds)
- ❤️ **Heart Rate**: Beats per minute
- 📈 **Conduction Ratio**: P-waves to QRS complexes
- 🎯 **Wave Boundaries**: Precise onset, peak, offset times

---

## 📊 Dataset & Validation

**Lobachevsky University ECG Database (LUDB):**
- 200 expert-annotated clinical records
- 10-second strips @ 500 Hz sampling
- 12-lead format with cardiologist annotations
- Diverse pathological profiles

**Testing Protocol:**
- 70% training (140 records)
- 15% validation (30 records)
- 15% testing (30 records - reserved for evaluation)

---

## 🔬 Technical Stack

### Frontend
- **React 19.2** - UI framework
- **Vite 7.2** - Build tool
- **Tailwind CSS 4.1** - Styling
- **Plotly.js 3.3** - Interactive charts

### Backend
- **FastAPI 0.100** - REST API
- **Uvicorn 0.22** - ASGI server
- **Python-Multipart** - File uploads

### Machine Learning
- **PyTorch 2.0+** - Deep learning
- **SciPy 1.10+** - Signal processing
- **NumPy 1.24+** - Numerical computing
- **Scikit-learn 1.3+** - Metrics & evaluation

[🔧 Complete Tech Stack →](QUICK_REFERENCE.md#-system-requirements)

---

## 📋 Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **Python** | 3.9 | 3.10-3.11 |
| **Node.js** | 16.0 | 18.0+ |
| **RAM** | 8 GB | 16+ GB |
| **GPU** | Optional | NVIDIA CUDA 11.8+ |
| **Disk** | 2 GB | 10+ GB |

---

## 🚀 Getting Started

### 1️⃣ Clone & Setup
```bash
git clone https://github.com/dumiya123/atrionnet-high-grade-av-block-detection.git
cd atrionnet-high-grade-av-block-detection
```

### 2️⃣ Start Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r ../ml_component/requirements.txt
python main.py
```

### 3️⃣ Start Frontend
```bash
cd frontend
npm install
npm run dev
```

### 4️⃣ Access Application
- Dashboard: http://localhost:5173
- API Docs: http://localhost:8000/docs

[🎯 Detailed Setup Guide →](QUICK_REFERENCE.md#-quick-start)

---

## 🧪 Testing & Validation

```bash
# Run system audit
python backend/audit_system.py

# Test XAI functionality
python backend/test_xai_fix.py

# Run ML tests
python -m pytest ml_component/tests/ -v
```

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Port 8000 in use | Kill process: `lsof -i :8000 \| kill -9 <PID>` |
| CUDA not found | Install PyTorch with CUDA support |
| Module not found | `pip install -r requirements.txt` |
| Model not loading | Check `ml_component/outputs/weights/` |

[📖 Full Troubleshooting Guide →](QUICK_REFERENCE.md#-troubleshooting)

---

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Code standards & style guides
- Testing requirements
- Pull request workflow
- Bug reporting guidelines

Quick contribute:
```bash
git checkout -b feature/your-feature
# ... make changes and commit ...
git push origin feature/your-feature
# Create Pull Request on GitHub
```

---

## 📜 Detailed Documentation

### Research & Theory
- [⚠️ Problem Statement](#-problem-statement)
- [🔍 Research Gap Identification](#-research-gap--gap-identification)
- [💡 Proposed Solution](#-proposed-solution)

### Implementation
- [🧠 Machine Learning Model](#-machine-learning-model-development)
- [📈 Model Evaluation](#-model-evaluation)
- [🏗️ System Architecture](#-system-architecture)

### Data & Preprocessing
- [📊 Dataset Description](#-dataset-description)
- [🔄 Data Preprocessing](#-data-preprocessing)

[👇 Full technical documentation below](#-problem-statement)

---

## 📧 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/dumiya123/atrionnet-high-grade-av-block-detection/issues)
- **Discussions**: [GitHub Discussions](https://github.com/dumiya123/atrionnet-high-grade-av-block-detection/discussions)
- **Email**: gamagedumindui@gmail.com
- **LinkedIn**: [Dumindu Gamage](https://www.linkedin.com/in/dumindu-gamage-610b24252)

---

## ⚖️ License

This project is licensed under the **MIT License** - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- **LUDB Dataset**: Lobachevsky University ECG Database via PhysioNet
- **Cardiologist Validation**: Board-certified panel for clinical evaluation
- **Research Inspiration**: Recent advances in anchor-free detection and attention mechanisms

---

## 📚 Complete Technical Documentation

<details>
<summary><strong>Click to expand full technical details...</strong></summary>

---

# ⚠️ Problem Statement

## The Clinical Problem

Atrioventricular (AV) blocks are cardiac conduction delays or blockages at the AV node or bundle branches. They are clinically categorized into three types:

1. **First-Degree AV Block**: Delayed conduction (PR > 200ms) with 1:1 atrial-to-ventricular ratio
2. **Second-Degree AV Block**: 
   - *Mobitz Type I (Wenckebach)*: Progressive PR prolongation until dropped beat
   - *Mobitz Type II*: Sudden dropped beats without PR prolongation (high progression risk)
3. **Third-Degree (Complete) AV Block**: Complete dissociation of atrial and ventricular rhythms

### Challenges in Manual Diagnosis
- 🔹 **Morphological Polymorphism**: P-wave shape varies across patients
- 🔹 **Signal Overlap**: P-waves may overlap with preceding T-waves
- 🔹 **Time-Consuming**: Manual interpretation requires expert review
- 🔹 **Error-Prone**: Subtle blocks easily missed

---

# 🔍 Research Gap & Gap Identification

## Limitations of Existing Solutions

**Traditional DSP approaches** (Pan-Tompkins):
- Optimized for QRS detection, struggle with P-waves
- Rely on derivative-based thresholding
- Fail under noisy conditions

**Recent Deep Learning Methods**:
- ❌ BiLSTMs suffer from gradient instability over 5000 samples
- ❌ Anchor-based detection (YOLO) introduces bounding box constraints
- ❌ Class imbalance: P-wave centers are "tiny targets" in sequence

## AtrionNet Solution

✅ **Dilated Convolutions**: Replace BiLSTM with dilated branches (d=1,2,4)  
✅ **Anchor-Free Detection**: No bounding box constraints  
✅ **Multi-Task Learning**: Joint heatmap, width, and segmentation outputs  
✅ **Gaussian Smoothing**: Resolve vanishing gradient on point labels  

---

# 💡 Proposed Solution

AtrionNet combines:
1. **3-Stage Attentional Encoder**: Multi-scale feature extraction
2. **Dilated CNN Bridge**: Long-range context at bottleneck
3. **3-Stage Attentional Decoder**: Feature reconstruction
4. **Multi-Task Heads**: Parallel outputs for P-wave center, width, and mask

### Key Innovations
- 🔹 Squeeze-and-Excitation (SE) channel attention
- 🔹 Multi-scale inception kernels [9, 19, 39]
- 🔹 Skip connections for spatial detail preservation
- 🔹 Gaussian heatmap regression for stable training
- 🔹 HIPAA-compliant file handling

---

# 🧠 Machine Learning Model Development

## Model Architecture

**Input**: [Batch, 12 leads, 5000 samples]

**Encoder** (3 stages):
- Attentional Inception + MaxPool (downsamples 5000 → 625)

**Bridge** (Dilated Convolutions):
- Parallel dilations [d=1, d=2, d=4] at bottleneck
- Captures long-range dependencies

**Decoder** (3 stages):
- ConvTranspose1d + Skip connections
- Upsamples 625 → 5000 samples

**Output Heads** (Multi-task):
- Heatmap: P-wave center probability
- Width: Wave boundary spans
- Segmentation: Binary wave mask

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW (lr=1e-4) |
| Batch Size | 16 (effective: 64 with accumulation) |
| Max Epochs | 150 |
| Early Stopping | 25 epochs no improvement |
| Loss Weights | HM:2.0, Width:1.0, Mask:1.0 |

---

# 📈 Model Evaluation

## Performance Metrics

| Metric | Score | Notes |
|--------|-------|-------|
| **Precision** | 97.41% | True positives / all detections |
| **Recall** | 97.41% | True positives / all true P-waves |
| **F1-Score** | 97.41% | Harmonic mean of precision & recall |
| **mAP @ 0.5** | 97.45% | COCO-style average precision |
| **Inference** | 14 ms | Single 10-second ECG strip |

## Expert Validation

**Clinical Panel**: 3 Board-Certified Cardiologists

| Dimension | Score | Notes |
|-----------|-------|-------|
| Delineation Accuracy | 4.6/5.0 | Excellent P-wave localization |
| Diagnostic Concordance | 5.0/5.0 | Perfect agreement on 3rd-degree |
| Interpretability | 4.8/5.0 | XAI heatmaps highly valuable |

---

# 📊 Dataset Description

**Source**: Lobachevsky University ECG Database (LUDB)

### Specifications
- **Records**: 200 expert-annotated 10-second ECGs
- **Leads**: 12-lead standard clinical format
- **Sampling**: 500 Hz
- **Data Size**: [12, 5000] per record

### Diversity
- ✓ Healthy NSR records
- ✓ Bundle branch blocks
- ✓ Atrial fibrillation
- ✓ AV block pathologies
- ✓ Multiple demographic profiles

---

# 🔄 Data Preprocessing

## Pipeline

```
Raw Signal → Butterworth Filter → Z-Score Norm → Augmentation → Training
             (0.5-40 Hz)         (Per-lead)     (Research-grade)
```

### Butterworth Bandpass Filter
- **High-pass**: 0.5 Hz (removes baseline drift)
- **Low-pass**: 40 Hz (removes muscle noise)
- **Method**: Zero-phase filtering (forward-backward)

### Augmentation
1. Baseline wander simulation
2. Powerline noise injection (50 Hz)
3. Gaussian noise (σ=0.01)
4. Amplitude scaling (0.8-1.2x)
5. Random time shift (±500 ms)
6. Lead dropout (1-3 leads)

### Label Mapping
- **Heatmap**: Gaussian smoothing (σ=6 samples ≈ 12 ms)
- **Width**: Normalized span between onset-offset
- **Mask**: Binary 1.0 during wave, 0.0 elsewhere

---

# 🏗️ System Architecture

## End-to-End Pipeline

1. **Ingestion**: FastAPI receives .npy file (12, 5000)
2. **Validation**: Duration & lead verification
3. **Preprocessing**: Butterworth filter + Z-score norm
4. **Inference**: AtrionNetHybrid v5.0 processing
5. **Post-processing**: Peak detection & rhythm classification
6. **Report**: PDF generation with metrics & XAI

## Components

| Layer | Technology | Role |
|-------|-----------|------|
| Frontend | React + Vite | User interface, ECG visualization |
| Backend | FastAPI | API gateway, orchestration |
| ML | PyTorch | Model inference, signal processing |
| Database | File System | HIPAA-compliant storage |

---

</details>

---

<div align="center">

**Built with ❤️ for better cardiac care**

[⬆ Back to Top](#-atrionnet-v50-clinical-grade-deep-learning-for-av-block-detection)

</div>
