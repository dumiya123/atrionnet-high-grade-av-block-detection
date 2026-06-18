# AtrionNet Quick Reference Guide

## 🚀 Quick Start

### Clone & Setup (5 minutes)
```bash
git clone https://github.com/dumiya123/atrionnet-high-grade-av-block-detection.git
cd atrionnet-high-grade-av-block-detection

# Backend
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r ../ml_component/requirements.txt
python main.py

# Frontend (new terminal)
cd frontend
npm install
npm run dev
```

Access at: `http://localhost:5173` (Frontend) | `http://localhost:8000/docs` (API)

---

## 📊 Key Features

| Feature | Status | Details |
|---------|--------|---------|
| 🔬 **P-Wave Detection** | ✅ 97.41% F1 | Anchor-free multi-task learning |
| 📱 **ECG Visualization** | ✅ Live | Interactive Plotly charts |
| 🤖 **Explainable AI** | ✅ XAI Maps | Attention heatmaps |
| 📄 **PDF Reports** | ✅ Auto | Clinical-grade documentation |
| 🏥 **Clinical Classification** | ✅ All Types | NSR, 1st/2nd/3rd degree AV blocks |

---

## 🏗️ Architecture

```
AtrionNet
├── Backend (FastAPI)
│   ├── main.py - API routes
│   ├── audit_system.py - Quality checks
│   └── uploads/ & reports/ - Data storage
├── Frontend (React + Vite)
│   ├── Dashboard - Main interface
│   ├── ECGViewer - Signal visualization
│   └── AnalysisMetrics - Results display
└── ML Pipeline (PyTorch)
    ├── preprocessing - Signal cleaning
    ├── model - AtrionNetHybrid v5.0
    └── inference - Prediction engine
```

---

## 🔧 API Endpoints

### Health Check
```bash
GET /health
# Response: {"status": "online", "model_loaded": true, "device": "cuda"}
```

### Preview Signal
```bash
POST /preview
Body: file=@signal.npy
# Response: {"signal": [...], "duration": 10, "format": "valid"}
```

### Run Analysis
```bash
POST /analyze
Body: file=@signal.npy
# Response: {
#   "diagnosis": "3rd Degree AV Block",
#   "confidence": 0.948,
#   "severity": "Severe",
#   "clinical_metrics": {...},
#   "report_id": "report_123.pdf"
# }
```

### Get Report
```bash
GET /report/{report_id}
# Downloads PDF report
```

---

## 📈 Model Performance

| Metric | Value | Details |
|--------|-------|---------|
| **Precision** | 97.41% | Minimal false positives |
| **Recall** | 97.41% | Catches subtle blocks |
| **F1-Score** | 97.41% | Balanced performance |
| **Inference Time** | 14 ms | Real-time capable |
| **Expert Agreement** | 100% | On 3rd-degree cases |

---

## 🧠 Model Details

**AtrionNetHybrid v5.0:**
- Input: [Batch, 12 leads, 5000 samples]
- Architecture: Attentional Encoder → Dilated Bridge → Decoder
- Output Heads:
  - Heatmap: P-wave center probability
  - Width: Wave boundary spans
  - Mask: Segmentation mask

**Key Components:**
- 🔹 Squeeze-and-Excitation attention blocks
- 🔹 Multi-scale inception modules [9, 19, 39 kernels]
- 🔹 Dilated convolutions [d=1, 2, 4] at bottleneck
- 🔹 Skip connections for spatial detail

---

## 📦 Dataset

**Lobachevsky University ECG Database (LUDB):**
- 200 clinical records
- 10 seconds each @ 500 Hz
- 12-lead format
- Expert annotations

**Split:**
- Training: 140 records (70%)
- Validation: 30 records (15%)
- Testing: 30 records (15%)

---

## 💾 File Formats

**Input:**
- Format: `.npy` (NumPy binary)
- Shape: [12, 5000] (12 leads × 5000 samples)
- Duration: 10 seconds @ 500 Hz

**Output:**
- Report: `.pdf` (Clinical summary)
- Metrics: JSON (Programmatic access)
- Visualization: Interactive HTML

---

## 🔐 Data Handling

**Security:**
- ✅ HIPAA-compliant file handling
- ✅ Automatic cleanup of uploads
- ✅ No data retention in cloud
- ✅ Secure report archival

**Processing:**
1. Signal validation (duration, leads)
2. Butterworth filtering (0.5-40 Hz)
3. Z-score normalization
4. Model inference
5. Post-processing & classification
6. PDF generation & storage

---

## 🐛 Debugging

### Check System Status
```bash
python backend/audit_system.py
# Verifies: PyTorch, CUDA, weights, model accuracy
```

### Run Model Test
```bash
python backend/test_xai_fix.py
# Tests attention heatmap generation
```

### View API Docs
Open: `http://localhost:8000/docs` (Swagger UI)

---

## 📊 Preprocessing Pipeline

```
Raw ECG Signal
    ↓
[2nd-Order Butterworth Bandpass Filter: 0.5-40 Hz]
    ↓
[Per-Lead Z-Score Normalization]
    ↓
[Optional: Data Augmentation]
    ├─ Baseline wander simulation
    ├─ Powerline noise injection
    ├─ Gaussian noise addition
    ├─ Amplitude scaling (0.8-1.2x)
    ├─ Random time shift (±500ms)
    └─ Random lead dropout
    ↓
[Ready for Inference]
```

---

## 🔍 Rhythm Classification

| Rhythm | Criteria | Notes |
|--------|----------|-------|
| **NSR** | PR 120-200ms, ratio 1:1 | Normal conduction |
| **1st Degree** | PR > 200ms, ratio 1:1 | Delayed but conducted |
| **2nd Mobitz I** | Progressive PR prolongation | Wenckebach pattern |
| **2nd Mobitz II** | Sudden dropped QRS | High risk of progression |
| **3rd Degree** | PR ratio > 1.9, dissociation | Complete block |

---

## 🎯 Clinical Metrics

**Calculated by System:**
- **PR Interval:** P-to-R distance (ms)
- **Heart Rate:** Beats per minute
- **Conduction Ratio:** P-waves to QRS ratio
- **Wave Boundaries:** Onset, peak, offset times

---

## 📝 Common Tasks

### Train Custom Model
```bash
cd ml_component
python train.py --config config.yaml --gpus 1
```

### Evaluate Model
```bash
cd ml_component
python evaluate_all.py --weights best_model.pth
```

### Generate Report Manually
```python
from backend.predictor import AVBlockPredictor
predictor = AVBlockPredictor()
result = predictor.predict(ecg_signal)
predictor.generate_report(result, "output.pdf")
```

---

## 🚨 Troubleshooting

| Issue | Solution |
|-------|----------|
| Port 8000 in use | `lsof -i :8000` then kill process |
| CUDA not found | Install PyTorch with CUDA support |
| Module not found | `pip install -r requirements.txt` |
| Model weights missing | Check `ml_component/outputs/weights/` |
| Frontend won't load | `npm install && npm run dev` |

---

## 📚 Documentation

- **Full README:** [README.md](README.md)
- **Contributing:** [CONTRIBUTING.md](CONTRIBUTING.md)
- **API Docs:** http://localhost:8000/docs
- **Research:** See methodology section in README

---

## 🔗 Useful Links

- 📖 [PhysioNet LUDB Dataset](https://physionet.org/content/ludb/1.0.1/)
- 📊 [Visualization Tools](https://visualgo.net/)
- 🤖 [PyTorch Docs](https://pytorch.org/docs/)
- ⚡ [FastAPI Docs](https://fastapi.tiangolo.com/)
- ⚛️ [React Docs](https://react.dev/)

---

## 💡 Tips & Tricks

**Performance:**
- Use GPU for faster inference (14ms vs 200ms)
- Batch processing available for multiple signals
- Model quantization available for edge deployment

**Development:**
- Use `--reload` flag in FastAPI for auto-restart
- Set `DEBUG=true` in frontend for better error messages
- Use breakpoints in PyCharm for ML debugging

**Deployment:**
- Docker image available for containerization
- GitHub Actions CI/CD configured
- Supports cloud deployment (AWS, GCP, Azure)

---

## 🤝 Getting Help

- **Issues:** https://github.com/dumiya123/atrionnet-high-grade-av-block-detection/issues
- **Discussions:** https://github.com/dumiya123/atrionnet-high-grade-av-block-detection/discussions
- **Email:** gamagedumindui@gmail.com

---

## 📋 System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **Python** | 3.9 | 3.10-3.11 |
| **PyTorch** | 2.0.0 | Latest |
| **Node.js** | 16.0 | 18.0+ |
| **GPU** | Not required | NVIDIA CUDA 11.8+ |
| **RAM** | 8 GB | 16+ GB |
| **Disk** | 2 GB | 10+ GB |

---

**Last Updated:** June 2026 | **Version:** 5.0.0
