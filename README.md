# Weave — Multimodal AI Fashion Ideation Assistant

**Weave** is an end-to-end AI fashion design assistant for early-stage ideation, silhouette adaptation, and moodboard curation.

This repository contains the **FastAPI Backend Services** and the **Machine Learning Subsystem** (Diffusion LoRA fine-tuning, ControlNet sketch conditioning, automated category-consistency scoring, and baseline comparative models).

## Google Colab

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1UVeSYKdwxKFgs_B38DDv4dPJZUEDJlrO)

---

## 1. System Architecture

```text
weave/
├── backend/
│   ├── app/
│   │   ├── api/v1/          # REST Endpoints (Auth, Generations, Variations, Inpaint, Boards, Exports)
│   │   ├── core/            # Configuration, Database Engine, Security & JWT
│   │   ├── models/          # Normalized SQLAlchemy Relational Schema
│   │   ├── repositories/    # Data Access Layer
│   │   ├── schemas/         # Pydantic Request/Response DTOs
│   │   └── services/        # Business Logic (Orchestrator, Diffusion Generation, Evaluation, Exports)
│   ├── media/               # Media Storage (uploads, generations, PDF exports)
│   ├── tests/               # Pytest Integration Suite
│   └── requirements.txt     # Backend Dependencies
├── ml/
│   ├── checkpoints/         # Pretrained & Fine-Tuned Model Weights
│   │   ├── baseline_acgan/  # Comparative WGAN-GP + SAGAN baseline
│   │   ├── classifier/      # ResNet-50 Fashion Category Consistency Classifier
│   │   └── final_fashion_lora/ # Stable Diffusion 1.5 Fashion LoRA adapter
│   ├── training/            # Training Pipelines (LoRA, Classifier, AC-GAN, Dataset utils)
│   ├── eval/                # Multi-metric evaluation suite (FID, IS, CLIP, LPIPS, Accuracy)
│   ├── notebooks/           # Master Google Colab training notebook
│   └── requirements.txt     # ML & Cloud GPU Training Dependencies
└── docs/
    └── GOOGLE_COLAB_TRAINING_NOTEBOOK.md  # Detailed cell-by-cell Colab training guide
```

---

## 2. Quickstart: Backend Setup

### Prerequisites
- Python 3.10+
- PostgreSQL 15 (Optional: automatically falls back to SQLite `weave_dev.db` for local development)

### Installation
```bash
# 1. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r backend/requirements.txt

# 3. Run the development server
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive API documentation will be available at:
- Swagger UI: `http://localhost:8000/api/v1/docs`
- ReDoc: `http://localhost:8000/api/v1/redoc`

### Running Backend Tests
```bash
pytest backend/tests
```

---

## 3. Machine Learning Subsystem

The ML subsystem fine-tunes domain-specific adapters while keeping base generative priors frozen for compute efficiency:

1. **Fashion LoRA (`ml/training/train_lora.py`)**:
   - Fine-tunes Stable Diffusion 1.5 cross-attention layers ($r=16, \alpha=32$) on fashion apparel catalogs.
2. **Category Consistency Classifier (`ml/training/train_classifier.py`)**:
   - ResNet-50 classifier validating that generated garments adhere to the designer's requested category ($\ge 90\%$ agreement target).
3. **Comparative AC-GAN Baseline (`ml/training/train_baseline_gan.py`)**:
   - From-scratch WGAN-GP + Spectral Normalization + SAGAN Self-Attention benchmark baseline.
4. **Automated Evaluation Suite (`ml/eval/evaluate_models.py`)**:
   - Computes Category-Consistency, CLIP style alignment, and LPIPS batch diversity.

### Training on Google Colab (Free T4 GPU)
Follow the step-by-step instructions in [`docs/GOOGLE_COLAB_TRAINING_NOTEBOOK.md`](docs/GOOGLE_COLAB_TRAINING_NOTEBOOK.md) using [`ml/notebooks/Weave_Fashion_LoRA_Training.ipynb`](ml/notebooks/Weave_Fashion_LoRA_Training.ipynb).

---

## 4. Model Checkpoints & Large File Handling

Trained weights reside in `ml/checkpoints/`:
- `baseline_acgan/acgan_baseline_final.pt` (16.3 MB)
- `final_fashion_lora/adapter_model.safetensors` (12.8 MB)
- `classifier/fashion_classifier_best.pt` (96.5 MB)

> **Note on Git LFS**: If pushing checkpoint binaries to GitHub, use Git LFS:
> ```bash
> git lfs install
> git lfs track "*.pt" "*.safetensors"
> ```
