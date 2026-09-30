# Weave — Machine Learning Subsystem (`ml/`)

This directory contains the training pipelines, evaluation suites, and Google Colab notebooks for **Weave: A Multimodal AI Design Assistant for Early-Stage Fashion Ideation**.

## Directory Structure

```text
ml/
├── notebooks/
│   └── Weave_Fashion_LoRA_Training.ipynb   # Master Google Colab training notebook
├── training/
│   ├── dataset_utils.py                    # Fashion dataset loader, transforms & taxonomy
│   ├── train_lora.py                       # Stable Diffusion 1.5 Fashion LoRA fine-tuning
│   ├── train_classifier.py                 # ResNet-50 Category Consistency Classifier
│   └── train_baseline_gan.py               # Comparative AC-GAN Baseline (WGAN-GP + SAGAN)
├── eval/
│   └── evaluate_models.py                  # Evaluation suite (FID, IS, CLIP, LPIPS, Accuracy)
├── checkpoints/                            # Drop location for trained model weights
└── requirements.txt                        # Python dependencies
```

## How to Train the Models on Google Colab

1. **Upload Notebook**:
   - Open [Google Colab](https://colab.research.google.com).
   - Upload [`notebooks/Weave_Fashion_LoRA_Training.ipynb`](file:///Users/nidhisingh/Desktop/ojt/ml/notebooks/Weave_Fashion_LoRA_Training.ipynb).
   - Set Hardware Accelerator to **T4 GPU** (`Runtime > Change runtime type > T4 GPU`).
2. **Execute**:
   - The notebook connects to Google Drive (`/content/drive/MyDrive/weave/checkpoints`).
   - Automatically checkpoints every 400–500 steps to prevent loss during Colab idle timeouts.
   - Automatically detects and resumes from previous checkpoints if restarted.
3. **Artifacts Export**:
   - At the end of Step 10 in the notebook, a zip archive `weave_trained_models.zip` is created in your Google Drive.
   - Download and extract it into `ml/checkpoints/` for backend inference serving.

For complete cell-by-cell documentation and code explanation, refer to [`docs/GOOGLE_COLAB_TRAINING_NOTEBOOK.md`](file:///Users/nidhisingh/Desktop/ojt/docs/GOOGLE_COLAB_TRAINING_NOTEBOOK.md).
