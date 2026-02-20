# I-JEPA Project - 6th Semester

A Machine Learning project implementing I-JEPA (Image-based Joint-Embedding Predictive Architecture).

## System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| OS | Windows 10 / Linux | Windows 10+ / Ubuntu 20.04+ |
| RAM | 8 GB | 16 GB |
| GPU | Optional | NVIDIA GPU with CUDA support |
| Disk Space | 10 GB | 20 GB |

## Tested Configuration

| Component | Version |
|-----------|---------|
| Python | 3.13.5 |
| PyTorch | 2.7.1+cu118 |
| CUDA | 11.8 |
| GPU | NVIDIA GeForce GTX 1650 (4GB) |

---

## Environment Setup

### Step 1: Install Python

Download and install **Python 3.10 or higher** from [python.org](https://www.python.org/downloads/)

During installation:
- ✅ Check **"Add Python to PATH"**

Verify installation:
```bash
python --version
# Expected: Python 3.10+ (tested with 3.13.5)
```

### Step 2: Clone the Repository

```bash
git clone <repository-url>
cd 6th_Sem_Project
```

### Step 3: Create Virtual Environment

```bash
# Create virtual environment
python -m venv ijepa_env

# Activate (Windows PowerShell)
.\ijepa_env\Scripts\Activate.ps1

# Activate (Windows CMD)
ijepa_env\Scripts\activate.bat

# Activate (Linux/Mac)
source ijepa_env/bin/activate
```

### Step 4: Install Dependencies

#### Option A: Using requirements.txt (Recommended)

```bash
pip install -r requirements.txt
```

#### Option B: Manual Installation

**For NVIDIA GPU with CUDA 11.7+:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For NVIDIA GPU with CUDA 12.1+:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**For CPU only:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**Install other dependencies:**
```bash
pip install numpy pandas matplotlib seaborn
pip install scikit-learn tqdm
pip install opencv-python
```

### Step 5: Verify Installation

```bash
python src.py
```

Expected output:
```
==================================================
ML Environment Setup Verification
==================================================

PyTorch version: 2.7.1+cu118
CUDA available: True
CUDA version: 11.8
GPU: NVIDIA GeForce GTX 1650
GPU Memory: 4.0 GB

GPU Tensor Test: SUCCESS
Tensor device: cuda:0

==================================================
All systems ready for I-JEPA project!
==================================================
```

---

## Package Versions

| Package | Version | Purpose |
|---------|---------|---------|
| torch | 2.7.1+cu118 | Deep Learning framework |
| torchvision | 0.22.1+cu118 | Computer Vision utilities |
| torchaudio | 2.7.1+cu118 | Audio processing |
| numpy | 2.3.5 | Numerical computing |
| pandas | 3.0.0 | Data manipulation |
| matplotlib | 3.10.8 | Plotting |
| seaborn | 0.13.2 | Statistical visualization |
| scikit-learn | 1.8.0 | Machine Learning utilities |
| opencv-python | 4.13.0 | Image processing |
| tqdm | 4.67.2 | Progress bars |
| scipy | 1.17.0 | Scientific computing |
| pillow | 12.0.0 | Image handling |

---

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'torch._strobelight'`

**Cause:** Corrupted PyTorch installation or broken virtual environment paths.

**Solution:**
```bash
# Delete and recreate the virtual environment
Remove-Item -Recurse -Force ijepa_env  # Windows PowerShell
# OR
rm -rf ijepa_env  # Linux/Mac

# Recreate and reinstall
python -m venv ijepa_env
.\ijepa_env\Scripts\Activate.ps1  # Windows
pip install -r requirements.txt
```

### Issue: `No matching distribution found for torch`

**Cause:** Python version incompatibility or wrong index URL.

**Solution:**
- Ensure Python version is 3.10-3.13
- Check your CUDA version with `nvidia-smi` and use the correct index URL
- Try the CPU-only version if GPU installation fails

### Issue: Virtual environment paths broken after moving project folder

**Cause:** Virtual environments have hardcoded paths.

**Solution:** Delete `ijepa_env` folder and recreate it in the new location.

### Check CUDA Version

```bash
nvidia-smi
```

Look for "CUDA Version" in the output to determine which PyTorch variant to install.

---

## Project Structure

```
6th_Sem_Project/
├── data/               # MVTec AD dataset (extracted)
├── src/                # Source code
│   ├── datasets.py     # MVTec loader, transforms, patchify
│   ├── model_autoencoder.py  # Baseline autoencoder (Phase 2)
│   ├── train_baseline.py     # Train autoencoder
│   ├── anomaly_eval.py       # ROC-AUC evaluation
│   ├── utils.py        # Utilities
│   └── ...
├── checkpoints/        # Saved models
├── results/            # Metrics and visualizations
├── run_baseline.py     # Train + evaluate baseline (Phase 2)
├── src.py              # Environment verification
├── requirements.txt
├── README.md
└── .gitignore
```

## Phase 2: Baseline Autoencoder

Train the reconstruction baseline (normal images only) and evaluate with ROC-AUC:

```bash
# Activate venv first, then:
python run_baseline.py --category leather --epochs 50
```

Evaluate only (using an existing checkpoint):
```bash
python -m src.anomaly_eval --category leather --checkpoint checkpoints/autoencoder_leather.pth
```

Results are saved under `results/baseline/<category>/` (metrics_baseline.json, test_scores.npy, test_labels.npy). Use these as baseline to compare with I-JEPA in Phase 3.

### Check a single image (Normal or Anomaly?)

**Input:** Path to one image file (must be the same category the model was trained on, e.g. leather).

**Output:** Anomaly score and result: "Normal" or "Anomaly (Defect)".

```bash
# Example: check a leather image
python check_image.py data/leather/test/color/000.png --category leather

# Or a normal leather image
python check_image.py data/leather/train/good/001.png --category leather
```

The script uses the checkpoint `checkpoints/autoencoder_<category>.pth` and (if available) the threshold from `results/baseline/<category>/metrics_baseline.json`.

---

## Author

[Aman Sheikh] - 6th Semester Project
