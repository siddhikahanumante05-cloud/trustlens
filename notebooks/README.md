# TrustLens DFDC Dataset Training & Quantization Guide

This guide walks you through training the **TrustLens** deepfake detection models on the official **Meta Deepfake Detection Challenge (DFDC)** dataset using free cloud GPUs (Kaggle or Google Colab) without filling up your local disk.

---

## Why Train on Kaggle?
- **Zero Local Disk Usage**: The full DFDC dataset is **472 GB**. On Kaggle, the dataset is pre-hosted and mounted instantaneously under `/kaggle/input/deepfake-detection-challenge/`.
- **Free Cloud GPUs**: Kaggle provides free access to NVIDIA Tesla T4 (dual GPU) and P100 GPUs (up to 30 hours/week).
- **Turnkey Export**: The pipeline automatically trains, dynamically quantizes the models to INT8, tests inference latency (<50ms on CPU), and exports a `trustlens_trained_models.zip` package ready for the backend.

---

## Option 1: 1-Click Run on Kaggle (Recommended)

1. **Open Kaggle**: Go to [kaggle.com/code](https://www.kaggle.com/code) and click **"New Notebook"**.
2. **Add the DFDC Dataset**:
   - In the right-hand panel under **Input**, click **"+ Add Input"**.
   - Search for: `deepfake-detection-challenge` (or [kaggle.com/c/deepfake-detection-challenge/data](https://www.kaggle.com/c/deepfake-detection-challenge/data)).
   - Click **Add**. The 472 GB dataset will instantly appear under `/kaggle/input/deepfake-detection-challenge/`.
3. **Enable GPU**:
   - In the right sidebar under **Settings** -> **Accelerator**, select **GPU T4 x2** or **GPU P100**.
   - Turn **Internet** -> **On**.
4. **Import the Notebook**:
   - In the top menu, click **File** -> **Import Notebook**.
   - Upload [`notebooks/TrustLens_DFDC_Training_Pipeline.ipynb`](file:///e:/GitHub%20Projects/trustlens/notebooks/TrustLens_DFDC_Training_Pipeline.ipynb).
5. **Run**:
   - Click **Run All** (or run cells sequentially).
6. **Download Models**:
   - Once complete, download `/tmp/trustlens_trained_models.zip` from the output files.
   - Unzip the files into your local `backend/models/` folder.

---

## Option 2: Run via Kaggle CLI

If you have your `kaggle.json` API token:

1. Place your `kaggle.json` file in:
   ```powershell
   $HOME\.kaggle\kaggle.json
   ```
2. Install the Kaggle CLI:
   ```powershell
   pip install kaggle
   ```
3. Push and execute the kernel directly:
   ```powershell
   kaggle kernels push -p notebooks/
   ```

---

## Option 3: Google Colab

1. Open [colab.research.google.com](https://colab.research.google.com).
2. Upload `notebooks/TrustLens_DFDC_Training_Pipeline.ipynb`.
3. Select **Runtime** -> **Change runtime type** -> **T4 GPU**.
4. Run all cells. At the end of the notebook, the trained model bundle will automatically download to your browser.

---

## Models Produced in `trustlens_trained_models.zip`

| File | Size (Target) | Architecture | Target Latency (CPU) |
| :--- | :--- | :--- | :--- |
| `edgenet_int8.onnx` | ~14 MB | MobileNetV3-Small + 2D FFT Log-Spectrum + GRU(128) | ~48 ms |
| `clip_head_int8.onnx` | ~28 MB | OpenCLIP ViT-B/32 LayerNorm Tuned Head + Temporal Attention | ~185 ms |
| `wavlm_head_int8.onnx` | ~5.5 MB | WavLM-base-plus Attentive Stats Pooling | ~32 ms |
| `models_registry.json` | ~1.2 KB | Updated SHA-256 hashes and specs (`stubs_in_use: false`) | N/A |
