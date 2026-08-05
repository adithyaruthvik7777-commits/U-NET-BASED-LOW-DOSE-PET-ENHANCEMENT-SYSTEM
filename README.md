# Development of a Web-Based Low-Dose PET Image Enhancement System using U-Net

![PET Image Enhancement UI](static/css/banner.png)

A professional, responsive, and medical-grade web application built with **Flask (Python)**, **PyTorch**, and **Bootstrap 5** to convert Low-Dose PET scan images into High-Dose quality PET images using a trained U-Net deep learning model.

---

## 🌟 Key Features

- **Medical AI Web Application**: Professional clinical interface with deep sapphire blue & white healthcare design theme.
- **Drag-and-Drop Dropzone**: Supports DICOM (`.dcm`, `.ima`), NumPy (`.npy`), PNG, and JPG image formats.
- **PyTorch U-Net Inference**: Integrates the trained 7.75M parameter U-Net architecture (`Cell3.py`).
- **Interactive Comparison Slider**: Dual-panel side-by-side comparison with a draggable split slider handle.
- **Real-Time Diagnostic Metrics**: Instant estimation of Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index (SSIM), NRMSE, and Noise Reduction %.
- **Custom Colormap Modes**: Toggle between Grayscale (Standard Radiology), PET Heatmap (Jet), and Thermal Hot Iron views.
- **Complete Waterfall Methodology Page**: Documenting all 7 Waterfall Model development phases.

---

## 📂 Project Structure

```
petnewdataset/
├── app.py                      # Flask Application Server & REST Endpoints
├── unet_model.py               # PyTorch U-Net Model Architecture
├── requirements.txt            # Python Dependencies
├── create_sample_pet_images.py # Helper script to generate sample test images
├── models/
│   └── best_unet_pet.pth       # Trained PyTorch Model Weights (from Kaggle)
├── utils/
│   ├── image_processing.py     # DICOM/PNG loader, normalization, U-Net postprocessing
│   └── metrics.py              # PSNR, SSIM, NRMSE, Noise Reduction calculators
├── static/
│   ├── css/
│   │   └── style.css           # Custom Healthcare UI & Glassmorphism Stylesheet
│   ├── js/
│   │   └── main.js             # Drag-and-drop UI logic & comparison slider
│   ├── uploads/                # Temporary processing directory for uploads & outputs
│   └── sample_images/          # Pre-generated sample PET images
└── templates/
    ├── base.html               # Master Layout with Navbar & Footer
    ├── index.html              # Landing Page
    ├── about.html              # PET Imaging & Radiation Information Page
    ├── how_it_works.html       # 7-Step Workflow Pipeline Page
    ├── enhance.html            # Image Enhancement Interactive Workspace
    ├── model_info.html         # U-Net Neural Architecture & Kaggle Training Page
    ├── methodology.html        # Waterfall Model (Objectives, Activities, Deliverables)
    ├── tech_stack.html         # Technology Stack Cards Page
    └── architecture.html      # End-to-End System Architecture Flowchart
```

---

## Dataset & Model (Kaggle)

- **Dataset:** [skarthik112/paired-low-dose-dataset-1to50](https://www.kaggle.com/datasets/skarthik112/paired-low-dose-dataset-1to50)
  - Low Dose / Full Dose paired PET (1:50), **9006** pairs
  - Split: Train **6304** / Val **1351** / Test **1351**
- **Trained weights:** `best_unet_pet.pth` (~30.8 MB) from notebook output  
  `adithyapanidepu/notebooka154a97ccb`

### Download dataset + export demo pairs
```bash
pip install kagglehub kaggle
python download_kaggle_dataset.py
```

Or download the model from the Kaggle notebook **Output** tab (`best_unet_pet.pth`) into:
`models/best_unet_pet.pth`

```bash
kaggle kernels output adithyapanidepu/notebooka154a97ccb -p ./models
```

---

## 🚀 Quick Start Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Place Model Weights
Copy `best_unet_pet.pth` from your Kaggle notebook output into `models/`.

### 3. (Recommended) Export real Low/Full pairs from the dataset
```bash
python download_kaggle_dataset.py
```

### 4. Run the Web Server
```bash
python app.py
```
Open your browser at: **`http://127.0.0.1:5000`**

---

## 🌊 Software Methodology (Waterfall Model)

This project strictly follows the sequential **Waterfall Model**:
1. **Requirements Analysis**: SRS document & paired dataset specification.
2. **System Planning**: Hardware selection & PyTorch U-Net pipeline roadmap.
3. **System Design**: Layer specifications, REST contract, UI wireframes.
4. **Implementation**: Kaggle PyTorch training & Flask web app coding.
5. **Testing**: PSNR/SSIM evaluation & DICOM parsing validation.
6. **Deployment**: Render / Vercel cloud deployment.
7. **Maintenance**: Monitoring latency & patient data privacy.

---

## 🎓 Academic Attribution

- **Project Title:** Development of a Web-Based Low-Dose PET Image Enhancement System using U-Net
- **Framework:** PyTorch & Flask
- **Primary Model:** 2D U-Net Encoder-Decoder Architecture
