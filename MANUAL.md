# 🌿 Leaf micro-CT Scan Segmentation — User Manual

**Version 2.0 | June 2026**  
**Author:** Worasit Sangjan  
**Live App:** [https://huggingface.co/spaces/WorasitSangjan/Leaf-CT-Segmentation](https://huggingface.co/spaces/WorasitSangjan/Leaf-CT-Segmentation)

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Accessing the Application](#2-accessing-the-application)
3. [Interface Overview](#3-interface-overview)
4. [Single Image Segmentation](#4-single-image-segmentation)
5. [Stack Image Segmentation](#5-stack-image-segmentation)
6. [Understanding the Output Files](#6-understanding-the-output-files)
7. [Model Information](#7-model-information)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Introduction

The Leaf micro-CT Scan Segmentation Web Application is an open-source, browser-based tool that automates the identification and measurement of leaf tissue regions in X-ray micro-computed tomography (micro-CT) images. The application requires no programming skills and is accessible to plant biologists, agronomists, and phenotyping researchers.

The tool offers a choice of four transformer-based deep learning models — **Mask2Former (Swin-B)**, **SegFormer-B4**, **EoMT (DINOv3 ViT-L)**, and **FPN (MiT-B4)** — selectable from a radio button panel. Each model segments every image pixel into one of five anatomical classes, then calculates area and volume statistics for quantitative analysis. See [Section 7](#7-model-information) for model recommendations.

### 1.1 What This Tool Does

- Accepts single micro-CT slice images (PNG, JPG, or TIF) or multi-slice TIFF stacks.
- Automatically segments five tissue classes: Background, Epidermis, Vascular Region, Mesophyll, and Air Space.
- Produces three output visualizations: a class label mask, a color-coded mask, and an overlay on the original image.
- Exports pixel-level area statistics and, for stacks, full volumetric statistics as downloadable CSV files.

### 1.2 System Requirements

No software installation is required to use the hosted version. For running on Google Colab or locally, the following are needed:

| Requirement | Details |
|---|---|
| **Python** | 3.10 or newer |
| **GPU (optional)** | NVIDIA GPU with CUDA (strongly recommended for stacks); CPU is supported but slower |
| **Browser** | Any modern browser (Chrome, Firefox, Safari, Edge) |
| **Internet** | Required for HuggingFace Spaces and Google Colab deployments |

---

## 2. Accessing the Application

The application can be launched in three ways. Choose the option that best fits your needs.

### 2.1 Option 1 — HuggingFace Spaces (CPU, No Setup)

> **Best for:** Quick exploration, small single images, no installation needed.

Visit the live app directly in your web browser — no account or installation required:  
👉 [https://huggingface.co/spaces/WorasitSangjan/Leaf-CT-Segmentation](https://huggingface.co/spaces/WorasitSangjan/Leaf-CT-Segmentation)

Processing runs on a shared CPU server. Inference is functional but slower than GPU-accelerated options, particularly for large images or TIFF stacks.

---

### 2.2 Option 2 — Google Colab (GPU, Recommended)

> **Best for:** Processing TIFF stacks or large images; free T4 GPU available.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/WorasitSangjan/WebApp-Leaf-microCT-Segmentation/blob/main/run_colab.ipynb)

1. Click the **Open in Colab** button above.
2. Go to **Runtime → Change runtime type** and select **T4 GPU**. Click Save.
3. Click **Runtime → Run all** (or press `Ctrl+F9` on Windows / `Cmd+F9` on Mac).
4. Wait for all cells to finish. A public URL ending in `.gradio.live` will appear in the output of the last cell.
5. Click the `.gradio.live` URL to open the application in your browser.

> **Note:** The console also shows a local URL (`http://127.0.0.1:7860`). This points to Google's internal server and will **not** work from your computer. Always use the `.gradio.live` link.

---

### 2.3 Option 3 — Run Locally (Python 3.10+)

> **Best for:** Offline use, institutional data that cannot leave your machine, repeated large-batch processing.

**Prerequisites:**
- Python 3.10+: [https://www.python.org/downloads/](https://www.python.org/downloads/)
- Git: [https://git-scm.com/downloads/](https://git-scm.com/downloads/)

**Step 1 — Download the repository**
```bash
git clone https://github.com/WorasitSangjan/WebApp-Leaf-microCT-Segmentation.git
cd WebApp-Leaf-microCT-Segmentation
```

**Step 2 — Create a virtual environment**
```bash
# Mac / Linux
python3 -m venv venv && source venv/bin/activate

# Windows
python -m venv venv && venv\Scripts\activate
```

**Step 3 — Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 4 — Start the application**
```bash
python app.py
```

When the terminal shows `Running on local URL: http://127.0.0.1:7860/`, open that address in your browser.  
Press `Ctrl+C` to stop the application.

---

## 3. Interface Overview

When the application loads, the main screen contains three areas:

| Area | Description |
|---|---|
| **Header** | App title, short description, deployment option links, and a **📖 User Manual** link (this document). |
| **Segmentation Model** | Radio button panel to choose which model runs (see [Section 7](#7-model-information)). |
| **Status** | Shows the current state (e.g., *Ready*, *Image uploaded*, *Segmentation complete*). |
| **Main Block** | Two tabs: **Single Image** (one slice at a time) and **Stack Image** (multi-page TIFF volumes). |

---

## 4. Single Image Segmentation

Use this tab to segment one CT slice at a time. The complete workflow time depends on image resolution and available hardware.

<img src="./docs/fig1_single_image.png" alt="Figure 1 — Single Image Interface" width="700">

*Figure 1: Graphical user interface of the Leaf CT Scan Segmentation website for a single image.*

### Step 1 — Upload Your Image

Click the **Upload CT Scan** drop-zone or drag and drop your image file onto it.

| Format | Details |
|---|---|
| `.png` | Standard PNG — most common format for exported CT slices. |
| `.jpg / .jpeg` | JPEG compression may slightly affect segmentation precision at tissue boundaries. |
| `.tif / .tiff` | Single-page TIFF. For multi-page volumetric TIFFs, use the **Stack Image** tab. |

After upload, a preview appears in the left panel and the Status bar changes to *Image uploaded — click Run Segmentation to process.*

### Step 2 — (Optional) Load an Example Image

Five built-in example species are available: **Lantana, Olive, Pine, Viburnum, and Wheat**. Click the species name button below its thumbnail to load it.

### Step 3 — Run Segmentation

Click the green **Run Segmentation** button. The model will:

- Convert the image to grayscale and normalize intensity values.
- Divide the image into overlapping **320 × 320-pixel patches** (stride = 80 pixels).
- Run each patch through the selected neural network.
- Stitch predictions back together using **Gaussian-weighted averaging** to minimize seam artifacts.
- Assign each pixel the class with the highest probability score.

The Status bar updates to *Done: Segmentation complete* when finished.

### Step 4 — Explore the Results

Three visualizations appear in the tabbed results panel:

| Result Tab | Description |
|---|---|
| **Class Label** | Grayscale mask where pixel intensity encodes class index (0–4). Background = 0, Epidermis = 50, Vascular Region = 100, Mesophyll = 150, Air Space = 200. |
| **Color Mask** | Per-class color visualization. Background = black, Epidermis = red, Vascular Region = green, Mesophyll = blue, Air Space = yellow. |
| **Overlay** | Original greyscale CT image blended with the color mask at 50% opacity. Useful for verifying tissue boundaries. |

### Step 5 — Review Area Statistics

Scroll down to the **Area Statistics** table for pixel counts and percentages per class.

### Step 6 — Download Outputs

| # | File | Contents |
|---|---|---|
| 1 | **Area Statistics** | CSV: class name, pixel count, and percentage. |
| 2 | **Label Mask** | PNG: grayscale integer label mask. |
| 3 | **Color Mask** | PNG: RGB color-coded segmentation mask. |
| 4 | **Overlay Image** | PNG: original image blended with color mask. |

### Clearing the Workspace

Click **Clear** to reset all inputs, previews, and outputs. Useful when processing a new image without refreshing the browser.

---

## 5. Stack Image Segmentation

The **Stack Image** tab processes multi-slice TIFF volumes, producing per-slice visualizations and aggregated volumetric statistics. Each slice is segmented independently using the same patch-based inference pipeline as Single Image mode.

<img src="./docs/fig2_stack_image.png" alt="Figure 2 — Stack Image Interface" width="700">

*Figure 2: Graphical user interface of the Leaf CT Scan Segmentation website for a stack image.*

### Step 1 — Upload Your TIFF Stack

Click the **Upload CT Stack** drop-zone or drag and drop your multi-page TIFF (`.tif` / `.tiff`). Single-channel (grayscale) stacks are expected.

After upload, the Status bar displays the number of slices detected (e.g., *Stack uploaded (45 slices)*).

### Step 2 — (Optional) Load an Example Stack

Three built-in example stacks are available: **Arabidopsis, Grape, and Oak**. Click the species name button to load it.

### Step 3 — Run Segmentation

Click **Run Segmentation on Stack**. The model processes every slice sequentially.

> **Tip:** For a 45-slice stack at 1024 × 1024 pixels, GPU processing takes ~2–5 minutes. CPU processing may take 15–30 minutes or more. Google Colab (Option 2) is strongly recommended for stacks.

### Step 4 — Explore the Results

The three result tabs display the segmentation of the **middle slice** as a representative preview.

### Step 5 — Review Volume Statistics

The **Volume Statistics (All slices)** table aggregates voxel counts and volume percentages across the entire stack — the primary outputs for 3D phenotyping workflows.

### Step 6 — Download Outputs

| # | File | Contents |
|---|---|---|
| 1 | **Volume Stats (CSV)** | Per-class voxel counts and volume % across all slices. |
| 2 | **Stats Per Slice (CSV)** | Per-class pixel counts and % for every individual slice. |
| 3 | **Label Stack (TIFF)** | Multi-page TIFF of grayscale label masks, one page per slice. |
| 4 | **Color Stack (TIFF)** | Multi-page TIFF of RGB color masks, one page per slice. |
| 5 | **Overlay Stack (TIFF)** | Multi-page TIFF of overlaid images, one page per slice. |

---

## 6. Understanding the Output Files

### 6.1 CSV Statistics Files

Output CSVs can be opened in Excel, Google Sheets, R, or Python/Pandas.

**Area Statistics (Single Image)**

| Column | Example | Meaning |
|---|---|---|
| Class | Mesophyll | Tissue class name. |
| Pixels | 245,032 | Total pixels assigned to this class. |
| Percentage | 38.24% | Proportion of total image area. |

**Volume Statistics (Stack)**

| Column | Example | Meaning |
|---|---|---|
| Class | Air Space | Tissue class name. |
| Voxels | 1,204,800 | Total voxels across all slices. |
| Volume % | 22.51% | Volumetric proportion across the full stack. |

**Per-Slice Statistics (Stack)**

| Column | Example | Type | Meaning |
|---|---|---|---|
| Slice | 12 | Integer | Slice number (1-indexed). |
| Class | Epidermis | String | Tissue class name. |
| Pixels | 14,420 | Integer | Pixel count for this class in this slice. |
| Percentage | 5.62% | Float | Fraction of total pixels in the slice. |

### 6.2 Reading the Label Mask in Python

The label mask stores class indices as pixel intensities × 50 (values: 0, 50, 100, 150, 200). To recover integer class indices:

```python
import numpy as np
from PIL import Image

label_img = Image.open("label.png").convert("L")
class_index = np.array(label_img) // 50  # values 0–4
```

For multi-page TIFF stacks, use the `tifffile` or `imageio` library to iterate through frames.

---

## 7. Model Information

The application provides four interchangeable segmentation models, selectable from the **Segmentation Model** radio panel. All share the same input pipeline, 5-class output, and patch-based inference; they differ in architecture, size, and performance characteristics.

### 7.1 Model Recommendations

| Model | Overall mIoU | Best for | Notes |
|---|---|---|---|
| **Mask2Former (Swin-B) ★** | 0.809 | General use · Mesophyll · Air Space | Highest overall accuracy; recommended default |
| **SegFormer-B4** | 0.808 | Vascular tissue | Highest Vascular Region IoU; best for water transport / venation studies |
| **EoMT (DINOv3 ViT-L)** | 0.801 | Epidermis | Highest Epidermis IoU and precision; best for epidermal thickness measurement |
| **FPN (MiT-B4)** | 0.800 | Lightweight alternative | Highest Vascular Region precision (fewest false positives); smallest download |

> All four models are within 0.009 mIoU of each other. For most users, **Mask2Former (Swin-B)** is the best starting point. Switch to a specialist model only when a specific tissue class is the primary measurement target.

### 7.2 Architecture Details

| Model | Architecture | Backbone | Weights file | Approx. size |
|---|---|---|---|---|
| **Mask2Former (Swin-B)** | Mask2Former | Swin-Base (window 12) | `Mask2Former_SwinB.pth` | ~1.6 GB |
| **SegFormer-B4** | SegFormer | MiT-B4 | `Segformer_B4.pth` | ~490 MB |
| **EoMT (DINOv3 ViT-L)** | Encoder-only Mask Transformer | DINOv3 ViT-L/16 | `EoMT_DinoV3.pth` | ~3.5 GB |
| **FPN (MiT-B4)** | Feature Pyramid Network | MiT-B4 | `FPN_MiTB4.pth` | ~480 MB |

**Shared properties**

| Parameter | Details |
|---|---|
| **Input channels** | 1 (grayscale). EoMT and SegFormer use a 1×3 conv adapter; Mask2Former repeats the channel to RGB; FPN's MiT-B4 encoder takes 1 channel directly. |
| **Patch size** | 320 × 320 px (stride = 80 px during inference) |
| **Number of classes** | 5 (Background, Epidermis, Vascular Region, Mesophyll, Air Space) |
| **Model weights** | [WorasitSangjan/Leaf-CT-Segmentation-Model](https://huggingface.co/WorasitSangjan/Leaf-CT-Segmentation-Model) |
| **Hardware target** | CUDA GPU → Apple MPS → CPU (automatic detection) |
| **Inference strategy** | Gaussian-weighted patch stitching to reduce boundary artifacts |

> **Note:** The selected model's weights are downloaded automatically from the HuggingFace Hub the first time it is chosen, then cached. Switching models the first time triggers a new download. An internet connection is required for the initial download of each model.

---

## 8. Troubleshooting

| Problem | Solution |
|---|---|
| **First segmentation takes a long time** | The selected model's weights (~0.5–3.5 GB depending on model) download on first use. Wait for completion — the same model is cached for subsequent runs. Switching to a model you haven't used yet triggers a new download. |
| **Inference is very slow on CPU** | Use Google Colab (Option 2) or a local GPU machine for large images or stacks. |
| **Upload fails or no preview shown** | Check the file format (PNG, JPG, TIF). For stack mode, the file must be a multi-page TIFF. Ensure the file is not corrupted. |
| **"Demo mode — no model loaded" error** | The selected model's weights could not be downloaded. Confirm the corresponding `.pth` file exists in the weights repo, check your internet connection, and refresh. If on HuggingFace Spaces, the server may be under load — try Google Colab instead, or select a smaller model. |
| **Segmentation looks incorrect** | The model was trained on ALS beamline 8.3.2 images. Images from different scanners or acquisition settings may require fine-tuning for optimal accuracy. |
| **Gradio link expired (Colab)** | The `.gradio.live` URL expires after ~72 hours of inactivity. Re-run the last cell in the Colab notebook to generate a new link. |
| **Out of memory error** | Reduce image size before uploading, or use a GPU with more VRAM. Very large images (> 4096 px) may require > 4 GB VRAM. |

---

*Leaf micro-CT Scan Segmentation Web Application | Version 2.0 | June 2026*
