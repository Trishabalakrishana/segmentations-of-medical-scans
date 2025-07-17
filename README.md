#Segmentations of Medical Scans

This repository contains a comprehensive implementation of **lung and brain region segmentation** using both traditional and deep learning techniques. The project is divided into two major parts:
- **Chest X-ray Segmentation**
- **CT Brain Scan Segmentation**

Each part includes:
- Traditional image segmentation methods
- Deep learning (U-Net)
- U-Net with Boundary Loss
- U-Net with CRF refinement

---

## 📂 Project Structure
segmentations-of-medical-scans/
├── chest_xray/
│ ├── traditional/
│ ├── unet/
│ ├── unet_boundary/
│ └── unet_crf/
│
├── ct_brain/
│ ├── unet/
│ ├── unet_boundary/
│ └── unet_crf/
│
├── data/
├── results/
├── requirements.txt
└── README.md

## Objective

To compare the performance of traditional and deep learning-based segmentation methods on medical imaging datasets, evaluated using:
- Dice Coefficient
- Intersection over Union (IoU)
- Hausdorff Distance


## 📁 Datasets

### 🫁 Chest X-ray Dataset
- Source: https://www.kaggle.com/datasets/nikhilpandey360/chest-xray-masks-and-labels
- Format: PNG images with ground-truth binary lung masks

### 🧠 CT Brain Scan Dataset
- Source: https://physionet.org/content/ct-ich/1.3.1/
- Format*: Slices from volumetric CT scans with binary region masks

---
## Methods Used

### Traditional Methods (Chest X-rays only)
- GraphCut
- GrabCut
- Chan-Vese
- DRLSE

### ✅ Deep Learning Approaches
| Method                    | Used For      | Description |
|---------------------------|---------------|-------------|
| U-Net                     | X-ray + CT    | Basic encoder-decoder with skip   |                                                                      connections |
| U-Net + Boundary Loss     | X-ray + CT    | Adds distance-based loss to refine|                                                                            edges |
| U-Net + CRF               | X-ray + CT    | Applies Conditional Random Fields |                                                                  post-prediction |

---

## 🏁 How to Run

1. **Clone the repository**
```bash
git clone https://github.com/TrishaBalakrishna/segmentations-of-medical-scans.git
cd segmentations-of-medical-scan

python -m venv venv
source venv/bin/activate        # On Linux/Mac
venv\Scripts\activate.bat       # On Windows

python -m venv venv
source venv/bin/activate        # On Linux/Mac
venv\Scripts\activate.bat       # On Windows


pip install -r requirements.txt



