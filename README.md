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

## Project Structure
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


## Datasets

### Chest X-ray Dataset
- Source: https://www.kaggle.com/datasets/nikhilpandey360/chest-xray-masks-and-labels
- Format: PNG images with ground-truth binary lung masks

### CT Brain Scan Dataset
- Source: https://physionet.org/content/ct-ich/1.3.1/
- Format*: Slices from volumetric CT scans with binary region masks

---
## Methods Used

### Traditional Methods (Chest X-rays only)
- GraphCut
- GrabCut
- Chan-Vese
- DRLSE

### Deep Learning Approaches
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

## Results(chest x-ray)

| Method                    | Dice Score | IoU Score | Hausdorff Distance |
|---------------------------|------------|-----------|--------------------|
| Traditional Ensemble      | 0.52       | 0.43      | 30.2               |
| U-Net                     | 0.94       | 0.91      | 20.5               |
| U-Net + Boundary Loss     | 0.98       | 0.95      | 15.4               |
| U-Net + CRF Refinement    | 0.91       | 0.82      | 15.2               |

---

## Results (CT Brain)

| Method                         | Dice Score | IoU Score | Hausdorff Distance |
|--------------------------------|------------|-----------|--------------------|
| U-Net                          | 0.015      | 0.013     | 100.2              |
| U-Net + Boundary loss          | 0.015      | 0.013     | 100.2              |
| U-Net + CRF Refinement         | 0.015      | 0.013     | 100.2              |



