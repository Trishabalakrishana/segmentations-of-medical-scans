import cv2
import numpy as np
from skimage import exposure
from scipy.ndimage import binary_fill_holes

def load_image(path):
    """Preprocess grayscale or RGB chest X-ray image for classical segmentation"""
    img = cv2.imread(path)

    if img is None:
        raise ValueError(f"[ERROR] Could not load image: {path}")

    # Resize to standard size
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LANCZOS4)

    # Convert to RGB if needed
    if len(img.shape) == 2 or img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Denoise using bilateral filtering
    img_denoised = np.zeros_like(img)
    for i in range(3):
        img_denoised[:, :, i] = cv2.bilateralFilter(img[:, :, i], d=9, sigmaColor=75, sigmaSpace=75)

    # Convert to float32 [0, 1]
    img_denoised = img_denoised.astype(np.float32) / 255.0

    # Contrast enhancement per channel
    img_clahe = np.zeros_like(img_denoised)
    for i in range(3):
        img_clahe[:, :, i] = exposure.equalize_adapthist(img_denoised[:, :, i], clip_limit=0.02)

    # Final float image, clipped to [0, 1]
    return np.clip(img_clahe, 0, 1)


def load_mask(path):
    """Preprocess and clean binary ground truth mask"""
    print(f"[INFO] Trying to load mask: {path}")  # Debug log
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise FileNotFoundError(f"[ERROR] Could not read mask at: {path}")

    try:
        # Resize to match image dimensions
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)

        # Binarize
        mask = (mask > 127).astype(np.uint8)

        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Fill holes
        mask = binary_fill_holes(mask).astype(np.uint8)

    except Exception as e:
        raise RuntimeError(f"[ERROR] Mask preprocessing failed for {path} → {str(e)}")

    return mask
