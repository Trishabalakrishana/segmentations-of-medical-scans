# ================================
# crf_refiner.py
# ================================
import numpy as np
import cv2
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax
from typing import Optional, Tuple
import warnings


def apply_crf(
    image: np.ndarray, 
    pred_mask: np.ndarray, 
    n_classes: int = 2, 
    iterations: int = 5,
    confidence_threshold: float = 0.5,
    gaussian_sxy: float = 3.0,
    gaussian_compat: float = 3.0,
    bilateral_sxy: float = 50.0,
    bilateral_srgb: float = 13.0,
    bilateral_compat: float = 10.0,
    eps: float = 1e-7
) -> np.ndarray:
    """
    Apply DenseCRF post-processing to refine segmentation mask.

    Args:
        image (np.ndarray): Original RGB image of shape (H, W, 3), uint8 or float32 [0–255 or 0–1]
        pred_mask (np.ndarray): Raw predicted mask of shape (H, W) with values in [0, 1]
        n_classes (int): Number of classes (default 2 for binary segmentation)
        iterations (int): Number of CRF inference iterations (default 5)
        confidence_threshold (float): Threshold for final binary mask (default 0.5)
        gaussian_sxy (float): Spatial standard deviation for Gaussian pairwise (default 3.0)
        gaussian_compat (float): Compatibility coefficient for Gaussian pairwise (default 3.0)
        bilateral_sxy (float): Spatial standard deviation for bilateral pairwise (default 50.0)
        bilateral_srgb (float): Color standard deviation for bilateral pairwise (default 13.0)
        bilateral_compat (float): Compatibility coefficient for bilateral pairwise (default 10.0)
        eps (float): Small epsilon to avoid numerical issues (default 1e-7)

    Returns:
        np.ndarray: Refined binary segmentation mask of shape (H, W), values in {0, 1}
        
    Raises:
        ValueError: If input dimensions or value ranges are invalid
        RuntimeError: If CRF inference fails
    """

    # ================================
    # 1. Input Validation
    # ================================
    if pred_mask.ndim != 2:
        raise ValueError(f"Expected pred_mask to be 2D, got shape: {pred_mask.shape}")

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected RGB image of shape (H, W, 3), got shape: {image.shape}")

    if image.shape[:2] != pred_mask.shape:
        raise ValueError(
            f"Image and mask dimensions must match. "
            f"Got image: {image.shape[:2]}, mask: {pred_mask.shape}"
        )

    if not (0 <= pred_mask.min() and pred_mask.max() <= 1):
        raise ValueError(
            f"pred_mask values must be in range [0, 1]. "
            f"Got range: [{pred_mask.min():.3f}, {pred_mask.max():.3f}]"
        )

    if not (0 <= confidence_threshold <= 1):
        raise ValueError(f"confidence_threshold must be in [0, 1], got: {confidence_threshold}")

    if iterations < 1:
        raise ValueError(f"iterations must be >= 1, got: {iterations}")

    if n_classes < 2:
        raise ValueError(f"n_classes must be >= 2, got: {n_classes}")

    H, W = pred_mask.shape

    # Warn if mask has very low confidence regions
    low_confidence_ratio = np.mean((pred_mask > 0.1) & (pred_mask < 0.9))
    if low_confidence_ratio > 0.8:
        warnings.warn(
            f"Mask has {low_confidence_ratio:.1%} low-confidence pixels. "
            "CRF may not improve results significantly.",
            UserWarning
        )

    # ================================
    # 2. Normalize Image if Needed
    # ================================
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            # Assume image is in [0, 1] range
            image = np.clip(image * 255, 0, 255).astype(np.uint8)
        else:
            # Assume image is in [0, 255] range but wrong dtype
            image = np.clip(image, 0, 255).astype(np.uint8)

    # ================================
    # 3. Create Softmax Probability Map
    #    (for CRF unary potential)
    # ================================
    probs = np.zeros((n_classes, H, W), dtype=np.float32)
    
    # Add epsilon to avoid numerical issues and ensure valid probabilities
    pred_mask_stable = np.clip(pred_mask, eps, 1 - eps)
    
    if n_classes == 2:
        # Binary segmentation
        probs[0] = 1.0 - pred_mask_stable  # Background class
        probs[1] = pred_mask_stable        # Foreground class
    else:
        # Multi-class segmentation - assume pred_mask contains class probabilities
        if pred_mask.shape != (n_classes, H, W):
            raise ValueError(
                f"For multi-class segmentation, pred_mask should have shape "
                f"({n_classes}, H, W), got: {pred_mask.shape}"
            )
        probs = np.clip(pred_mask, eps, 1 - eps)

    # Ensure probabilities sum to 1 along class dimension
    probs = probs / np.sum(probs, axis=0, keepdims=True)

    # ================================
    # 4. Setup CRF Model
    # ================================
    try:
        d = dcrf.DenseCRF2D(W, H, n_classes)
        unary = unary_from_softmax(probs)  # Convert softmax to unary energy
        d.setUnaryEnergy(unary)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize CRF model: {e}")

    # ================================
    # 5. Add Pairwise Potentials
    # ================================
    try:
        # Gaussian pairwise — encourages spatial smoothness
        d.addPairwiseGaussian(sxy=gaussian_sxy, compat=gaussian_compat)

        # Bilateral pairwise — encourages appearance consistency (color + position)
        d.addPairwiseBilateral(
            sxy=bilateral_sxy, 
            srgb=bilateral_srgb, 
            rgbim=image, 
            compat=bilateral_compat
        )
    except Exception as e:
        raise RuntimeError(f"Failed to add pairwise potentials: {e}")

    # ================================
    # 6. Inference
    # ================================
    try:
        Q = d.inference(iterations)
        refined_probs = np.array(Q).reshape((n_classes, H, W))
        
        if n_classes == 2:
            # For binary segmentation, apply threshold to foreground class
            refined_mask = (refined_probs[1] > confidence_threshold).astype(np.uint8)
        else:
            # For multi-class, take argmax
            refined_mask = np.argmax(refined_probs, axis=0).astype(np.uint8)
            
    except Exception as e:
        raise RuntimeError(f"CRF inference failed: {e}")

    return refined_mask


def apply_crf_with_confidence(
    image: np.ndarray, 
    pred_mask: np.ndarray, 
    return_confidence: bool = False,
    **kwargs
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Apply CRF and optionally return confidence scores.
    
    Args:
        image (np.ndarray): Original RGB image
        pred_mask (np.ndarray): Raw predicted mask
        return_confidence (bool): Whether to return confidence scores
        **kwargs: Additional arguments passed to apply_crf
        
    Returns:
        Tuple[np.ndarray, Optional[np.ndarray]]: 
            - Refined mask
            - Confidence scores (if return_confidence=True)
    """
    # Store original parameters
    original_n_classes = kwargs.get('n_classes', 2)
    
    # Run CRF
    refined_mask = apply_crf(image, pred_mask, **kwargs)
    
    if return_confidence:
        # Re-run inference to get confidence scores
        H, W = pred_mask.shape
        probs = np.zeros((original_n_classes, H, W), dtype=np.float32)
        eps = kwargs.get('eps', 1e-7)
        
        pred_mask_stable = np.clip(pred_mask, eps, 1 - eps)
        probs[0] = 1.0 - pred_mask_stable
        probs[1] = pred_mask_stable
        probs = probs / np.sum(probs, axis=0, keepdims=True)
        
        d = dcrf.DenseCRF2D(W, H, original_n_classes)
        unary = unary_from_softmax(probs)
        d.setUnaryEnergy(unary)
        
        # Add same pairwise potentials
        d.addPairwiseGaussian(
            sxy=kwargs.get('gaussian_sxy', 3.0), 
            compat=kwargs.get('gaussian_compat', 3.0)
        )
        
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = np.clip(image * 255, 0, 255).astype(np.uint8)
            else:
                image = np.clip(image, 0, 255).astype(np.uint8)
        
        d.addPairwiseBilateral(
            sxy=kwargs.get('bilateral_sxy', 50.0), 
            srgb=kwargs.get('bilateral_srgb', 13.0), 
            rgbim=image, 
            compat=kwargs.get('bilateral_compat', 10.0)
        )
        
        Q = d.inference(kwargs.get('iterations', 5))
        confidence_scores = np.array(Q).reshape((original_n_classes, H, W))
        
        return refined_mask, confidence_scores[1]  # Return foreground confidence
    
    return refined_mask, None


# ================================
# Utility Functions
# ================================

def get_default_crf_params(image_size: Tuple[int, int], detail_level: str = 'medium') -> dict:
    """
    Get recommended CRF parameters based on image size and desired detail level.
    
    Args:
        image_size (Tuple[int, int]): (Height, Width) of the image
        detail_level (str): 'low', 'medium', or 'high' detail preservation
        
    Returns:
        dict: Dictionary of CRF parameters
    """
    H, W = image_size
    scale_factor = np.sqrt(H * W) / 512  # Normalize to 512x512 baseline
    
    if detail_level == 'low':
        return {
            'gaussian_sxy': 3.0 * scale_factor,
            'gaussian_compat': 3.0,
            'bilateral_sxy': 80.0 * scale_factor,
            'bilateral_srgb': 20.0,
            'bilateral_compat': 10.0,
            'iterations': 5
        }
    elif detail_level == 'high':
        return {
            'gaussian_sxy': 1.0 * scale_factor,
            'gaussian_compat': 1.0,
            'bilateral_sxy': 20.0 * scale_factor,
            'bilateral_srgb': 5.0,
            'bilateral_compat': 5.0,
            'iterations': 10
        }
    else:  # medium
        return {
            'gaussian_sxy': 3.0 * scale_factor,
            'gaussian_compat': 3.0,
            'bilateral_sxy': 50.0 * scale_factor,
            'bilateral_srgb': 13.0,
            'bilateral_compat': 10.0,
            'iterations': 5
        }


# ================================
# Example Usage
# ================================

if __name__ == "__main__":
    # Example usage
    import matplotlib.pyplot as plt
    
    # Create dummy data
    H, W = 256, 256
    image = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
    pred_mask = np.random.rand(H, W).astype(np.float32)
    
    # Apply CRF with default parameters
    refined_mask = apply_crf(image, pred_mask)
    
    # Apply CRF with custom parameters for high detail
    custom_params = get_default_crf_params((H, W), detail_level='high')
    refined_mask_detailed = apply_crf(image, pred_mask, **custom_params)
    
    # Apply CRF with confidence scores
    refined_mask_conf, confidence = apply_crf_with_confidence(
        image, pred_mask, return_confidence=True
    )
    
    print(f"Original mask range: [{pred_mask.min():.3f}, {pred_mask.max():.3f}]")
    print(f"Refined mask unique values: {np.unique(refined_mask)}")
    if confidence is not None:
        print(f"Confidence range: [{confidence.min():.3f}, {confidence.max():.3f}]")