# ================================
# predict_and_crf.py - Enhanced Version
# ================================
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import os
import argparse
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List, Union
import time
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import json
import warnings
from tqdm import tqdm
import gc

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

try:
    from model import UNET # Support both model versions
except ImportError:
    print("Warning: Could not import model classes. Make sure model.py is available.")
    UNET = None
    UNET_V2 = None

try:
    from crf_refiner import apply_crf, get_default_crf_params, apply_crf_with_confidence
except ImportError:
    print("Warning: Could not import CRF functions. CRF refinement will be disabled.")
    apply_crf = None
    get_default_crf_params = None
    apply_crf_with_confidence = None

# Configure logging with better formatting
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('prediction.log')
    ]
)
logger = logging.getLogger(__name__)

# ----------------------------
# Enhanced Configuration Class
# ----------------------------
import pydensecrf.densecrf as dcrf
import numpy as np
import cv2

def apply_crf_to_prediction(image: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """
    Apply DenseCRF to refine the predicted mask.

    Args:
        image (np.ndarray): Original RGB image (H, W, 3).
        prediction (np.ndarray): Predicted mask (H, W) with values in [0, 255].

    Returns:
        np.ndarray: Refined mask (H, W), values in [0, 255].
    """
    h, w = image.shape[:2]
    image = np.ascontiguousarray(image)
    prediction = prediction.astype(np.uint8)

    d = dcrf.DenseCRF2D(w, h, 2)

    # Unary potentials (negative log probability)
    probs = prediction.astype(np.float32) / 255.0
    probs = np.stack([1 - probs, probs], axis=0)
    unary = -np.log(probs + 1e-8)
    unary = unary.reshape((2, -1))
    d.setUnaryEnergy(unary)

    # Add pairwise terms
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=image, compat=10)

    # Inference
    Q = d.inference(5)
    refined_mask = np.argmax(Q, axis=0).reshape((h, w)).astype(np.uint8) * 255

    return refined_mask



from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any

@dataclass
class PredictionConfig:
    """Enhanced configuration for the prediction pipeline"""
    
    # Model settings
    model_path: str = "best_model.pth"
    model_type: str = "UNET"  # "UNET" or "UNET_V2"
    model_channels: Tuple[int, int] = (3, 1)
    
    # Input/Output settings
    image_path: str = "sample_input.png"
    save_dir: str = "outputs"
    image_size: Tuple[int, int] = (256, 256)
    maintain_aspect_ratio: bool = True

    # CRF settings
    apply_crf: bool = True  # ✅ Add this field to fix the error
    use_crf: bool = True
    crf_detail_level: str = "medium"  # "low", "medium", "high", "custom"
    custom_crf_params: Optional[Dict[str, Any]] = None
    confidence_threshold: float = 0.5

    # Processing settings
    device: str = "auto"
    batch_size: int = 1
    num_workers: int = 4
    use_mixed_precision: bool = True

    # Output settings
    save_raw_mask: bool = True
    save_overlay: bool = True
    save_confidence: bool = False
    save_probabilities: bool = False
    overlay_alpha: float = 0.5
    overlay_color: Tuple[int, int, int] = (255, 0, 0)

    # Advanced settings
    batch_processing: bool = False
    recursive_search: bool = False
    supported_formats: Tuple[str, ...] = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp')
    output_format: str = 'png'
    compression_level: int = 5

    # Post-processing
    apply_morphology: bool = False
    morphology_kernel_size: int = 3
    morphology_iterations: int = 1
    apply_gaussian_blur: bool = False
    gaussian_sigma: float = 1.0

    # Evaluation settings
    ground_truth_dir: Optional[str] = None
    calculate_metrics: bool = False

    # Performance settings
    profile_performance: bool = False
    memory_efficient: bool = False

    def __post_init__(self):
        if self.image_size[0] <= 0 or self.image_size[1] <= 0:
            raise ValueError("Image size must be positive")
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError("Confidence threshold must be between 0 and 1")
        if not 0 <= self.overlay_alpha <= 1:
            raise ValueError("Overlay alpha must be between 0 and 1")

# ----------------------------
# Enhanced Utility Functions
# ----------------------------
def get_device(device_preference: str = "auto") -> torch.device:
    """Get the appropriate device for inference with enhanced detection"""
    if device_preference == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name()
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"Using GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Using Apple Metal Performance Shaders (MPS)")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU")
    else:
        device = torch.device(device_preference)
        logger.info(f"Using specified device: {device}")
    
    return device

def preprocess_image(
    image_path: str, 
    size: Tuple[int, int] = (256, 256),
    maintain_aspect_ratio: bool = True
) -> Tuple[torch.Tensor, np.ndarray, Dict[str, Any]]:
    """
    Enhanced image preprocessing with aspect ratio preservation
    
    Args:
        image_path: Path to input image
        size: Target size (H, W)
        maintain_aspect_ratio: Whether to maintain aspect ratio
        
    Returns:
        Tuple of (tensor for model, original image for CRF, metadata)
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    original_shape = image.shape[:2]
    original_image = image.copy()
    
    # Resize with aspect ratio preservation if requested
    if maintain_aspect_ratio:
        h, w = original_shape
        target_h, target_w = size
        
        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize
        image_resized = cv2.resize(image, (new_w, new_h))
        
        # Create padded image
        padded_image = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        start_y = (target_h - new_h) // 2
        start_x = (target_w - new_w) // 2
        padded_image[start_y:start_y + new_h, start_x:start_x + new_w] = image_resized
        image_resized = padded_image
        
        padding_info = {
            'scale': scale,
            'new_size': (new_w, new_h),
            'padding': (start_x, start_y, target_w - new_w - start_x, target_h - new_h - start_y)
        }
    else:
        image_resized = cv2.resize(image, size)
        padding_info = {'scale': 1.0, 'new_size': size, 'padding': (0, 0, 0, 0)}
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    
    # Normalize
    image_normalized = image_rgb.astype(np.float32) / 255.0
    
    # Convert to tensor
    tensor = torch.tensor(image_normalized).permute(2, 0, 1).unsqueeze(0)
    
    metadata = {
        'original_shape': original_shape,
        'processed_shape': image_resized.shape[:2],
        'padding_info': padding_info,
        'file_size': os.path.getsize(image_path)
    }
    
    return tensor, image_rgb, metadata

def postprocess_prediction(
    pred: torch.Tensor, 
    apply_sigmoid: bool = True,
    apply_softmax: bool = False
) -> np.ndarray:
    """
    Enhanced postprocessing with multiple activation options
    """
    if apply_softmax:
        pred = F.softmax(pred, dim=1)
    elif apply_sigmoid:
        pred = torch.sigmoid(pred)
    
    pred_np = pred.squeeze().cpu().numpy()
    pred_np = np.clip(pred_np, 0, 1)
    
    return pred_np

def apply_morphological_operations(
    mask: np.ndarray,
    kernel_size: int = 3,
    iterations: int = 1
) -> np.ndarray:
    """Apply morphological operations to clean up the mask"""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Apply opening followed by closing
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    
    return mask_clean

def create_enhanced_overlay(
    image: np.ndarray, 
    mask: np.ndarray, 
    alpha: float = 0.5,
    color: Tuple[int, int, int] = (255, 0, 0),
    blend_mode: str = 'normal'
) -> np.ndarray:
    """
    Create enhanced overlay with multiple blend modes
    """
    colored_mask = np.zeros_like(image)
    colored_mask[mask > 0] = color
    
    if blend_mode == 'normal':
        overlay = cv2.addWeighted(image.astype(np.uint8), 1 - alpha, colored_mask, alpha, 0)
    elif blend_mode == 'multiply':
        overlay = (image.astype(np.float32) * colored_mask.astype(np.float32) / 255.0).astype(np.uint8)
    elif blend_mode == 'screen':
        overlay = (255 - (255 - image.astype(np.float32)) * (255 - colored_mask.astype(np.float32)) / 255.0).astype(np.uint8)
    else:
        overlay = cv2.addWeighted(image.astype(np.uint8), 1 - alpha, colored_mask, alpha, 0)
    
    return overlay

def calculate_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Dict[str, float]:
    """Calculate segmentation metrics"""
    pred_binary = (pred_mask > 0.5).astype(np.uint8)
    gt_binary = (gt_mask > 0.5).astype(np.uint8)
    
    # Calculate intersection and union
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union = np.logical_or(pred_binary, gt_binary).sum()
    
    # IoU
    iou = intersection / (union + 1e-7)
    
    # Dice coefficient
    dice = 2 * intersection / (pred_binary.sum() + gt_binary.sum() + 1e-7)
    
    # Precision, Recall, F1
    tp = intersection
    fp = pred_binary.sum() - tp
    fn = gt_binary.sum() - tp
    
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)
    
    return {
        'iou': float(iou),
        'dice': float(dice),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1)
    }

@contextmanager
def performance_monitor(operation_name: str):
    """Context manager for performance monitoring"""
    start_time = time.time()
    start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    
    try:
        yield
    finally:
        end_time = time.time()
        end_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        logger.info(f"{operation_name} - Time: {end_time - start_time:.3f}s, "
                   f"Memory: {(end_memory - start_memory) / 1024**2:.1f} MB")

def save_enhanced_results(
    config: PredictionConfig,
    base_name: str,
    original_image: np.ndarray,
    pred_mask: np.ndarray,
    refined_mask: np.ndarray,
    metadata: Dict[str, Any],
    confidence_scores: Optional[np.ndarray] = None,
    probabilities: Optional[np.ndarray] = None,
    metrics: Optional[Dict[str, float]] = None,
    inference_time: float = 0.0
) -> Dict[str, str]:
    """Enhanced result saving with more options"""
    os.makedirs(config.save_dir, exist_ok=True)
    saved_files = {}
    
    # Determine output format settings
    if config.output_format.lower() == 'png':
        ext = '.png'
        save_params = [cv2.IMWRITE_PNG_COMPRESSION, config.compression_level]
    elif config.output_format.lower() == 'jpg':
        ext = '.jpg'
        save_params = [cv2.IMWRITE_JPEG_QUALITY, 95]
    else:
        ext = '.tiff'
        save_params = []
    
    # Save raw prediction mask
    if config.save_raw_mask:
        raw_mask_path = os.path.join(config.save_dir, f"{base_name}_raw{ext}")
        cv2.imwrite(raw_mask_path, (pred_mask * 255).astype(np.uint8), save_params)
        saved_files['raw_mask'] = raw_mask_path
    
    # Save CRF-refined mask
    refined_mask_path = os.path.join(config.save_dir, f"{base_name}_refined{ext}")
    cv2.imwrite(refined_mask_path, (refined_mask * 255).astype(np.uint8), save_params)
    saved_files['refined_mask'] = refined_mask_path
    
    # Save overlay
    if config.save_overlay:
        overlay = create_enhanced_overlay(
            original_image, 
            refined_mask, 
            alpha=config.overlay_alpha,
            color=config.overlay_color
        )
        overlay_path = os.path.join(config.save_dir, f"{base_name}_overlay{ext}")
        cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR), save_params)
        saved_files['overlay'] = overlay_path
    
    # Save confidence scores
    if config.save_confidence and confidence_scores is not None:
        confidence_path = os.path.join(config.save_dir, f"{base_name}_confidence{ext}")
        cv2.imwrite(confidence_path, (confidence_scores * 255).astype(np.uint8), save_params)
        saved_files['confidence'] = confidence_path
    
    # Save probabilities
    if config.save_probabilities and probabilities is not None:
        prob_path = os.path.join(config.save_dir, f"{base_name}_probabilities{ext}")
        cv2.imwrite(prob_path, (probabilities * 255).astype(np.uint8), save_params)
        saved_files['probabilities'] = prob_path
    
    # Save detailed metadata as JSON
    metadata_path = os.path.join(config.save_dir, f"{base_name}_metadata.json")
    full_metadata = {
        'input': {
            'image_path': config.image_path,
            'original_shape': metadata['original_shape'],
            'file_size_mb': metadata['file_size'] / (1024 * 1024)
        },
        'model': {
            'model_path': config.model_path,
            'model_type': config.model_type,
            'image_size': config.image_size,
            'device': str(config.device)
        },
        'processing': {
            'use_crf': config.use_crf,
            'crf_detail_level': config.crf_detail_level,
            'confidence_threshold': config.confidence_threshold,
            'maintain_aspect_ratio': config.maintain_aspect_ratio
        },
        'performance': {
            'inference_time_seconds': inference_time,
            'pixels_per_second': np.prod(config.image_size) / inference_time if inference_time > 0 else 0
        },
        'statistics': {
            'raw_mask': {
                'min': float(pred_mask.min()),
                'max': float(pred_mask.max()),
                'mean': float(pred_mask.mean()),
                'std': float(pred_mask.std())
            },
            'refined_mask': {
                'unique_values': [int(x) for x in np.unique(refined_mask)],
                'foreground_pixels': int(np.sum(refined_mask > 0)),
                'background_pixels': int(np.sum(refined_mask == 0))
            }
        }
    }
    
    if metrics:
        full_metadata['metrics'] = metrics
    
    with open(metadata_path, 'w') as f:
        json.dump(full_metadata, f, indent=2)
    
    saved_files['metadata'] = metadata_path
    
    return saved_files

# ----------------------------
# Enhanced Prediction Pipeline
# ----------------------------
class EnhancedSegmentationPredictor:
    """
    Enhanced segmentation predictor with advanced features
    """

    def __init__(self, config: PredictionConfig):
        self.config = config
        self.device = get_device(config.device)
        self.model = None
        self.scaler = torch.cuda.amp.GradScaler() if config.use_mixed_precision else None

    def load_model(self) -> None:
        """Load the trained model with enhanced error handling"""
        import os
        if not os.path.exists(self.config.model_path):
            raise FileNotFoundError(f"Model not found: {self.config.model_path}")

        logger.info(f"Loading model from: {self.config.model_path}")

        # Initialize model based on type
        if self.config.model_type == "UNET" and UNET is not None:
            self.model = UNET(
                in_channels=self.config.model_channels[0],
                out_channels=self.config.model_channels[1]
            )
        elif self.config.model_type == "UNET_V2" and UNET_V2 is not None:
            self.model = UNET_V2(
                in_channels=self.config.model_channels[0],
                out_channels=self.config.model_channels[1]
            )
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")

        try:
            checkpoint = torch.load(self.config.model_path, map_location=self.device)

            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint

            self.model.load_state_dict(state_dict)

            logger.info(f"Loaded checkpoint - Epoch: {checkpoint.get('epoch', 'unknown')}, "
                        f"Loss: {checkpoint.get('loss', 'unknown')}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

        self.model.to(self.device)
        self.model.eval()

        # ✅ Fixed location
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Model loaded successfully - Total params: {total_params:,}, "
                    f"Trainable: {trainable_params:,}")

    def predict_single_image(self, image_path: str) -> Dict[str, Any]:
        """Enhanced single image prediction"""
        logger.info(f"Processing image: {Path(image_path).name}")
        
        with performance_monitor("Single Image Prediction"):
            # Preprocess image
            input_tensor, original_image, metadata = preprocess_image(
                image_path, 
                size=self.config.image_size,
                maintain_aspect_ratio=self.config.maintain_aspect_ratio
            )
            input_tensor = input_tensor.to(self.device)
            
            # Model inference
            start_time = time.time()
            with torch.no_grad():
                if self.config.use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        pred = self.model(input_tensor)
                else:
                    pred = self.model(input_tensor)
                
                pred_mask = postprocess_prediction(pred)
            
            inference_time = time.time() - start_time
            
            # Memory cleanup
            if self.config.memory_efficient:
                del input_tensor, pred
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                gc.collect()
            
            # Apply CRF refinement
            refined_mask = pred_mask.copy()
            confidence_scores = None
            
            if self.config.use_crf and apply_crf is not None:
                logger.info("Applying CRF refinement...")
                crf_start_time = time.time()
                
                if self.config.custom_crf_params:
                    crf_params = self.config.custom_crf_params
                else:
                    crf_params = get_default_crf_params(
                        self.config.image_size, 
                        detail_level=self.config.crf_detail_level
                    )
                
                crf_params['confidence_threshold'] = self.config.confidence_threshold
                
                try:
                    if self.config.save_confidence and apply_crf_with_confidence is not None:
                        refined_mask, confidence_scores = apply_crf_with_confidence(
                            image=original_image,
                            pred_mask=pred_mask,
                            return_confidence=True,
                            **crf_params
                        )
                    else:
                        refined_mask = apply_crf(
                            image=original_image,
                            pred_mask=pred_mask,
                            **crf_params
                        )
                except Exception as e:
                    logger.warning(f"CRF refinement failed: {e}, using thresholded prediction")
                    refined_mask = (pred_mask > self.config.confidence_threshold).astype(np.uint8)
                
                crf_time = time.time() - crf_start_time
                inference_time += crf_time
                logger.info(f"CRF refinement completed in {crf_time:.3f}s")
            else:
                refined_mask = (pred_mask > self.config.confidence_threshold).astype(np.uint8)
            
            # Apply morphological operations
            if self.config.apply_morphology:
                refined_mask = apply_morphological_operations(
                    refined_mask,
                    kernel_size=self.config.morphology_kernel_size,
                    iterations=self.config.morphology_iterations
                )
            
            # Apply Gaussian blur if requested
            if self.config.apply_gaussian_blur:
                refined_mask = cv2.GaussianBlur(
                    refined_mask.astype(np.float32),
                    (0, 0),
                    self.config.gaussian_sigma
                )
                refined_mask = (refined_mask > 0.5).astype(np.uint8)
            
            # Calculate metrics if ground truth is available
            metrics = None
            if self.config.calculate_metrics and self.config.ground_truth_dir:
                gt_path = os.path.join(
                    self.config.ground_truth_dir, 
                    Path(image_path).stem + '.png'
                )
                if os.path.exists(gt_path):
                    gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                    if gt_mask is not None:
                        gt_mask = gt_mask / 255.0
                        metrics = calculate_metrics(refined_mask, gt_mask)
            
            # Save results
            base_name = Path(image_path).stem
            saved_files = save_enhanced_results(
                config=self.config,
                base_name=base_name,
                original_image=original_image,
                pred_mask=pred_mask,
                refined_mask=refined_mask,
                metadata=metadata,
                confidence_scores=confidence_scores,
                probabilities=pred_mask if self.config.save_probabilities else None,
                metrics=metrics,
                inference_time=inference_time
            )
            
            # Log results
            logger.info(f"✅ Processing completed in {inference_time:.3f}s")
            if metrics:
                logger.info(f"   Metrics - IoU: {metrics['iou']:.3f}, Dice: {metrics['dice']:.3f}")
            
            return {
                'pred_mask': pred_mask,
                'refined_mask': refined_mask,
                'confidence_scores': confidence_scores,
                'metadata': metadata,
                'metrics': metrics,
                'inference_time': inference_time,
                'saved_files': saved_files
            }

    def predict_batch(self, image_dir: str) -> Dict[str, Dict[str, Any]]:
        """Enhanced batch processing with progress tracking"""
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"Directory not found: {image_dir}")
        
        # Find all supported image files
        image_files = []
        search_pattern = "**/*" if self.config.recursive_search else "*"
        
        for ext in self.config.supported_formats:
            image_files.extend(Path(image_dir).glob(f"{search_pattern}{ext}"))
            image_files.extend(Path(image_dir).glob(f"{search_pattern}{ext.upper()}"))
        
        image_files = list(set(image_files))  # Remove duplicates
        
        if not image_files:
            raise ValueError(f"No supported image files found in {image_dir}")
        
        logger.info(f"Found {len(image_files)} images to process")
        
        results = {}
        total_start_time = time.time()
        
        # Process with progress bar
        with tqdm(total=len(image_files), desc="Processing images") as pbar:
            if self.config.num_workers > 1:
                # Parallel processing
                with ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
                    future_to_file = {
                        executor.submit(self.predict_single_image, str(image_file)): image_file
                        for image_file in image_files
                    }
                    
                    for future in as_completed(future_to_file):
                        image_file = future_to_file[future]
                        try:
                            result = future.result()
                            results[image_file.name] = result
                        except Exception as e:
                            logger.error(f"Failed to process {image_file.name}: {e}")
                            results[image_file.name] = {'error': str(e)}
                        pbar.update(1)
            else:
                # Sequential processing
                for image_file in image_files:
                    try:
                        result = self.predict_single_image(str(image_file))
                        results[image_file.name] = result
                    except Exception as e:
                        logger.error(f"Failed to process {image_file.name}: {e}")
                        results[image_file.name] = {'error': str(e)}
                    pbar.update(1)
        
        total_time = time.time() - total_start_time
        successful_predictions = sum(1 for r in results.values() if 'error' not in r)
        
        logger.info(f"Batch processing completed in {total_time:.3f}s")
        logger.info(f"Successfully processed {successful_predictions}/{len(image_files)} images")
        
        # Save batch summary
        summary_path = os.path.join(self.config.save_dir, "batch_summary.json")
        summary = {
            'total_images': len(image_files),
            'successful_predictions': successful_predictions,
            'total_time_seconds': total_time,
            'average_time_per_image': total_time / len(image_files),
            'config': self.config.__dict__,
            'results': {k: v.get('metrics', {}) for k, v in results.items() if 'error' not in v}
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        return results

# ----------------------------
# Enhanced Main Function
# ----------------------------
def main():
    """Enhanced main function with comprehensive argument parsing"""
    parser = argparse.ArgumentParser(
        description="Enhanced segmentation prediction with CRF refinement",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model settings
    model_group = parser.add_argument_group('Model Settings')
    model_group.add_argument("--model_path", type=str, default="best_model.pth", 
                            help="Path to trained model")
    model_group.add_argument("--model_type", type=str, default="UNET", 
                            choices=["UNET", "UNET_V2"], help="Model architecture type")
    model_group.add_argument("--model_channels", type=int, nargs=2, default=[3, 1],
                            help="Model input and output channels")
    
    # Input/Output settings
    io_group = parser.add_argument_group('Input/Output Settings')
    io_group.add_argument("--image_path", type=str, default="sample_input.png", 
                         help="Path to input image or directory")
    io_group.add_argument("--save_dir", type=str, default="outputs", 
                         help="Directory to save results")
    io_group.add_argument("--image_size", type=int, nargs=2, default=[256, 256], 
                         help="Image size (H W)")
    io_group.add_argument("--maintain_aspect_ratio", action="store_true", 
                         help="Maintain aspect ratio during resizing")
    io_group.add_argument("--output_format", type=str, default="png", 
                         choices=["png", "jpg", "tiff"], help="Output image format")
    io_group.add_argument("--compression_level", type=int, default=5, 
                         help="Compression level for PNG (0-9)")
    
    # CRF settings
    crf_group = parser.add_argument_group('CRF Settings')
    crf_group.add_argument("--no_crf", action="store_true", help="Disable CRF refinement")
    crf_group.add_argument("--crf_detail", type=str, default="medium", 
                          choices=["low", "medium", "high", "custom"], help="CRF detail level")
    crf_group.add_argument("--threshold", type=float, default=0.5, 
                          help="Confidence threshold")
    crf_group.add_argument("--custom_crf_config", type=str, 
                          help="Path to custom CRF configuration JSON file")
    
    # Processing settings
    process_group = parser.add_argument_group('Processing Settings')
    process_group.add_argument("--device", type=str, default="auto", 
                              choices=["auto", "cpu", "cuda", "mps"], help="Device to use")
    process_group.add_argument("--batch_size", type=int, default=1, 
                              help="Batch size for processing")
    process_group.add_argument("--num_workers", type=int, default=4, 
                              help="Number of worker threads for batch processing")
    process_group.add_argument("--mixed_precision", action="store_true", 
                              help="Use mixed precision inference")
    process_group.add_argument("--memory_efficient", action="store_true", 
                              help="Enable memory efficient processing")
    
    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument("--no_raw_mask", action="store_true", 
                             help="Don't save raw prediction mask")
    output_group.add_argument("--no_overlay", action="store_true", 
                             help="Don't save overlay images")
    output_group.add_argument("--save_confidence", action="store_true", 
                             help="Save confidence scores")
    output_group.add_argument("--save_probabilities", action="store_true", 
                             help="Save probability maps")
    output_group.add_argument("--overlay_alpha", type=float, default=0.5, 
                             help="Overlay transparency (0-1)")
    output_group.add_argument("--overlay_color", type=int, nargs=3, default=[255, 0, 0],
                             help="Overlay color (R G B)")
    
    # Post-processing options
    post_group = parser.add_argument_group('Post-processing Options')
    post_group.add_argument("--apply_morphology", action="store_true", 
                           help="Apply morphological operations")
    post_group.add_argument("--morphology_kernel_size", type=int, default=3, 
                           help="Morphology kernel size")
    post_group.add_argument("--morphology_iterations", type=int, default=1, 
                           help="Morphology iterations")
    post_group.add_argument("--apply_gaussian_blur", action="store_true", 
                           help="Apply Gaussian blur to final mask")
    post_group.add_argument("--gaussian_sigma", type=float, default=1.0, 
                           help="Gaussian blur sigma")
    
    # Batch processing
    batch_group = parser.add_argument_group('Batch Processing')
    batch_group.add_argument("--batch", action="store_true", 
                            help="Process all images in directory")
    batch_group.add_argument("--recursive", action="store_true", 
                            help="Search recursively in subdirectories")
    
    # Evaluation options
    eval_group = parser.add_argument_group('Evaluation Options')
    eval_group.add_argument("--ground_truth_dir", type=str, 
                           help="Directory containing ground truth masks")
    eval_group.add_argument("--calculate_metrics", action="store_true", 
                           help="Calculate segmentation metrics")
    
    # Advanced options
    advanced_group = parser.add_argument_group('Advanced Options')
    advanced_group.add_argument("--profile", action="store_true", 
                               help="Enable performance profiling")
    advanced_group.add_argument("--config_file", type=str, 
                               help="Path to configuration file")
    advanced_group.add_argument("--log_level", type=str, default="INFO",
                               choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                               help="Logging level")
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Load configuration from file if provided
    if args.config_file and os.path.exists(args.config_file):
        with open(args.config_file, 'r') as f:
            config_dict = json.load(f)
        logger.info(f"Loaded configuration from {args.config_file}")
    else:
        config_dict = {}
    
    # Load custom CRF parameters if provided
    custom_crf_params = None
    if args.custom_crf_config and os.path.exists(args.custom_crf_config):
        with open(args.custom_crf_config, 'r') as f:
            custom_crf_params = json.load(f)
        logger.info(f"Loaded custom CRF configuration from {args.custom_crf_config}")
    
    # Create configuration with precedence: command line > config file > defaults
    config = PredictionConfig(
        # Model settings
        model_path=args.model_path,
        model_type=args.model_type,
        model_channels=tuple(args.model_channels),
        
        # Input/Output settings
        image_path=args.image_path,
        save_dir=args.save_dir,
        image_size=tuple(args.image_size),
        maintain_aspect_ratio=args.maintain_aspect_ratio,
        output_format=args.output_format,
        compression_level=args.compression_level,
        
        # CRF settings
        use_crf=not args.no_crf,
        crf_detail_level=args.crf_detail,
        custom_crf_params=custom_crf_params,
        confidence_threshold=args.threshold,
        
        # Processing settings
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_mixed_precision=args.mixed_precision,
        memory_efficient=args.memory_efficient,
        
        # Output settings
        save_raw_mask=not args.no_raw_mask,
        save_overlay=not args.no_overlay,
        save_confidence=args.save_confidence,
        save_probabilities=args.save_probabilities,
        overlay_alpha=args.overlay_alpha,
        overlay_color=tuple(args.overlay_color),
        
        # Post-processing
        apply_morphology=args.apply_morphology,
        morphology_kernel_size=args.morphology_kernel_size,
        morphology_iterations=args.morphology_iterations,
        apply_gaussian_blur=args.apply_gaussian_blur,
        gaussian_sigma=args.gaussian_sigma,
        
        # Batch processing
        batch_processing=args.batch,
        recursive_search=args.recursive,
        
        # Evaluation
        ground_truth_dir=args.ground_truth_dir,
        calculate_metrics=args.calculate_metrics,
        
        # Advanced
        profile_performance=args.profile,
        
        # Override with config file values
        **config_dict
    )
    
    # Validate configuration
    try:
        # Check if model file exists
        if not os.path.exists(config.model_path):
            raise FileNotFoundError(f"Model file not found: {config.model_path}")
        
        # Check if input path exists
        if not os.path.exists(config.image_path):
            raise FileNotFoundError(f"Input path not found: {config.image_path}")
        
        # Validate batch processing requirements
        if config.batch_processing and not os.path.isdir(config.image_path):
            raise ValueError("Batch processing requires a directory path")
        
        # Validate ground truth directory
        if config.calculate_metrics and not config.ground_truth_dir:
            raise ValueError("Ground truth directory required for metric calculation")
        
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        return 1
    
    # Log configuration summary
    logger.info("=== Configuration Summary ===")
    logger.info(f"Model: {config.model_path} ({config.model_type})")
    logger.info(f"Input: {config.image_path}")
    logger.info(f"Output: {config.save_dir}")
    logger.info(f"Image Size: {config.image_size}")
    logger.info(f"Device: {config.device}")
    logger.info(f"CRF: {'Enabled' if config.use_crf else 'Disabled'}")
    logger.info(f"Batch Processing: {'Enabled' if config.batch_processing else 'Disabled'}")
    logger.info("=============================")
    
    # Initialize predictor
    try:
        predictor = EnhancedSegmentationPredictor(config)
        predictor.load_model()
    except Exception as e:
        logger.error(f"Failed to initialize predictor: {e}")
        return 1
    
    # Run prediction
    try:
        if config.batch_processing:
            logger.info("Starting batch processing...")
            results = predictor.predict_batch(config.image_path)
            
            # Log batch results summary
            successful = sum(1 for r in results.values() if 'error' not in r)
            total = len(results)
            logger.info(f"Batch processing completed: {successful}/{total} images processed successfully")
            
            # Calculate average metrics if available
            if config.calculate_metrics:
                all_metrics = [r['metrics'] for r in results.values() 
                             if 'metrics' in r and r['metrics'] is not None]
                if all_metrics:
                    avg_metrics = {
                        metric: np.mean([m[metric] for m in all_metrics])
                        for metric in all_metrics[0].keys()
                    }
                    logger.info("Average metrics across all images:")
                    for metric, value in avg_metrics.items():
                        logger.info(f"  {metric}: {value:.4f}")
        else:
            logger.info("Starting single image processing...")
            result = predictor.predict_single_image(config.image_path)
            logger.info("Single image processing completed successfully")
            
            if result.get('metrics'):
                logger.info("Segmentation metrics:")
                for metric, value in result['metrics'].items():
                    logger.info(f"  {metric}: {value:.4f}")
    
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return 1
    
    logger.info("All processing completed successfully!")
    return 0

# ----------------------------
# Additional Utility Functions
# ----------------------------
def create_sample_config():
    """Create a sample configuration file"""
    sample_config = {
        "model_path": "best_model.pth",
        "model_type": "UNET",
        "image_size": [512, 512],
        "use_crf": True,
        "crf_detail_level": "high",
        "confidence_threshold": 0.6,
        "save_confidence": True,
        "apply_morphology": True,
        "morphology_kernel_size": 5,
        "overlay_alpha": 0.3,
        "overlay_color": [0, 255, 0]
    }
    
    with open("sample_config.json", "w") as f:
        json.dump(sample_config, f, indent=2)
    
    print("Sample configuration saved to sample_config.json")

def create_sample_crf_config():
    """Create a sample CRF configuration file"""
    sample_crf_config = {
        "bilateral_spatial_sigma": 80,
        "bilateral_color_sigma": 13,
        "bilateral_iterations": 10,
        "gaussian_spatial_sigma": 3,
        "gaussian_iterations": 2,
        "confidence_threshold": 0.5,
        "use_bilateral": True,
        "use_gaussian": True
    }
    
    with open("sample_crf_config.json", "w") as f:
        json.dump(sample_crf_config, f, indent=2)
    
    print("Sample CRF configuration saved to sample_crf_config.json")

# ----------------------------
# Entry Point
# ----------------------------
if __name__ == "__main__":
    import sys
    
    # Handle special commands
    if len(sys.argv) > 1:
        if sys.argv[1] == "create_sample_config":
            create_sample_config()
            sys.exit(0)
        elif sys.argv[1] == "create_sample_crf_config":
            create_sample_crf_config()
            sys.exit(0)
    
    # Run main function
    sys.exit(main())