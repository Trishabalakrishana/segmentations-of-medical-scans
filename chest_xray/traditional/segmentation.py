import os
import glob
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.segmentation import morphological_chan_vese

# -------------------- Image Preprocessing --------------------
def preprocess_for_segmentation(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image.copy()
    
    # Histogram equalization to improve contrast
    equalized = cv2.equalizeHist(gray)
    
    # Gaussian smoothing to reduce noise
    blurred = cv2.GaussianBlur(equalized, (5, 5), 0)
    
    return blurred

# -------------------- Simple Image Loading --------------------
def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not load image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not load mask: {path}")
    return (mask > 127).astype(np.uint8)


# -------------------- Simple Metrics --------------------
def calculate_dice(gt, pred):
    intersection = np.logical_and(gt, pred).sum()
    return 2.0 * intersection / (gt.sum() + pred.sum() + 1e-8)


def calculate_iou(gt, pred):
    intersection = np.logical_and(gt, pred).sum()
    union = np.logical_or(gt, pred).sum()
    return intersection / (union + 1e-8)


# -------------------- Segmentation Methods --------------------
import cv2
import numpy as np

# -------------------- Image Preprocessing --------------------
def preprocess_for_segmentation(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image.copy()
    
    # Histogram Equalization (contrast boost)
    equalized = cv2.equalizeHist(gray)

    # Gaussian smoothing
    blurred = cv2.GaussianBlur(equalized, (5, 5), 0)

    return blurred


# -------------------- GrabCut --------------------
def apply_improved_grabcut(image):
    print("    Starting GrabCut...")

    preprocessed = preprocess_for_segmentation(image)

    # Convert to 3-channel BGR
    if len(preprocessed.shape) == 2:
        img_bgr = cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2BGR)
    else:
        img_bgr = cv2.cvtColor(preprocessed, cv2.COLOR_RGB2BGR)

    h, w = img_bgr.shape[:2]
    mask = np.zeros((h, w), np.uint8)

    # Rect: slightly inside border
    margin_h, margin_w = h // 10, w // 10
    rect = (margin_w, margin_h, w - 2 * margin_w, h - 2 * margin_h)

    bgModel = np.zeros((1, 65), np.float64)
    fgModel = np.zeros((1, 65), np.float64)

    cv2.grabCut(img_bgr, mask, rect, bgModel, fgModel, 5, cv2.GC_INIT_WITH_RECT)

    # Create binary mask: 1 for foreground
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.uint8)
    print(f"    GrabCut result: {mask2.sum()} foreground pixels")

    return mask2


# -------------------- GraphCut via Watershed --------------------
def apply_improved_graphcut(image):
    print("    Starting GraphCut (Watershed)...")

    preprocessed = preprocess_for_segmentation(image)
    gray = preprocessed

    print(f"    Gray image range: {gray.min()}-{gray.max()}")

    # Thresholding
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(f"    Binary pixels: {binary.sum() // 255}")

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    unknown = cv2.subtract(sure_bg, sure_fg)

    # Label markers
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    if len(image.shape) == 3:
        img_color = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        img_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    cv2.watershed(img_color, markers)

    # Output mask
    mask = np.where(markers > 1, 1, 0).astype(np.uint8)
    print(f"    Watershed result: {mask.sum()} foreground pixels")

    return mask


# -------------------- Chan-Vese Inspired --------------------
def apply_chan_vese(image, iterations=50):
    print("    Starting Chan-Vese...")

    preprocessed = preprocess_for_segmentation(image)
    gray = preprocessed.astype(np.float32) / 255.0
    h, w = gray.shape

    # Initial mask: center square
    mask = np.zeros((h, w), dtype=np.uint8)
    ch, cw = h // 2, w // 2
    size = min(h, w) // 8
    mask[ch - size:ch + size, cw - size:cw + size] = 1

    for i in range(iterations):
        fg_mean = gray[mask == 1].mean() if mask.sum() > 0 else 0.5
        bg_mean = gray[mask == 0].mean() if (mask == 0).sum() > 0 else 0.5

        new_mask = np.where(np.abs(gray - fg_mean) < np.abs(gray - bg_mean), 1, 0).astype(np.uint8)

        if np.array_equal(mask, new_mask):
            break
        mask = new_mask

    print(f"    Chan-Vese result: {mask.sum()} foreground pixels")
    return mask


# -------------------- DRLSE (Basic Level Set) --------------------
def apply_morph_acwe(image, iterations=200):
    print("    Starting Morphological Chan-Vese (MorphACWE)...")

    preprocessed = preprocess_for_segmentation(image)
    img = preprocessed.astype(np.float32) / 255.0

    init_ls = np.zeros_like(img)
    cx, cy = img.shape[1] // 2, img.shape[0] // 2
    cv2.circle(init_ls, (cx, cy), min(img.shape) // 4, 1, -1)

    seg = morphological_chan_vese(img, iterations=iterations, init_level_set=init_ls)

    print(f"    MorphACWE result: {seg.sum()} foreground pixels")
    return seg.astype(np.uint8)





# -------------------- Visualization --------------------
def visualize(image, gt, results, name, scores):
    print(f"    Creating visualization for {name}...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')
    
    # Ground truth
    axes[0, 1].imshow(gt, cmap='gray')
    axes[0, 1].set_title("Ground Truth")
    axes[0, 1].axis('off')
    
    # Results
    method_names = list(results.keys())
    positions = [(0, 2), (1, 0), (1, 1), (1, 2)]
    
    for i, method in enumerate(method_names):
        if i < len(positions):
            row, col = positions[i]
            axes[row, col].imshow(results[method], cmap='gray')
            
            if method in scores:
                dice, iou = scores[method]
                title = f"{method}\nDice: {dice:.3f} | IoU: {iou:.3f}"
            else:
                title = method
            
            axes[row, col].set_title(title)
            axes[row, col].axis('off')
    
    # Hide unused subplots
    for i in range(len(method_names), 4):
        if i < len(positions):
            row, col = positions[i]
            axes[row, col].axis('off')
    
    plt.tight_layout()
    
    # Save
    os.makedirs("outputs/visualizations", exist_ok=True)
    plt.savefig(f"outputs/visualizations/{name}_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"    Visualization saved for {name}")


# -------------------- Main Runner --------------------
def run_classical_segmentation():
    print("Starting Classical Segmentation Pipeline...")

    # Find images and masks
    image_paths = sorted(glob.glob("data/images/*"))
    mask_paths = sorted(glob.glob("data/mask/*"))

    print(f"Found {len(image_paths)} images")
    print(f"Found {len(mask_paths)} masks")

    if len(image_paths) == 0:
        print("ERROR: No images found in data/images/")
        return

    if len(mask_paths) == 0:
        print("ERROR: No masks found in data/mask/")
        return

    # Create output directories
    os.makedirs("outputs", exist_ok=True)
    for method in ["grabcut", "graphcut", "chanvese", "drlse"]:
        os.makedirs(f"outputs/{method}", exist_ok=True)

    # Store results
    all_results = {
        "GrabCut": {"dice": [], "iou": []},
        "GraphCut": {"dice": [], "iou": []},
        "ChanVese": {"dice": [], "iou": []},
        "DRLSE": {"dice": [], "iou": []}
    }

    # Process each image
    for i, (img_path, mask_path) in enumerate(zip(image_paths, mask_paths)):
        name = os.path.basename(img_path).split('.')[0]
        print(f"\n[{i+1}/{len(image_paths)}] Processing {name}...")

        try:
            # Load data
            print(f"  Loading image: {img_path}")
            img = load_image(img_path)
            print(f"  Image shape: {img.shape}")

            print(f"  Loading mask: {mask_path}")
            gt = load_mask(mask_path)
            print(f"  Mask shape: {gt.shape}, GT pixels: {gt.sum()}")

        except Exception as e:
            print(f"  ERROR loading {name}: {e}")
            continue

        results = {}
        scores = {}

        # Method 1: GrabCut
        try:
            pred = apply_improved_grabcut(img)
            pred_resized = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(f"outputs/grabcut/{name}.png", pred_resized * 255)

            dice = calculate_dice(gt, pred_resized)
            iou = calculate_iou(gt, pred_resized)
            scores["GrabCut"] = (dice, iou)
            results["GrabCut"] = pred_resized

            all_results["GrabCut"]["dice"].append(dice)
            all_results["GrabCut"]["iou"].append(iou)

            print(f"  GrabCut: Dice={dice:.3f}, IoU={iou:.3f}")

        except Exception as e:
            print(f"  GrabCut FAILED: {e}")

        # Method 2: GraphCut
        try:
            pred = apply_improved_graphcut(img)
            pred_resized = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(f"outputs/graphcut/{name}.png", pred_resized * 255)

            dice = calculate_dice(gt, pred_resized)
            iou = calculate_iou(gt, pred_resized)
            scores["GraphCut"] = (dice, iou)
            results["GraphCut"] = pred_resized

            all_results["GraphCut"]["dice"].append(dice)
            all_results["GraphCut"]["iou"].append(iou)

            print(f"  GraphCut: Dice={dice:.3f}, IoU={iou:.3f}")

        except Exception as e:
            print(f"  GraphCut FAILED: {e}")

        # Method 3: Chan-Vese
        try:
            pred = apply_chan_vese(img)
            pred_resized = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(f"outputs/chanvese/{name}.png", pred_resized * 255)

            dice = calculate_dice(gt, pred_resized)
            iou = calculate_iou(gt, pred_resized)
            scores["ChanVese"] = (dice, iou)
            results["ChanVese"] = pred_resized

            all_results["ChanVese"]["dice"].append(dice)
            all_results["ChanVese"]["iou"].append(iou)

            print(f"  ChanVese: Dice={dice:.3f}, IoU={iou:.3f}")

        except Exception as e:
            print(f"  ChanVese FAILED: {e}")

# Method 4: MorphACWE (replacing DRLSE)
        try:
            pred = apply_morph_acwe(img)
            
            # Validate prediction output
            if pred is None:
                raise ValueError("MorphACWE returned None")
            
            # Ensure pred is binary and in correct format
            if pred.dtype != np.uint8:
                pred = (pred > 0.5).astype(np.uint8)
            
            # Normalize to 0-1 range if needed
            if pred.max() > 1:
                pred = pred / 255.0
            
            pred_resized = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            # Ensure output directory exists
            os.makedirs("outputs/drlse", exist_ok=True)
            
            # Convert to proper format for saving
            save_pred = (pred_resized * 255).astype(np.uint8)
            cv2.imwrite(f"outputs/drlse/{name}.png", save_pred)

            dice = calculate_dice(gt, pred_resized)
            iou = calculate_iou(gt, pred_resized)

            scores["MorphACWE"] = (dice, iou)
            results["MorphACWE"] = pred_resized

            all_results["MorphACWE"]["dice"].append(dice)
            all_results["MorphACWE"]["iou"].append(iou)

            print(f"  MorphACWE: Dice={dice:.3f}, IoU={iou:.3f}")

        except Exception as e:
            print(f"  MorphACWE FAILED: {e}")
            # Add fallback values to maintain consistency
            scores["MorphACWE"] = (0.0, 0.0)
            results["MorphACWE"] = np.zeros_like(gt)
            all_results["MorphACWE"]["dice"].append(0.0)
            all_results["MorphACWE"]["iou"].append(0.0)

        # Create visualization
        if results:
            try:
                visualize(img, gt, results, name, scores)
            except Exception as e:
                print(f"  Visualization FAILED: {e}")

        print(f"  Completed {name}")

    # Print final statistics
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)

    for method_name, metrics in all_results.items():
        if metrics["dice"]:
            avg_dice = np.mean(metrics["dice"])
            avg_iou = np.mean(metrics["iou"])
            print(f"{method_name:>10}: Dice={avg_dice:.4f}, IoU={avg_iou:.4f} ({len(metrics['dice'])} images)")
        else:
            print(f"{method_name:>10}: No successful results")

    print("\nSegmentation complete!")
    return all_results



if __name__ == "__main__":
    try:
        run_classical_segmentation()
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()