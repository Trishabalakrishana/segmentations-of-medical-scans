import os
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import json
import logging
import time
import traceback

from preprocess import load_image, load_mask
from metrics import metrics
from segmentation import apply_improved_graphcut, apply_improved_grabcut, apply_chan_vese, apply_morph_acwe
from visualization import save_visualization
from tqdm import tqdm


def run_comprehensive_segmentation():
    """Run all classical segmentation algorithms and evaluate them"""
    # Find all images and masks
    image_paths = sorted(glob.glob("data/images/*.png"))
    mask_paths = sorted(glob.glob("data/mask/*.png"))

    if len(image_paths) != len(mask_paths):
        print(f"Warning: {len(image_paths)} images vs {len(mask_paths)} masks")
        matched_pairs = []
        image_names = [os.path.basename(p).split('.')[0] for p in image_paths]
        mask_names = [os.path.basename(p).split('.')[0] for p in mask_paths]
        for img_path, img_name in zip(image_paths, image_names):
            if img_name in mask_names:
                mask_idx = mask_names.index(img_name)
                matched_pairs.append((img_path, mask_paths[mask_idx]))
        if not matched_pairs:
            raise ValueError("No matching image-mask pairs found!")
        print(f"Found {len(matched_pairs)} matching pairs")
    else:
        matched_pairs = list(zip(image_paths, mask_paths))

    # Create output directories
    os.makedirs("outputs/graphcut", exist_ok=True)
    os.makedirs("outputs/grabcut", exist_ok=True)
    os.makedirs("outputs/chanvese", exist_ok=True)
    os.makedirs("outputs/morphacwe", exist_ok=True)
    os.makedirs("outputs/visualizations", exist_ok=True)

    print(f"\nProcessing {len(matched_pairs)} image pairs...\n")

    for img_path, mask_path in tqdm(matched_pairs, desc="Processing images", unit="img"):
        name = os.path.basename(img_path).split('.')[0]

        try:
            img = load_image(img_path)
            gt_mask = load_mask(mask_path)

            # Run GraphCut
            seg_graphcut = apply_improved_graphcut(img)
            metrics.add_scores("graphcut", gt_mask, seg_graphcut)
            cv2.imwrite(f"outputs/graphcut/{name}_graphcut.png", seg_graphcut * 255)

            # Run GrabCut
            seg_grabcut = apply_improved_grabcut(img)
            metrics.add_scores("grabcut", gt_mask, seg_grabcut)
            cv2.imwrite(f"outputs/grabcut/{name}_grabcut.png", seg_grabcut * 255)

            # Run Chan-Vese
            seg_chanvese = apply_chan_vese(img)
            metrics.add_scores("chanvese", gt_mask, seg_chanvese)
            cv2.imwrite(f"outputs/chanvese/{name}_chanvese.png", seg_chanvese * 255)

            # Run morphacwe
            seg_morphacwe = apply_morph_acwe(img)
            metrics.add_scores("morphacwe", gt_mask, seg_morphacwe)
            cv2.imwrite(f"outputs/MorphACWE/{name}_morphacwe.png", seg_morphacwe * 255)  # Optional: change folder to morphacwe

            save_visualization(
                image=img,
                gt_mask=gt_mask,
                seg_chanvese=seg_chanvese,
                seg_morphacwe=seg_morphacwe,
                seg_grabcut=seg_grabcut,
                seg_graphcut=seg_graphcut,
                save_path=f"outputs/visualizations/{name}_comparison.png"
            )

        except Exception as e:
            print(f"[ERROR] Error processing {name}: {str(e)}")
            traceback.print_exc()
            continue

    # Final statistics
    final_results = metrics.compute_statistics()

    with open("outputs/metrics_summary.json", "w") as f:
        json.dump(final_results, f, indent=4)

    print("\nLung segmentation pipeline completed successfully!")
    return final_results


def validate_dataset(image_dir="data/images", mask_dir="data/mask"):
    """Validate dataset structure and image-mask pairs"""
    print("Validating dataset...")

    if not os.path.exists(image_dir):
        print(f"ERROR: Image directory '{image_dir}' not found!")
        return False

    if not os.path.exists(mask_dir):
        print(f"ERROR: Mask directory '{mask_dir}' not found!")
        return False

    image_extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']
    images, masks = [], []

    for ext in image_extensions:
        images.extend(glob.glob(os.path.join(image_dir, f"*{ext}")))
        images.extend(glob.glob(os.path.join(image_dir, f"*{ext.upper()}")))
        masks.extend(glob.glob(os.path.join(mask_dir, f"*{ext}")))
        masks.extend(glob.glob(os.path.join(mask_dir, f"*{ext.upper()}")))

    print(f"Found {len(images)} images and {len(masks)} masks")

    if len(images) == 0:
        print("ERROR: No images found!")
        return False

    if len(masks) == 0:
        print("ERROR: No masks found!")
        return False

    image_names = [os.path.basename(p).split('.')[0] for p in images]
    mask_names = [os.path.basename(p).split('.')[0] for p in masks]
    matched = set(image_names) & set(mask_names)

    print(f"Found {len(matched)} matching image-mask pairs")

    if len(matched) == 0:
        print("ERROR: No matching image-mask pairs found!")
        return False

    print("Validating sample images...")
    for i, name in enumerate(list(matched)[:3]):
        img_path = next(p for p in images if name in p)
        mask_path = next(p for p in masks if name in p)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            print(f"ERROR: Cannot read image or mask for {name}")
            return False

        print(f"Sample {i+1}: {name} - Image: {img.shape}, Mask: {mask.shape}")

    print("Dataset validation PASSED!")
    return True


def create_sample_data(num_samples=5):
    """Create sample synthetic data for testing"""
    print("Creating sample synthetic data...")

    os.makedirs("data/images", exist_ok=True)
    os.makedirs("data/mask", exist_ok=True)

    for i in range(num_samples):
        img = np.zeros((512, 512), dtype=np.uint8)
        noise = np.random.normal(30, 10, (512, 512))
        img = np.clip(img + noise, 0, 255).astype(np.uint8)

        mask = np.zeros((512, 512), dtype=np.uint8)
        left_center = (128 + np.random.randint(-20, 20), 256 + np.random.randint(-30, 30))
        left_axes = (80 + np.random.randint(-10, 10), 120 + np.random.randint(-15, 15))
        cv2.ellipse(img, left_center, left_axes, 0, 0, 360, 180 + np.random.randint(-20, 20), -1)
        cv2.ellipse(mask, left_center, left_axes, 0, 0, 360, 255, -1)

        right_center = (384 + np.random.randint(-20, 20), 256 + np.random.randint(-30, 30))
        right_axes = (80 + np.random.randint(-10, 10), 120 + np.random.randint(-15, 15))
        cv2.ellipse(img, right_center, right_axes, 0, 0, 360, 180 + np.random.randint(-20, 20), -1)
        cv2.ellipse(mask, right_center, right_axes, 0, 0, 360, 255, -1)

        img = cv2.GaussianBlur(img, (5, 5), 1)

        cv2.imwrite(f"data/images/sample_{i:03d}.png", img)
        cv2.imwrite(f"data/mask/sample_{i:03d}.png", mask)

    print(f"Created {num_samples} synthetic lung samples in data/ directory")


def analyze_results(json_path="outputs/metrics_summary.json"):
    """Analyze and visualize segmentation results"""
    if not os.path.exists(json_path):
        print(f"Results file {json_path} not found!")
        return

    with open(json_path, 'r') as f:
        results = json.load(f)

    algorithms = list(results.keys())
    metrics_names = ['dice', 'iou', 'precision', 'recall']

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()

    for i, metric in enumerate(metrics_names):
        means = [results[algo][f'{metric}_mean'] for algo in algorithms]
        stds = [results[algo][f'{metric}_std'] for algo in algorithms]
        x_pos = np.arange(len(algorithms))

        bars = axes[i].bar(x_pos, means, yerr=stds, capsize=5,
                           color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])

        axes[i].set_xlabel('Algorithm')
        axes[i].set_ylabel(metric.capitalize())
        axes[i].set_title(f'{metric.capitalize()} Comparison')
        axes[i].set_xticks(x_pos)
        axes[i].set_xticklabels(algorithms, rotation=45)
        axes[i].grid(True, alpha=0.3)

        for j, (bar, mean, std) in enumerate(zip(bars, means, stds)):
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width() / 2., height,
                         f'{mean:.3f}±{std:.3f}',
                         ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('outputs/metrics_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\n" + "=" * 60)
    print("ALGORITHM RANKING BY METRICS")
    print("=" * 60)

    for metric in metrics_names:
        print(f"\n{metric.upper()} Ranking:")
        metric_results = [(algo, results[algo][f'{metric}_mean']) for algo in algorithms]
        metric_results.sort(key=lambda x: x[1], reverse=True)
        for rank, (algo, score) in enumerate(metric_results, 1):
            print(f"  {rank}. {algo:<15}: {score:.4f}")


def setup_project():
    """Set up project structure"""
    print("Setting up project structure...")

    directories = [
        "data/images",
        "data/mask",
        "outputs/graphcut",
        "outputs/grabcut",
        "outputs/chanvese",
        "outputs/morphACWE",
        "outputs/visualizations"
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")

    print("Project setup complete!")


def install_requirements():
    """Install required packages"""
    import subprocess
    import sys

    requirements = [
        "numpy>=1.21.0",
        "opencv-python>=4.5.0",
        "scikit-image>=0.18.0",
        "scikit-learn>=1.0.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "tqdm>=4.62.0"
    ]

    print("Installing required packages...")
    for req in requirements:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", req])
            print(f"✓ Installed {req}")
        except subprocess.CalledProcessError:
            print(f"✗ Failed to install {req}")

    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "PyMaxflow>=1.2.13"])
        print("✓ Installed PyMaxflow")
    except subprocess.CalledProcessError:
        print("✗ Failed to install PyMaxflow - GraphCut algorithm will not work")
        print("  Try: conda install -c conda-forge pymaxflow")


if __name__ == "__main__":
    print("Utils module loaded successfully!")
    print("Available functions:")
    print("- run_comprehensive_segmentation()")
    print("- validate_dataset()")
    print("- create_sample_data()")
    print("- analyze_results()")
    print("- setup_project()")
    print("- install_requirements()")
