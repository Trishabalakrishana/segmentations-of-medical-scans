import matplotlib.pyplot as plt
import numpy as np
import os

def save_visualization(image, gt_mask, seg_chanvese, seg_morphacwe, seg_grabcut, seg_graphcut, save_path):
    """Create and save visualization comparing all classical segmentation methods"""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    titles = [
        'Original Image', 'Ground Truth', 'Chan-Vese',
        'MorphACWE', 'GrabCut', 'GraphCut'
    ]
    images = [
        image, gt_mask, seg_chanvese,
        seg_morphacwe, seg_grabcut, seg_graphcut
    ]

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i], cmap='gray')
        ax.set_title(titles[i])
        ax.axis('off')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
