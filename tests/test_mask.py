# import cv2
# import matplotlib.pyplot as plt
# import numpy as np

# # Load the mask as a grayscale image
# img = cv2.imread('C:\GitHub\PseudoSegWSSS\datasets\BCSS\masks\TCGA-A1-A0SK-DX1_xmin45749_ymin25055_MAG-10.00.png', cv2.IMREAD_GRAYSCALE)
# mask = cv2.imread('C:\GitHub\PseudoSegWSSS\datasets\BCSS\masks\TCGA-A1-A0SK-DX1_xmin45749_ymin25055_MAG-10.00.png', cv2.IMREAD_GRAYSCALE)


# # Quick trick: Multiply by a constant to stretch the values across the 0-255 spectrum
# # Or use a color map to visualize the distinct classes
# plt.imshow(mask, cmap='jet')
# plt.colorbar()
# plt.show()

# plt.imshow(img, cmap='jet')
# plt.colorbar()
# plt.show()


import random
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# BCSS Dataset class mapping template (adjust if your pipeline uses a custom ordering)
BCSS_CLASSES = {
    0: "Outside / Background",
    1: "Tumor (Epithelium)",
    2: "Stroma (Lamina Propria)",
    3: "Inflammatory Infiltrate",
    4: "Necrosis / Debris",
    5: "Glandular / Secretions",
    6: "Blood / Vessel",
    7: "Metaplasia"
}

# 1. Define paths
base_dir = Path(r"C:\GitHub\PseudoSegWSSS\outputs\patches\TCGA-A1-A0SK-DX1_xmin45749_ymin25055_MAG-10.00")
img_dir = base_dir / "imgs"
mask_dir = base_dir / "mask"

# 2. Gather and sort all files
img_files = sorted(list(img_dir.glob("*.png")))
mask_files = sorted(list(mask_dir.glob("*.png")))

total_patches = min(len(img_files), len(mask_files))


if total_patches == 0:
    print("❌ No images or masks found.")
else:
    # 3. Pick 5 random unique indices
    num_to_show = min(5, total_patches)
    random_indices = random.sample(range(total_patches), num_to_show)

    # Increased width slightly (from 7 to 9) to accommodate the legend labels cleanly
    fig, axes = plt.subplots(num_to_show, 2, figsize=(9, 3 * num_to_show))
    
    if num_to_show == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, idx in enumerate(random_indices):
        img_path = img_files[idx]
        mask_path = mask_files[idx]
        
        # Load images
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        # --- Left Column: Image ---
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Img: {img_path.name}", fontsize=9)
        axes[i, 0].axis('off')
        
        # --- Right Column: Mask ---
        # Using 'tab10' colormap for distinct categorical colors
        cmap = plt.colormaps['tab10']
        im_mask = axes[i, 1].imshow(mask, cmap=cmap, vmin=0, vmax=9) 
        axes[i, 1].set_title(f"Mask: {mask_path.name}", fontsize=9)
        axes[i, 1].axis('off')
        
        # --- Dynamic Legend Generation ---
        # Find which unique class values actually exist in this specific patch
        unique_classes = np.unique(mask)
        
        legend_handles = []
        for cls_id in unique_classes:
            # Match class mapping string, default to "Unknown" if not in dict
            class_name = BCSS_CLASSES.get(cls_id, f"Class {cls_id}")
            
            # Fetch the color assigned by the colormap for this exact ID
            color = cmap(cls_id / 9.0) # Normalizing index to map correctly to colormap bounds
            
            # Create a colored patch wrapper for the legend
            patch_handle = mpatches.Patch(color=color, label=f"[{cls_id}] {class_name}")
            legend_handles.append(patch_handle)
            
        # Place the discrete legend bounding box nicely right next to the mask axis
        axes[i, 1].legend(
            handles=legend_handles, 
            bbox_to_anchor=(1.05, 1), 
            loc='upper left', 
            borderaxespad=0.,
            fontsize=8
        )

    # Clean layout execution
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.6, hspace=0.4) # Increased width spacing to make room for legend text
    
    plt.show()