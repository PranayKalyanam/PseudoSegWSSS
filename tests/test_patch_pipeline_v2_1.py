import os
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

from utils.logger import get_logger

# 1. Define the official/distinct RGB palette for BCSS classes
# Format: {class_id: (B, G, R)} -> OpenCV uses BGR format for saving images
BCSS_COLOR_PALETTE = {
    0: (0, 0, 0),       # Outside / Background -> Black
    1: (0, 0, 255),     # Tumor (Epithelium) -> Red
    2: (0, 255, 0),     # Stroma (Lamina Propria) -> Green
    3: (255, 255, 0),   # Inflammatory Infiltrate -> Blue
    4: (0, 255, 255),   # Necrosis / Debris -> Yellow
    5: (0, 0, 0),   # Glandular / Secretions -> Magenta
    6: (0, 0, 0),   # Blood / Vessel -> Cyan
    7: (0, 0, 0),   # Metaplasia -> Purple
}

logger = get_logger(
    name="MaskColorConverter",
    log_dir="logs",
    level="INFO",
)

def convert_8bit_to_color(mask_8bit, palette):
    """Converts a 1-channel 8-bit integer mask to a 3-channel BGR color mask."""
    h, w = mask_8bit.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Map each class integer to its defined BGR color array
    for class_id, color in palette.items():
        color_mask[mask_8bit == class_id] = color
        
    return color_mask

def main():
    # 2. Define the root output directory
    patches_root = Path(r"C:\GitHub\PseudoSegWSSS\outputs\patches")
    
    if not patches_root.exists():
        # print(f"❌ Error: Root path does not exist: {patches_root}")
        logger.warning(
                    f"❌ Error: Root path does not exist: {patches_root}"
                )
        return

    # 3. Find all patient subdirectories inside the patches folder
    patient_folders = [p for p in patches_root.iterdir() if p.is_dir()]

    # print(f"Found {len(patient_folders)} patient folder(s) to process.")
    logger.info(
        f"Found {len(patient_folders)} patient folder(s) to process."
    )

    for patient_dir in patient_folders:
        mask_dir = patient_dir / "mask"
        color_mask_dir = patient_dir / "mask_color"
        
        # Skip if this directory doesn't have a 'mask' folder inside it
        if not mask_dir.exists():
            logger.warning(
                    f"Skipping {patient_dir.name}: mask directory not found."
                )
            continue
            
        # print(f"\nProcessing patient: {patient_dir.name}")
        logger.info(
            f"Processing patient: {patient_dir.name}"
        )
        
        
        # Create the new 'mask_color/' folder if it doesn't exist
        color_mask_dir.mkdir(parents=True, exist_ok=True)
        
        # Gather all 8-bit masks
        mask_files = list(mask_dir.glob("*.png"))
        
        # Process and save each mask
        for mask_path in tqdm(mask_files, desc="Converting masks"):
            # Read the 8-bit mask in grayscale mode
            mask_8bit = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            
            if mask_8bit is None:
                logger.warning(
                        f"Could not read mask: {mask_path}"
                    )
                continue
                
            # Convert to RGB visual representation
            color_mask = convert_8bit_to_color(mask_8bit, BCSS_COLOR_PALETTE)
            
            # Save the colored mask with the exact same name into 'mask_color/'
            output_path = color_mask_dir / mask_path.name
            cv2.imwrite(str(output_path), color_mask)

    # print("\n✅ Optimization Complete! All 8-bit masks have been colorized and saved.")
    logger.info(
        "✅ Optimization Complete! All 8-bit masks have been colorized and saved."
    )

if __name__ == "__main__":
    main()