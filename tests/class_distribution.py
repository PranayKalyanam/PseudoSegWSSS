from collections import Counter
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

from utils.logger import get_logger

# BCSS Dataset categories
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
logger = get_logger(
    name="ClassDistribution",
    log_dir="logs",
    level="INFO",
)
def main():
    patches_root = Path(r"C:\GitHub\PseudoSegWSSS\outputs\patches")
    
    if not patches_root.exists():
        # print(f"❌ Error: Root path does not exist: {patches_root}")
        logger.error(
            f"Root path does not exist: {patches_root}"
        )
        return

    # Master pixel counter across all masks

    class_patch_counts = Counter()
    total_patches_processed = 0
    total_pixel_counts = Counter()

    # Find all patient directories
    patient_folders = [p for p in patches_root.iterdir() if p.is_dir()]
    # print(f"Scanning masks across {len(patient_folders)} patient folder(s)...")

    logger.info(
        f"Scanning masks across {len(patient_folders)} patient folder(s)..."
    )

    # ADDED: Wrapped the patient loop in tqdm to display the progress bar
    for patient_dir in tqdm(patient_folders, desc="Processing Patients", unit="patient"):
        mask_dir = patient_dir / "mask"
        if not mask_dir.exists():
            continue
            
        mask_files = list(mask_dir.glob("*.png"))
        
        for mask_path in mask_files:
            # Read the 8-bit mask
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue

            total_patches_processed += 1

            # Find the unique class IDs present in this single patch
            unique_classes = np.unique(mask)

            # Increment the patch count by 1 for every class present in this file
            for class_id in unique_classes:
                class_patch_counts[class_id] += 1
                
            # Count the occurrences of each unique integer in this specific mask
            unique, counts = np.unique(mask, return_counts=True)
            total_pixel_counts.update(dict(zip(unique, counts)))

    # Calculate total aggregated pixels across the whole dataset
    overall_total_pixels = sum(total_pixel_counts.values())

    if overall_total_pixels == 0:
        # print("❌ No pixel data found inside masks.")
        logger.warning(
            f"❌ No pixel data found inside masks."
        )
        return
    
    if total_patches_processed == 0:
        # print("❌ No patch data found inside masks.")
        logger.warning(
            f"❌ No patch data found inside masks."
        )
        return

    # Print the Results Table
    separator = "=" * 120

    header = (
        f"{'Class ID':<10} | "
        f"{'Category Name':<30} | "
        f"{'Patch Count':<15} | "
        f"{'% of Total Patches':<20} | "
        f"{'Pixel Count':<15} | "
        f"{'Percentage'}"
    )
    # print("\n" + "=" * 65)
    logger.info(separator)
    # print(f"{'Class ID':<10} | {'Category Name':<30} | {'Patch Count':<15} | {'% of Total Patches'} | {'Pixel Count':<15} | {'Percentage'}")
    logger.info(header)
    # print("=" * 65)
    logger.info(separator)

    # Sort by Class ID numerically
    for class_id in sorted(BCSS_CLASSES.keys()):
        pixel_count = total_pixel_counts.get(class_id, 0)
        pixel_percentage = (pixel_count / overall_total_pixels) * 100
        patch_count = class_patch_counts.get(class_id, 0)
        # Percentage of total patches that contain this class
        patch_percentage = (patch_count / total_patches_processed) * 100
        class_name = BCSS_CLASSES[class_id]

        row = (
            f"{class_id:<10} | "
            f"{class_name:<30} | "
            f"{patch_count:<15,} | "
            f"{patch_percentage:<20.2f} | "
            f"{pixel_count:<15,} | "
            f"{pixel_percentage:.2f}%"
        )
        
        # print(f"{class_id:<10} | {class_name:<30} | {patch_count:<15,} | {patch_percentage:.2f} | {pixel_count:<15,} | {pixel_percentage:.2f}%")
        logger.info(row)

    # print("=" * 65)
    logger.info(separator)
    # print(f"Total Unique Patches Evaluated: {total_patches_processed:,}")
    logger.info(
        f"Total Unique Patches Evaluated: "
        f"{total_patches_processed:,}"
    )

    # print(f"Total Aggregated Pixels: {overall_total_pixels:,}\n")
    logger.info(
        f"Total Aggregated Pixels: "
        f"{overall_total_pixels:,}"
    )
    
    # print("Note: Percentages won't add up to 100% because one patch can contain multiple classes.\n")
    logger.info(
        "Note: Percentages will not add up to 100% because "
        "one patch can contain multiple semantic classes."
    )


if __name__ == "__main__":
    main()