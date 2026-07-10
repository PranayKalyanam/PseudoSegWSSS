from collections import Counter
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

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

def main():
    patches_root = Path(r"C:\GitHub\PseudoSegWSSS\outputs\patches")
    
    if not patches_root.exists():
        print(f"❌ Error: Root path does not exist: {patches_root}")
        return

    # Master pixel counter across all masks

    class_patch_counts = Counter()
    total_patches_processed = 0
    total_pixel_counts = Counter()

    # Find all patient directories
    patient_folders = [p for p in patches_root.iterdir() if p.is_dir()]
    print(f"Scanning masks across {len(patient_folders)} patient folder(s)...")

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
        print("❌ No pixel data found inside masks.")
        return
    
    if total_patches_processed == 0:
        print("❌ No patch data found inside masks.")
        return

    # Print the Results Table
    print("\n" + "=" * 65)
    print(f"{'Class ID':<10} | {'Category Name':<30} | {'Patch Count':<15} | {'% of Total Patches'} | {'Pixel Count':<15} | {'Percentage'}")
    print("=" * 65)

    # Sort by Class ID numerically
    for class_id in sorted(BCSS_CLASSES.keys()):
        pixel_count = total_pixel_counts.get(class_id, 0)
        pixel_percentage = (pixel_count / overall_total_pixels) * 100
        patch_count = class_patch_counts.get(class_id, 0)
        # Percentage of total patches that contain this class
        patch_percentage = (patch_count / total_patches_processed) * 100
        class_name = BCSS_CLASSES[class_id]
        
        print(f"{class_id:<10} | {class_name:<30} | {patch_count:<15,} | {patch_percentage:.2f} | {pixel_count:<15,} | {pixel_percentage:.2f}%")

    print("=" * 65)
    print(f"Total Unique Patches Evaluated: {total_patches_processed:,}")

    print(f"Total Aggregated Pixels: {overall_total_pixels:,}\n")
    
    print("Note: Percentages won't add up to 100% because one patch can contain multiple classes.\n")


if __name__ == "__main__":
    main()