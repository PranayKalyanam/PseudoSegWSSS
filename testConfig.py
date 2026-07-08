#Module 1 test config.py
# ============================
from configs.config import get_config

args = get_config()

print("Configuration:")
for arg in vars(args):
    print(f"{arg}: {getattr(args, arg)}")
    


#Module 2 test logger.py
# ===========================
# from utils.logger import get_logger
# from configs.config import get_config

# args = get_config()

# logger = get_logger(
#     name="SlideLoader",
#     log_dir=args.output_dir + "/logs",
#     level=args.log_level
# )

# logger.info("Opening WSI...")
# logger.warning("Mask file missing.")
# logger.error("Unable to read slide.")


#Module 2 test exceptions.py
# ===========================

# from pathlib import Path
# from PIL import Image
# from utils.exceptions import SlideOpenError

# slide_path = Path("datasets/BCSS/imgs/TCGA-A2-A0CM-DX1_xmin18562_ymin56852_MPP-0.2500.png")
# mask_path = Path("datasets/BCSS/masks/TCGA-A1-A0SP-DX1_xmin6798_ymin53719_MPP-0.2500.png")


# try:
#     # Use PIL for standard image formats like .png
#     slide = Image.open(slide_path)
#     mask = Image.open(mask_path)
#     print(f"Successfully opened image patch: {slide_path}")
#     print(slide.format, slide.size, slide.mode) 
    
#     print(f"Successfully opened mask patch: {mask_path}")
#     print(mask.format, mask.size, mask.mode) # L means an 8-bit grayscale image (luminance), not RGB.
# except Exception as exc:
#     raise SlideOpenError(f"Failed to open image patch: {slide_path}") from exc


#Module 3 test validator.py
# from utils.validator import Validator
# from pathlib import Path
# from PIL import Image
# from configs.config import get_config

# args = get_config()

# slide_path = Path("datasets/BCSS/imgs/TCGA-A2-A0CM-DX1_xmin18562_ymin56852_MPP-0.2500.png")
# mask_path = Path("datasets/BCSS/masks/TCGA-A1-A0SP-DX1_xmin6798_ymin53719_MPP-0.2500.png")


# Validator.validate_wsi_path(slide_path)
# Validator.validate_mask_path(mask_path)
# Validator.validate_magnification(args.magnification)