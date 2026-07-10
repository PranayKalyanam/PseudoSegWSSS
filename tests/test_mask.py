import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load the mask as a grayscale image
mask = cv2.imread('C:\GitHub\PseudoSegWSSS\datasets\BCSS\masks\TCGA-A1-A0SK-DX1_xmin45749_ymin25055_MAG-10.00.png', cv2.IMREAD_GRAYSCALE)


# Quick trick: Multiply by a constant to stretch the values across the 0-255 spectrum
# Or use a color map to visualize the distinct classes
plt.imshow(mask, cmap='jet')
plt.colorbar()
plt.show()