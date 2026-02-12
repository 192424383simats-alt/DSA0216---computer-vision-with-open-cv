
# Experiment 9
# Aim: Perform Image Translation (Moving Image) using Affine Transformation.

import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Give the correct image path
    path = "Picture4.jpg"   # Change this path if needed

    img = cv2.imread(path)

    if img is None:
        print("Image not loaded. Check the path.")
        return

    rows, cols = img.shape[:2]

    M = np.float32([[1, 0, 100],
                    [0, 1, 50]])

    moved = cv2.warpAffine(img, M, (cols, rows))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    moved = cv2.cvtColor(moved, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(8,4))

    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(moved)
    plt.title("Moved Image")
    plt.axis("off")

    plt.show()

if __name__ == "__main__":
    main()
