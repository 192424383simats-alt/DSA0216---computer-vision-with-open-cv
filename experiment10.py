
# Experiment 10
# Aim: Perform Affine Transformation on an image and display results.

import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Give the correct image path
    path = "Picture3.jpg"   # Change this path if needed

    img = cv2.imread(path)

    if img is None:
        print("Image not loaded. Check the path.")
        return

    rows, cols = img.shape[:2]

    pts1 = np.float32([[0,0], [cols-1,0], [0,rows-1]])
    pts2 = np.float32([[50,50], [cols-50,30], [40,rows-40]])

    M = cv2.getAffineTransform(pts1, pts2)

    affine = cv2.warpAffine(img, M, (cols, rows))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    affine = cv2.cvtColor(affine, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(8,4))

    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(affine)
    plt.title("Affine Transformed Image")
    plt.axis("off")

    plt.show()

if __name__ == "__main__":
    main()
