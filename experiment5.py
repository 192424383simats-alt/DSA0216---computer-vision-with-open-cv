
# Experiment 5
# Aim: Perform Image Erosion and display Original and Eroded images.

import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Give the correct image path
    path = "Picture5.jpg"   # Change this path if needed

    img = cv2.imread(path)

    if img is None:
        print("Image not loaded. Check the path.")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kernel = np.ones((5,5), np.uint8)

    eroded = cv2.erode(gray, kernel, iterations=1)

    plt.figure(figsize=(8,4))

    plt.subplot(1,2,1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(eroded, cmap='gray')
    plt.title("Eroded Image")
    plt.axis("off")

    plt.show()

if __name__ == "__main__":
    main()
