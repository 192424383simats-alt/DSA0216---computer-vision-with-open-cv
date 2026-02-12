
# Experiment 4
# Aim: Perform Image Dilation and display Original and Dilated images.

import cv2
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Give the correct image path
    path = "Picture4.jpg"   # Change this path if needed

    image = cv2.imread(path)

    if image is None:
        print("Image not loaded. Check the path.")
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        kernel = np.ones((5,5), np.uint8)

        dilated = cv2.dilate(gray, kernel, iterations=1)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(8,4))

        plt.subplot(1,2,1)
        plt.imshow(image_rgb)
        plt.title("Original Image")
        plt.axis("off")

        plt.subplot(1,2,2)
        plt.imshow(dilated, cmap='gray')
        plt.title("Dilated Image")
        plt.axis("off")

        plt.show()

if __name__ == "__main__":
    main()
