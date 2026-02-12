
# Experiment 3
# Aim: Perform Canny Edge Detection and display Original and Edge-detected images.

import cv2
import matplotlib.pyplot as plt

def main():
    # Give the correct image path
    path = "Picture3.jpg"   # Change this path if needed

    image = cv2.imread(path)

    if image is None:
        print("Image not loaded. Check the path.")
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(gray, 100, 200)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(8,4))

        plt.subplot(1,2,1)
        plt.imshow(image_rgb)
        plt.title("Original Image")
        plt.axis("off")

        plt.subplot(1,2,2)
        plt.imshow(edges, cmap='gray')
        plt.title("Canny Edge Detection (Outline)")
        plt.axis("off")

        plt.show()

if __name__ == "__main__":
    main()
