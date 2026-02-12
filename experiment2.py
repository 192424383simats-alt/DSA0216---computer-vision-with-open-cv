
# Experiment 2
# Aim: Apply Gaussian Blur to an image and display Original and Blurred images.

import cv2
import matplotlib.pyplot as plt

def main():
    # Give the correct image path
    path = "Picture2.jpg"   # Change this path if needed

    image = cv2.imread(path)

    if image is None:
        print("Image not loaded. Check the path.")
    else:
        blur = cv2.GaussianBlur(image, (15, 15), 0)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        blur_rgb = cv2.cvtColor(blur, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(8,4))

        plt.subplot(1,2,1)
        plt.imshow(image_rgb)
        plt.title("Original Image")
        plt.axis("off")

        plt.subplot(1,2,2)
        plt.imshow(blur_rgb)
        plt.title("Gaussian Blurred Image")
        plt.axis("off")

        plt.show()

if __name__ == "__main__":
    main()
