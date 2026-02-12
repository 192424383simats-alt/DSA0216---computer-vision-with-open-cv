
# Experiment 1
# Aim: Read an image, convert it to grayscale, and display it.

import cv2
import matplotlib.pyplot as plt

def main():
    # Give the correct image path
    path = "Picture1.jpg"   # Change this path if needed

    image = cv2.imread(path)

    if image is None:
        print("Image not found")
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        plt.imshow(gray, cmap='gray')
        plt.title("Grayscale Image")
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    main()
