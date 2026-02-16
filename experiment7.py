
# Experiment 7
# Aim: Scale an image to Bigger and Smaller sizes and display dimensions.

import cv2
import matplotlib.pyplot as plt

def main():
    # Give the correct image path
    path = "Picture4.jpg"   # Change this path if needed

    img = cv2.imread(path)

    if img is None:
        print("Image not loaded. Check the path.")
        return

    bigger = cv2.resize(img, None, fx=2, fy=2)
    smaller = cv2.resize(img, None, fx=0.5, fy=0.5)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    bigger = cv2.cvtColor(bigger, cv2.COLOR_BGR2RGB)
    smaller = cv2.cvtColor(smaller, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(4,3))
    plt.imshow(img)
    plt.title(f"Original {img.shape[1]}x{img.shape[0]}")
    plt.axis("off")
    plt.show()

    plt.figure(figsize=(8,6))
    plt.imshow(bigger)
    plt.title(f"Bigger {bigger.shape[1]}x{bigger.shape[0]}")
    plt.axis("off")
    plt.show()

    plt.figure(figsize=(2,1.5))
    plt.imshow(smaller)
    plt.title(f"Smaller {smaller.shape[1]}x{smaller.shape[0]}")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    main()
