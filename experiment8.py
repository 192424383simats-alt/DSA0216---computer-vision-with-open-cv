
# Experiment 8
# Aim: Rotate an image clockwise and counterclockwise and display results.

import cv2
import matplotlib.pyplot as plt

def main():
    # Give the correct image path
    path = "Picture2.jpg"   # Change this path if needed

    img = cv2.imread(path)

    if img is None:
        print("Image not loaded. Check the path.")
        return

    clockwise = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    anticlockwise = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    clockwise = cv2.cvtColor(clockwise, cv2.COLOR_BGR2RGB)
    anticlockwise = cv2.cvtColor(anticlockwise, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12,4))

    plt.subplot(1,3,1)
    plt.imshow(img)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.imshow(clockwise)
    plt.title("Clockwise")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.imshow(anticlockwise)
    plt.title("Counter Clockwise")
    plt.axis("off")

    plt.show()

if __name__ == "__main__":
    main()
