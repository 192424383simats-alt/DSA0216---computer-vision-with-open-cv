
# Experiment 6
# Aim: Read and display video file frame by frame (for Google Colab).

from google.colab.patches import cv2_imshow
import cv2

def main():
    # Give the correct video path
    path = "sample_640x360.mp4"   # Change this path if needed

    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        print("Error opening video file")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        cv2_imshow(frame)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
