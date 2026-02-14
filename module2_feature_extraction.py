import cv2
import mediapipe as mp
import numpy as np
import csv
import os

# -----------------------------
# Initialize MediaPipe
# -----------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands()

cap = cv2.VideoCapture(0)

prev_x, prev_y = 0, 0
distances = []

fps = 30

# -----------------------------
# Ask for Label (for dataset)
# -----------------------------
label = int(input("Enter Label (0 = Normal, 1 = Tremor): "))

file_path = "tremor_data.csv"
file_exists = os.path.isfile(file_path)

csv_file = open(file_path, mode="a", newline="")
csv_writer = csv.writer(csv_file)

# Write header only if file is new
if not file_exists:
    csv_writer.writerow(["Amplitude", "Frequency", "Label"])

print("Recording... Show hand to camera.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            tip = hand_landmarks.landmark[8]
            x = int(tip.x * w)
            y = int(tip.y * h)

            if prev_x != 0:
                distance = np.sqrt((x-prev_x)**2 + (y-prev_y)**2)
                distances.append(distance)

            prev_x, prev_y = x, y

            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    # When enough motion samples collected (~3 sec)
    if len(distances) >= 90:

        signal = np.array(distances)

        # Remove DC component
        signal = signal - np.mean(signal)

        # Remove large voluntary spikes
        threshold = np.mean(signal) + 2*np.std(signal)
        signal = np.where(abs(signal) > threshold, 0, signal)

        # FFT
        fft = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal), 1/fps)

        positive_freqs = freqs[:len(freqs)//2]
        positive_fft = np.abs(fft[:len(fft)//2])

        tremor_band = (positive_freqs >= 2) & (positive_freqs <= 8)

        if np.any(tremor_band):
            dominant_freq = positive_freqs[tremor_band][
                np.argmax(positive_fft[tremor_band])
            ]
        else:
            dominant_freq = 0

        amplitude = np.max(np.abs(signal))

        # Display Features Only
        cv2.putText(frame, f"Amplitude: {round(amplitude,2)} px",
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.putText(frame, f"Frequency: {round(dominant_freq,2)} Hz",
                    (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        # Save dataset
        csv_writer.writerow([
            round(amplitude,2),
            round(dominant_freq,2),
            label
        ])
        csv_file.flush()

        print("Saved → Amplitude:",
              round(amplitude,2),
              "Frequency:",
              round(dominant_freq,2))

        distances.clear()

    cv2.imshow("Module 2 - Feature Extraction Only", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
csv_file.close()
cv2.destroyAllWindows()
