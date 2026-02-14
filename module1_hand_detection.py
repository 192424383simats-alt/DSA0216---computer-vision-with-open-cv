import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            h, w, c = frame.shape

            # Example: Index fingertip (landmark 8)
            index_tip = hand_landmarks.landmark[8]

            # Convert normalized coordinates to pixels
            x_pixel = int(index_tip.x * w)
            y_pixel = int(index_tip.y * h)

            print("Pixel Position:", x_pixel, y_pixel)

            # Draw circle at fingertip
            cv2.circle(frame, (x_pixel, y_pixel), 6, (0, 255, 0), -1)

            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    cv2.imshow("Module 1 - Pixel Extraction", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
