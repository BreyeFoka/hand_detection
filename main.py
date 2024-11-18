import cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8)
mp_draw = mp.solutions.drawing_utils


# Set up canvas and initial parameters
canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]  # Blue, Green, Red, Yellow
color_index = 0  # Start with the first color
drawing = False  # Drawing state
x1, y1 = 0, 0    # Previous coordinates

# Count raised fingers based on hand landmarks
def count_fingers(hand_landmarks):
    # Count fingers raised using landmark positions
    fingers = []
    tips = [8, 12, 16, 20]  # Index, middle, ring, and pinky finger tips
    for tip in tips:
        if hand_landmarks[tip].y < hand_landmarks[tip - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers.count(1)  # Number of fingers raised

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Convert to RGB for Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the number of fingers raised
            finger_count = count_fingers(hand_landmarks.landmark)

            # Gesture-based control
            if finger_count == 1:  # One finger - draw mode
                drawing = True
                x, y = int(hand_landmarks.landmark[8].x * w), int(hand_landmarks.landmark[8].y * h)
                if x1 == 0 and y1 == 0:
                    x1, y1 = x, y
                cv2.line(canvas, (x1, y1), (x, y), colors[color_index], 5)
                x1, y1 = x, y

            elif finger_count == 2:  # Two fingers - erase mode
                drawing = False
                x1, y1 = 0, 0
                x, y = int(hand_landmarks.landmark[8].x * w), int(hand_landmarks.landmark[8].y * h)
                cv2.circle(canvas, (x, y), 20, (0, 0, 0), -1)

            elif finger_count == 3:  # Three fingers - switch color
                drawing = False
                color_index = (color_index + 1) % len(colors)
                x1, y1 = 0, 0

            elif finger_count == 5:  # Five fingers - clear canvas
                canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
                x1, y1 = 0, 0
                drawing = False

    # Overlay the canvas on the frame
    frame = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

    # Display the selected color in the corner
    cv2.rectangle(frame, (10, 10), (110, 110), colors[color_index], -1)
    cv2.putText(frame, "Color", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Show the frame
    cv2.imshow("Air Canvas", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()