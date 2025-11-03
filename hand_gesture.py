import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Webcam input
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6) as hands:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Convert to RGB for Mediapipe
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # Convert back to BGR for OpenCV
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):

                # === Gesture Classifier ===
                def gesture_classifier(hand_landmarks):
                    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
                    thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]

                    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
                    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

                    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
                    middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
                    ring_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
                    pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]

                    # Which fingers are extended
                    index_extended = index_tip.y < index_mcp.y
                    middle_extended = middle_tip.y < middle_mcp.y
                    ring_extended = ring_tip.y < ring_mcp.y
                    pinky_extended = pinky_tip.y < pinky_mcp.y

                    # Thumb direction
                    thumb_up = thumb_tip.y < thumb_ip.y < thumb_mcp.y
                    thumb_down = thumb_tip.y > thumb_ip.y > thumb_mcp.y

                    # === Gesture Rules ===
                    if thumb_up and not (index_extended or middle_extended or ring_extended or pinky_extended):
                        return "Thumbs Up"
                    elif thumb_down and not (index_extended or middle_extended or ring_extended or pinky_extended):
                        return "Thumbs Down"
                    elif all([thumb_up, index_extended, middle_extended, ring_extended, pinky_extended]):
                        return "Hello Buddy"
                    elif index_extended and not (middle_extended or ring_extended or pinky_extended):
                        return "One"
                    elif index_extended and middle_extended and not (ring_extended or pinky_extended):
                        return "Two"
                    elif index_extended and middle_extended and ring_extended and not pinky_extended:
                        return "Three"
                    elif index_extended and pinky_extended and not (middle_extended or ring_extended):
                        return "Cool"
                    else:
                        return "No Recognized Gesture"

                # Detect gesture
                gesture = gesture_classifier(hand_landmarks)

                # ✏️ Custom thin landmark lines
                landmark_style = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=2)
                connection_style = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1)

                # Draw thinner landmarks
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    landmark_style,
                    connection_style
                )

                # 🟥 Display red thin text
                cv2.putText(image, f'{gesture}', (10, 50 + (idx * 40)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)

        # Show the image
        cv2.imshow('Hand Gesture Recognition', image)

        # Exit on ESC
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
