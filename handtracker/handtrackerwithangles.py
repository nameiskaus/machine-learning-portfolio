import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def calculate_angle(a, b, c):
    a = np.array(a)  
    b = np.array(b) 
    c = np.array(c)  
    
    ba = a - b  
    bc = c - b  
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))  
    return np.degrees(angle)


cap = cv2.VideoCapture(0)
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            finger_joints = {
                "Index": [5, 6, 7, 8],
                "Middle": [9, 10, 11, 12],
                "Ring": [13, 14, 15, 16],
                "Pinky": [17, 18, 19, 20]
            }

            for finger, joints in finger_joints.items():
                for i in range(1, 3):  
                    p1 = hand_landmarks.landmark[joints[i - 1]]
                    p2 = hand_landmarks.landmark[joints[i]]
                    p3 = hand_landmarks.landmark[joints[i + 1]]

                    p1 = (p1.x * image.shape[1], p1.y * image.shape[0])
                    p2 = (p2.x * image.shape[1], p2.y * image.shape[0])
                    p3 = (p3.x * image.shape[1], p3.y * image.shape[0])

                    
                    angle = calculate_angle(p1, p2, p3)

                    
                    cv2.putText(image, f"{int(angle)}°", (int(p2[0]), int(p2[1])),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Hand Tracking with Angles', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
