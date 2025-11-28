import pickle
import cv2
import mediapipe as mp
import numpy as np

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,          
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

labels_dict = {0: 'L', 1: 'V', 2: 'W'}

while True:

    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        continue

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )

        for l in hand_landmarks.landmark:
            x_.append(l.x)
            y_.append(l.y)

        for l in hand_landmarks.landmark:
            data_aux.append(l.x - min(x_))
            data_aux.append(l.y - min(y_))

        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict[int(prediction[0])]

        x1, y1 = int(min(x_) * W) - 10, int(min(y_) * H) - 10 # 10 margin for better visualization
        x2, y2 = int(max(x_) * W) + 10, int(max(y_) * H) + 10

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3) #font style , font size , color , thickness

        print("Predicted:", predicted_character)

    cv2.imshow('frame', frame) # Window name
    if cv2.waitKey(1) == 27: # Esc key to quite
        break

cap.release()
cv2.destroyAllWindows()
