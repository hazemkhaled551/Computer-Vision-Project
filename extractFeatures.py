import os
import pickle
import cv2
import shutil
import mediapipe as mp

DATA_DIR = "data"
INVALID_DIR = "invalid"

if not os.path.exists(INVALID_DIR):
    os.makedirs(INVALID_DIR)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

data = []
labels = []

with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
) as hands:

    for label in os.listdir(DATA_DIR):
        label_path = os.path.join(DATA_DIR, label)

        if not os.path.isdir(label_path):
            continue

        for img_name in os.listdir(label_path):

            img_path = os.path.join(label_path, img_name)
            img = cv2.imread(img_path)

            
            if img is None:
                print("Invalid image (None):", img_path)
                shutil.move(img_path, os.path.join(INVALID_DIR, img_name))
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

        
            if not results.multi_hand_landmarks:
                print("No landmarks detected:", img_path)
                shutil.move(img_path, os.path.join(INVALID_DIR, img_name))
                continue

        
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = []

                h, w, _ = img.shape

                for lm in hand_landmarks.landmark:
                    x = lm.x * w
                    y = lm.y * h
                    landmarks.append(x)
                    landmarks.append(y)

                data.append(landmarks)
                labels.append(int(label))


with open("data.pickle", "wb") as f:
    pickle.dump((data, labels), f)

print("saved to data.pickle")
print("length of data:", len(data))
