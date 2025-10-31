## Project Idea

**Description:** This project aims to develop a Real-Time Sign Language Translator using computer vision and machine learning.

**Objectives:**

- Build a real-time system to recognize ASL gestures using the webcam.

- Collect and preprocess hand landmarks data using MediaPipe.

- Train a classification model to identify alphabet signs.

- Integrate the trained model into a live video stream for instant translation.

- Evaluate model accuracy and optimize for better real-world performance.

**Scope:** Real-time recognition of ASL alphabet letters using webcam input.

Out of Scope: Full word or sentence recognition, voice synthesis, and multilingual support (can be added later).

## Team Members and Roles

| Team Member     | GitHub Account   | Role   | Responsibilities   |
| --------------- | ---------------- | ------ | ------------------ |
| Hazem Khaled    | hazemkhaled551   |   _    | Convert each image into a set of points representing the hand shape.|
| Eslam Elhosuiny | eslamelhosuniy   |   _    | Build a model capable of recognizing different hand signs. |
| Farha Ashraf    | farha-158        |   _    | Create a database containing images of the hand performing different sign gestures. |

## Tools and Usage



| Tool/Library | Purpose                             | Usage Details   |
| ------------ | ------------------------------------| --------------- |
| OpenCV (cv2) | Camera control & visualization      | Used to open webcam, capture frames, and display predictions in real-time |
| MediaPipe    | Hand tracking & landmark extraction | Detects hand and generates 21 3D landmarks for gesture recognition |
| NumPy        |Data manipulation                    | Converts hand landmark coordinates into numeric arrays for model input |
| Scikit-learn | Model training & prediction         | Used to train and test gesture classification models |
| Pickle       | Model persistence                   | Save and load trained models for reuse without retraining |

**Hardware/Environment:** [Note any special requirements.]

## 4-Week Plan

Break down the project into a 4-week timeline with milestones, deliverables, and assigned team members.

### Week 1: Data Collection Phase

- **Milestones:** 

    - Create a database containing images of the hand performing different sign gestures.
    - Run the camera using OpenCV.
    - Define the letters to be trained (A, B, L).
- **Deliverables:** 

    - 100 images captured for each letter.
    - Images organized into folders: data/0, data/1, data/2.
    - Images saved upon pressing the (Q) key.
- **Assigned:** Farha Ashraf


### Week 2: Feature Extraction Phase

- **Milestones:** 

    - Convert each image into a set of points representing the hand shape.
    - Use MediaPipe to detect the hand in each image.
    - Extract 21 landmarks for each hand.
- **Deliverables:** 

    - Store only (x, y) coordinate values.
    - Save the processed data into data.pickle as lists (data, labels).
- **Assigned:** Hazem Khaled


### Week 3: Model Training Phase

- **Milestones:** 

    - Build a model capable of recognizing different hand signs.
    - Load data from data.pickle.
    - Train a RandomForestClassifier model using scikit-learn.
- **Deliverables:** 

    - Trained model evaluated using accuracy_score.
    - Save the trained model as model.p using pickle.
- **Assigned:** Eslam Elhosuiny


### Week 4: Real-Time Testing Phase

- **Milestones:** 

    - Test the trained model using the webcam in real-time.
    - Detect hand landmarks from the video stream.
    - Pass the coordinates to the trained model for prediction.
- **Deliverables:** 

    - Display predicted letter (A, B, L) on the screen.
    - Show a rectangle around the detected hand.
- **Assigned:** Hazem Khaled, Eslam Elhosuiny, Farha Ashraf

**Overall Timeline Notes:** [Any additional notes.]

### Checklist for detailed tasks

- [ ] Task 1
- [ ] Task 2
- [ ] Task 3
- [ ] Task 4

## Evaluation Criteria

- **Success Metrics:** Smooth real-time camera performance
- **Next Steps:** Expand from single letters to full words.

Submit this proposal for approval before starting. Good luck!
