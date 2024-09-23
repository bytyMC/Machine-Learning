import pandas as pd
import cv2
import dlib 
import numpy as np

img = cv2.imread('data/age_data/minion.jpg') 
img = cv2.resize(img, (720, 640)) 
frame = img.copy()

# Load pre-trained model
age_weights = "data/age_data/age_deploy.prototxt"
age_config = "data/age_data/age_net.caffemodel"
age_Net = cv2.dnn.readNet(age_config, age_weights) 

# Classes
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', 
        '(25-32)', '(38-43)', '(48-53)', '(60-100)'] 
# Precalculated mean values
model_mean = (78.4263377603, 87.7689143744, 114.895847746) 

frame_height = img.shape[0] 
frame_width = img.shape[1]

face_coord = []
msg = 'Face Detected' # to display on image 

face_detector = dlib.get_frontal_face_detector() 
# converting to grayscale 
img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

faces = face_detector(img_gray) 

if not faces: 
    mssg = 'No face detected'
    cv2.putText(img, f'{mssg}', (40, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (200), 2) 
    # Visualize
    cv2.imshow("Image",img) 
    cv2.waitKey(0) 

else: 
    for face in faces: 
        x1 = face.left()
        y1 = face.top() 
        x2 = face.right() 
        y2 = face.bottom() 

        # rescaling those coordinates for our image 
        box = [x1, y1, x2, y2] 
        face_coord.append(box) 
        cv2.rectangle(frame, (x1, y1), (x2, y2), 
                    (00, 200, 200), 2) 

    for box in face_coord: 
        face = frame[box[1]:box[3], box[0]:box[2]] 

        blob = cv2.dnn.blobFromImage( 
            face, 1.0, (227, 227), model_mean, swapRB=False) 

        age_Net.setInput(blob) 
        age_preds = age_Net.forward() 
        age = ageList[age_preds[0].argmax()] 

        cv2.putText(frame, f'{msg}:{age}', (box[0], 
                                            box[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, 
                    (0, 255, 255), 2, cv2.LINE_AA) 
        # Visualize
        cv2.imshow("Image",frame) 
        cv2.waitKey(0)