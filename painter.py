import mediapipe as mp
import cv2 as cv
import numpy as np
from numpy.linalg import norm

def convertPixels(landmark, height, width):
    px = int(landmark.x*w)
    py = int(landmark.y*h)
    return px, py

def findNumFinger(landmarkList):
    finger = 0
    if norm(landmarkList[8][2]-landmarkList[0][2])>120:
        finger = 1
        if norm(landmarkList[12][2]-landmarkList[0][2])>120:
            finger = 2
            if norm(landmarkList[16][2]-landmarkList[0][2])>120:
                finger = 3
                if norm(landmarkList[20][2]-landmarkList[0][2])>120:
                    finger = 4
                    if norm(landmarkList[4][1]-landmarkList[0][1])>50:
                        finger = 5
    return finger

def showFingers(fingerNum, img):
    cv.putText(img, "Finger = " + str(fingerNum), (0, 100), cv.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2, cv.LINE_AA)
    if fingerNum == 1:
        cv.putText(img, "Color = Red", (0, 150), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 2, cv.LINE_AA)
    elif fingerNum == 2:
        cv.putText(img, "Color = Green" , (0, 150), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2, cv.LINE_AA)
    elif fingerNum == 3:
        cv.putText(img, "Color = Blue", (0, 150), cv.FONT_HERSHEY_SIMPLEX, 1.5, (255,0,0), 2, cv.LINE_AA)
    return img

def drawCircles(img, drawList):
    for point in drawList:
        if point[2] == 1:
            cv.circle(img, (point[0],point[1]), 20, (0,0,255), -1)
        elif point[2] == 2:
            cv.circle(img, (point[0],point[1]), 20, (0,255,0), -1)
        elif point[2] == 3:
            cv.circle(img, (point[0],point[1]), 20, (255,0,0), -1)
    return img

mpDrawing = mp.solutions.drawing_utils
mpFace = mp.solutions.face_detection

mpHands = mp.solutions.hands

cap = cv.VideoCapture(0)
_, img = cap.read()
h,w,_ = img.shape

drawList = []


with mpHands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.4,
    min_tracking_confidence=0.5) as hands:

    with mpFace.FaceDetection(
        min_detection_confidence=0.5) as face_detection:
        while(True):
            sucess, img = cap.read()
            if not sucess:
                print("Error opening webcam")
                continue

            landmarkList = []
            results = hands.process(img)

            resultsFace = face_detection.process(img)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for id, landmark in enumerate(hand_landmarks.landmark):
                        px, py = convertPixels(landmark, h, w)
                        auxList = id, px, py
                        landmarkList.append(auxList)

                    fingerNum = findNumFinger(landmarkList)
                    img = showFingers(fingerNum, img)

                    if fingerNum == 1 or fingerNum == 2 or fingerNum == 3:
                        drawX, drawY = landmarkList[8][1], landmarkList[8][2]
                        auxList = drawX, drawY, fingerNum
                        drawList.append(auxList)
                    mpDrawing.draw_landmarks(img, hand_landmarks)
            img = drawCircles(img, drawList)

            if resultsFace.detections:
                for detection in resultsFace.detections:
                    mpDrawing.draw_detection(img, detection)
            cv.imshow('MediaPipe Hands', img)
                
                
            if cv.waitKey(5) & 0xFF == 27:
                break


