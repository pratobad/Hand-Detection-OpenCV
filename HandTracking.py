from ast import And
import enum
import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

cTime=0
pTime=0

while True:
    sucess,img = cap.read()
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    
    if results.multi_hand_landmarks:
        for HandLMS in results.multi_hand_landmarks:
            for ID,lm in enumerate(HandLMS.landmark):
                h,w,c = img.shape
                cx,cy = int(lm.x*w) , int(lm.y*h)
                print(ID,lm)
            mpDraw.draw_landmarks(img,HandLMS,mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255,255,0),1)

    cv2.imshow("Image",img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break