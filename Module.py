from ast import And
import enum
import cv2
import mediapipe as mp
import time


def main():
   cTime=0
   pTime=0
   cap = cv2.VideoCapture(0)
   detector = HandDetector()
   while True:
       sucess,img = cap.read()
       detector.find_hands(img)
       cTime = time.time()
       fps = 1/(cTime-pTime)
       pTime = cTime
       cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255,255,0),1) 
       cv2.imshow("Image",img)
       key = cv2.waitKey(1) & 0xFF
       if key == ord('q'):
           break  



if __name__ == "__main__":
    main()


class HandDetector():
    def __init__(self,mode = False,max_hands=2,detect_confi = 0.5,track_confi=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detect_confi = detect_confi
        self.track_confi = track_confi

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,self.max_hands,self.detect_confi,self.track_confi)
        self.mpDraw = mp.solutions.drawing_utils


    def find_hands(self,img,draw = True):
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)
    
        if results.multi_hand_landmarks:
            for HandLMS in results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img,HandLMS,self.mpHands.HAND_CONNECTIONS)


        