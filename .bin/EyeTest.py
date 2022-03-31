import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
# cap = cv2.VideoCapture('Video.mp4')
cap = cv2.VideoCapture(0)
detector = FaceMeshDetector(maxFaces=1)
idList = [22,23,24,26,110,157,158,159,160,161,130,243]
while True:
    success, img = cap.read()
    img, faces = detector.findFaceMesh(img,draw=False)
    if faces:
        face = faces[0]
        for id in idList:
            cv2.circle(img,face[id],2,(255,0,255),2)
        leftUp = face[159]
        leftDown = face[23]
        leftLeft = face[130]
        leftRight = face[243]
        Vertical_length,_ = detector.findDistance(leftUp,leftDown)
        Horizontal_length,_ = detector.findDistance(leftLeft,leftRight)
        blink_test = int((Vertical_length/Horizontal_length)*100)
        # print(blink_test)
        if (blink_test<=29):
            print("Blinked----")
    cv2.imshow("Image",img)
    cv2.waitKey(1)