from datetime import datetime
from pydoc import classname
import cv2
from cv2 import waitKey
import numpy as np
import face_recognition
import os

def findEncoding(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList
def MarkAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            timestr = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{timestr}')

path = 'dataImg'
images = []
classanames = []
mylist = os.listdir(path)
print(mylist) #// list of all the images


for cls in mylist:
    currImg = cv2.imread(f'{path}/{cls}') # name of the image cls
    images.append(currImg)
    classanames.append(os.path.splitext(cls)[0]) # will give name of the student
#print(classanames)

encodeListKnown = findEncoding(images)
print(len(encodeListKnown))
print("Encoding completes")


vid = cv2.VideoCapture(0)
while True:
    ret,img = vid.read()
    imgs = cv2.resize(img,(0,0),None,0.25,0.25)
    imgs = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    
    faceCurrFrame = face_recognition.face_locations(imgs)
    encodeCurrframe = face_recognition.face_encodings(imgs,faceCurrFrame)
    
    for encodeFace,FaceLoc in zip(encodeCurrframe,faceCurrFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        # print(faceDis)
        matchIndex = np.argmin(faceDis)
        
        if matches[matchIndex]:
            name = classanames[matchIndex].upper()
            print(name)
            y1,x2,y2,x1 = FaceLoc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            MarkAttendance(name)
        else:
            print("No match found")
            
        
    
    cv2.imshow('frame', img)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break





vid.release()
# Destroy all the windows
cv2.destroyAllWindows()