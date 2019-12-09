import cv2
import numpy as np

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "Classifiers/face.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
Id = 0

cam = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
while True:
    ret, im =cam.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray, 1.2,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
        Id, conf = recognizer.predict(gray[y:y+h,x:x+w])
        if(conf<50):
            if(Id==1):
                Id="Dezha"
            elif(Id==2):
                Id="Anam"
        else:
            Id="ini teh saha?"
        cv2.putText(im,str(Id), (x,y+h),font,2,(0,255,0), 2)
    cv2.imshow('webcam',im) 
    if cv2.waitKey(10) & 0xFF==ord('x'):
        break
cam.release()
cv2.destroyAllWindows()
