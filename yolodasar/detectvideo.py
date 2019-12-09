import cv2
import numpy as np
import time

videoCam = cv2.VideoCapture(0)
time.sleep(2)

face = cv2.CascadeClassifier('face-detect.xml')
eye = cv2.CascadeClassifier('eye-detect.xml')

while True:
    ret, frame = videoCam.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    muka = face.detectMultiScale(gray, 1.05, 3)
    for (x,y,w,h) in muka:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 5)

        roi_warna = frame[y:y+h, x:x+w]
        roi_gray = gray[y:y+h, x:x+w]
        mata = eye.detectMultiScale(roi_gray)
        for (mx,my,mw,mh)in mata:
            cv2.rectangle(roi_warna, (mx,my), (mx+mw, my+mh), (255,255,0), 2)

    cv2.imshow('webcam', frame)

    k = cv2.waitKey(1) & 0xff
    if k == ord('q'):
        break

videoCam.release()
cv2.destroyAllWindows()