import numpy as np
import cv2

cap = cv2.VideoCapture(0)

eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascade+'haarcascade_eye.xml')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascade+"haarcascade_frontalface_alt.xml")

smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascade+'haarcascade_smile.xml')


while 1:
    ret , img = cap.read()
    gray = img
    faces = face_cascade.DetectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:
        cv2.rectange(img,(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_color = img[y:y+h,x:x+w]

        eyes = eye_cascade.DetectMultiScale(roi_gray)

        for(ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    
cap.release()
cv2.destroyAllWindows()
