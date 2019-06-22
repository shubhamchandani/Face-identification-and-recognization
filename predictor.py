import cv2
import numpy as np
from PIL import Image
import os
path ='/home/shubham/image/dataset/'
face_data = cv2.CascadeClassifier('face_data.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
data= recognizer.read('trainer.yml')
cap = cv2.VideoCapture(0)
while True:
	status,frame = cap.read(0)
	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	faces = face_data.detectMultiScale(gray,1.3,5)
	for (x,y,w,h) in faces:
		cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
		roi_gray = gray[y:y+h,x:x+w]
		roi_img = frame[y:y+h,x:x+w]
		idd1, conf = recognizer.predict(roi_gray)
		font = font = cv2.FONT_HERSHEY_SIMPLEX
		for i in os.listdir(path):
			ip = os.path.join(path,i)
			if (int((os.path.split(ip)[-1].split(".")[0])) == int(idd1)):
				nam = (os.path.split(ip)[-1].split(".")[1])
				#print(nam)
		cv2.putText(frame,str(nam) +"." +str(conf) ,(x,y-10),font,0.55,(120,255,120),1)
		cv2.imshow("predictor",frame)
	if cv2.waitKey(100) & 0xFF == ord('q'):
		break;

cap.release();


