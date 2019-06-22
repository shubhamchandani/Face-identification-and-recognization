import numpy as np
import cv2
import os

face_data = cv2.CascadeClassifier('face_data.xml')
name = (input("Enter your name "))
idd = int(input("Enter your id"))
#try:
#	os.mkdir(name)
#except:
#	print("Already except")
count = 0
cap = cv2.VideoCapture(0)
while True:
	status,frame = cap.read()
	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	faces = face_data.detectMultiScale(gray,1.3,5)
	for (x,y,w,h) in faces:
		cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
		roi_gray = gray[y:y+h,x:x+w]
		roi_img = frame[y:y+h,x:x+w]
		cv2.imwrite("dataset/"+str(idd)+"."+str(name)+"."+str(count)+".jpg",roi_gray )
		#print (str(name)+"/"+str(count)+".jpg")
		count += 1		
		cv2.imshow(name,frame)
	if cv2.waitKey(100) & 0xFF == ord('q'):
			break
	elif count == 30:
			break

cap.release()



