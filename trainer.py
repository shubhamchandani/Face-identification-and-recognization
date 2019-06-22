import cv2
import numpy as np
from PIL import Image
import os
face_samples = []
ids = []
path = '/home/shubham/image/dataset/'
recognizer = cv2.face.LBPHFaceRecognizer_create()
for i in os.listdir(path):
	ip = os.path.join(path,i)
	faceImg = Image.open(ip).convert('L')
	faceNp = np.array(faceImg,'uint8')
	face_samples.append(faceNp)
	idd = int((os.path.split(ip)[-1].split(".")[0]))
	ids.append(idd)
trainer = recognizer.train(face_samples,np.array(ids))
recognizer.save('trainer.yml') 

	

