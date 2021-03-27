from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import os


directory=""
images=os.listdir(directory)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

landmarks=np.array(126)

for i in images:
	img=cv2.imread(i)
	img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	shape=predictor(img,detector(img,1))
	coords=face_utils.shape_to_np(shape).reshape(126)
	landmarks=np.append(landmarks,coords)

np.save("landmarks.npy",landmarks)







