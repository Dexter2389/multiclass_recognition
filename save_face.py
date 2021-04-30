# MIT License
#
# Copyright (c) 2019 Saurabh Ghanekar
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# 
# Do not remove or modify any license notices.
# ==============================================================================

import cv2
import numpy as np
import os
import time
import dlib
from imutils.video import VideoStream
from imutils import face_utils
from imutils.face_utils import FaceAligner
import urllib.request

DIR = "faces/"
MODEL = "dlib_face_recognition_resnet_model_v1.dat"
SHAPE_PREDICTOR = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(SHAPE_PREDICTOR)
face_aligner = FaceAligner(shape_predictor, desiredFaceWidth=250)

if not os.path.exists(MODEL) or not os.path.exists(SHAPE_PREDICTOR):
    print('Beginning model download')
    
    if not os.path.exists(MODEL):
        url = "https://raw.githubusercontent.com/ageitgey/face_recognition_models/master/face_recognition_models/models/dlib_face_recognition_resnet_model_v1.dat"
        urllib.request.urlretrieve(url, MODEL)
    
    if not os.path.exists(SHAPE_PREDICTOR):
        url = "https://raw.githubusercontent.com/tzutalin/dlib-android/master/data/shape_predictor_68_face_landmarks.dat"
        urllib.request.urlretrieve(url, SHAPE_PREDICTOR)

def create_dir(name):
    if not os.path.exists(name):
        os.mkdir(name)


create_dir(name=DIR)

print("[INFO] Preparing to prepare dataset")

# Geting Face ID
while True:
    face_id = input("[Input] Enter ID for face: ")
    try:
        face_id = int(face_id)
        face_dir = DIR + str(face_id) + "/"
        create_dir(face_dir)
        print("[INFO] Directory created to store your dataset")
        break

    except:
        print("[Info] Invalid Input. ID must be an integer")
        continue

print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
total_imgs = 0

# Get Face Images
while True:
    frame = vs.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(frame_gray)

    if len(faces) == 1:
        face = faces[0]
        (x, y, w, h) = face_utils.rect_to_bb(face)
        face_img = frame_gray[y:y+h, x:x+w]
        face_align = face_aligner.align(frame, frame_gray, face)

        face_img = face_align
        key = cv2.waitKey(1) & 0xFF
        if key == ord("k"):
            save_img_path = face_dir + str(total_imgs) + ".png"
            cv2.imwrite(save_img_path, face_img)
            total_imgs += 1
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 3)
        cv2.imshow("Aligned", face_img)

    cv2.imshow("Preparing Dataset", frame)
    key = cv2.waitKey(1) & 0xFF
    cv2.waitKey(1)

    if key == ord("q"):
        break
    elif total_imgs == 300:
        break


print("[INFO] {} face images stored".format(total_imgs))
print("[INFO] cleaning up...")
cv2.destroyAllWindows()
vs.stop()