import cv2
import numpy as np
import dlib
import pickle
import os
import csv
import shutil
from random import shuffle
import numpy as np

MODEL = "dlib_face_recognition_resnet_model_v1.dat"
SHAPE_PREDICTOR = "shape_predictor_68_face_landmarks.dat"
INITIAL_FACES = "faces/"
FACE_DATASET = 'dataset_faces/'
CSV_FILE = "dataset.csv"


def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)


def csv_to_pickel(csv_file):
	print("\n[INFO] Initiating CSV to pickle conversion\n")
	features = []
	labels = []
	with open(csv_file, newline="") as csvfile:
		reader = csv.reader(csvfile)
		for row in reader:
			labels.append(int(row[0]))
			features.append(np.array(row[1:], dtype=np.float32))
	
	no_of_faces = len(features)
	train_features = features[:int(0.9*no_of_faces)]
	train_labels = labels[:int(0.9*no_of_faces)]
	test_features = features[int(0.9*no_of_faces):]
	test_labels = labels[int(0.9*no_of_faces):]
	
	print("\t[INFO] Length of train_features: ", len(train_features))
	with open('train_features', 'wb') as f:
		pickle.dump(train_features, f)
	del train_features
	
	print("\t[INFO] Length of train_labels: ", len(train_labels))
	with open('train_labels', 'wb') as f:
	    pickle.dump(train_labels, f)
	del train_labels
	
	print("\t[INFO] Length of test_features: ", len(test_features))
	with open('test_features', 'wb') as f:
	    pickle.dump(test_features, f)
	del test_features
	
	print("\t[INFO] Length of test_labels: ", len(test_labels))
	with open('test_labels', 'wb') as f:
	    pickle.dump(test_labels, f)
	del test_labels


print("\n[INFO] Initiating to store facial features...\n")

face_rec = dlib.face_recognition_model_v1(MODEL)
shape_predictor = dlib.shape_predictor(SHAPE_PREDICTOR)
detector = dlib.get_frontal_face_detector()

create_folder(FACE_DATASET)

folder_names = os.listdir(INITIAL_FACES)
face_descriptors = []
row = ""
folder_count = 0
for folder_name in folder_names:
    folder_count += 1
    create_folder(FACE_DATASET+folder_name)
    full_folder_path = INITIAL_FACES+folder_name+"/"
    print("[INFO] Storing facial features of folder " +
          str(folder_count) + "/" + str(len(folder_names)))

    images = os.listdir(full_folder_path)
    image_count = 0
    errors = 0
    success = 0
    for image in images:
        image_count += 1
        full_image_path = full_folder_path+image
        print(str(folder_count) + "/" + str(len(folder_names)) + "\t[INFO] Working with " + str(image_count) + "/" + str(len(images)))
        img = cv2.imread(full_image_path)
        cv2.imwrite(FACE_DATASET+folder_name+"/"+image, img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        try:
            face = detector(img, 1)[0]
            print("\t\t[INFO] Success.!")
            success += 1
        except:
            print("\t\t[INFO] Error.!")
            errors += 1
            continue
        shape = shape_predictor(img, face)
        face_descriptor = face_rec.compute_face_descriptor(img, shape)
        face_descriptor = list(face_descriptor)
        face_descriptor.insert(0, int(folder_name))
        with open(CSV_FILE, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(face_descriptor)
        print("Successful Encoded : " + str(success) + "| Error while Encoding : " + str(errors))
    shutil.rmtree(full_folder_path)
shutil.rmtree(INITIAL_FACES)

print("[INFO] cleaning up...")

with open(CSV_FILE) as f:
    li = f.readlines()

shuffle(li)
shuffle(li)
shuffle(li)
with open(CSV_FILE, 'w') as f:
    f.writelines(li)

print("[INFO] Facial Features successfully stored in " + str(CSV_FILE) + "...")

csv_to_pickel(CSV_FILE)
