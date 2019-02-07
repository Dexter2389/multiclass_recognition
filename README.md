# Multi-class Face Recognition
Face recognition is a method of identifying or verifying the identity of an individual using their face. With the rise in the popularity of Apple's FaceID, OnePlus Face Unlock fearture, and other companies also following this trend, I wanted a face unlock method for my lab. So I developed a simple multi-layer perceptron neural network model with Adam optimizer to achieve this.
This project uses dlib's face recognition resnet model for face detection. The detected faces' facial features are then encoded using 68 face landmarks shape predictor. Once the faces are encoded, they are passed through a 3 layered mlp with adam optimizer.

## Requirements
* Set up a python environment with required dependencies installed.

## Usage
### For Python Environment:
#### 1. Downloading this Respository
  Start by [downloading](https://github.com/Dexter2389/multiclass_recognition/archive/master.zip) or clone the repository:
  
  ```
  $ git clone https://github.com/Dexter2389/multiclass_recognition.git
  $ cd multiclass_recognition
  ```

#### 2. Install Dependencies and getting important files
  * In order to run this repository, you will need to get dlib's 68 face landmarks file and face recognition resnet model. You can download the and extract by running:

  [For dlib's 68 Face Landmarks file]
  ```
  $ cd multiclass_recognition
  $ wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
  $ bzip2 -dk shape_predictor_68_face_landmarks.dat.bz2
  ```

  [For dlib's face recognition resnet model file]
  ```
  $ cd multiclass_recognition
  $ wget http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2
  $ bzip2 -dk dlib_face_recognition_resnet_model_v1.dat.bz2
  ```
  * You will also need to install specific python dependencies for this project:
  
  ```
  pip install -r requirements.txt
  ```

#### 3. Running the program
  1. Run ```save_face.py``` to start the face capture process. I would recommend to capture atleast 250-300 photos of atleast 3 people.
  [But if you have already have your train images, please amke sure you save it in a folder called "faces"]

  2. Run ```pickling_facial_features.py``` to start the encoding process.

  3. Run ```train_model_keras.py``` to start the training process.

  4. Once the training is done, run ```recognize.py``` to see the output of you model.

Bravo!! Guess what you computer know how your face looks with just 250 to 300 images....

## Acknowledgements
  * Thanks to Adam Geitgey for his (awsome article)[https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78] which gave me great ideas to implement this project.
  * Also thanks to (Adrian Rosebrock)[https://www.pyimagesearch.com/2018/06/18/face-recognition-with-opencv-python-and-deep-learning/] for kick starting my project.