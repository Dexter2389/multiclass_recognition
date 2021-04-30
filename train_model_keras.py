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

from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.utils import np_utils
from tensorflow.keras.callbacks import ModelCheckpoint
from keras import backend as K
import numpy as np
import pickle
import os
K.set_image_dim_ordering('tf')


def get_num_of_faces():
    return len(os.listdir('dataset_faces/'))


def mlp_model():
    num_of_faces = get_num_of_faces()
    model = Sequential()
    model.add(Dense(128, input_shape=(128, ), activation='relu'))
    model.add(Dropout(0.8))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.8))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.8))
    model.add(Dense(num_of_faces, activation='softmax'))
    adam = optimizers.Adam(lr=1e-2)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam, metrics=['accuracy'])
    filepath = "mlp_model_keras2.h5"
    checkpoint1 = ModelCheckpoint(
        filepath, monitor='acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint1]
    from keras.utils import plot_model
    plot_model(model, to_file='model.png', show_shapes=True)
    return model, callbacks_list


def train():
    with open("train_features", "rb") as f:
        train_images = np.array(pickle.load(f))
    with open("train_labels", "rb") as f:
        train_labels = np.array(pickle.load(f), dtype=np.int32)

    with open("test_features", "rb") as f:
        test_images = np.array(pickle.load(f))
    with open("test_labels", "rb") as f:
        test_labels = np.array(pickle.load(f), dtype=np.int32)

    train_labels = np_utils.to_categorical(train_labels)
    test_labels = np_utils.to_categorical(test_labels)
    model, callbacks_list = mlp_model()
    model.fit(train_images, train_labels, validation_data=(
        test_images, test_labels), epochs=1500, batch_size=50, callbacks=callbacks_list)
    scores = model.evaluate(test_images, test_labels, verbose=3)
    print(scores)
    print("MLP Error: %.2f%%" % (100-scores[1]*100))


train()
