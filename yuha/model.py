import glob
import h5py
import numpy as np
import os
import random

import keras.backend as K
from PIL import Image
from keras import Input
from keras import objectives
from keras.engine import Model
from keras.layers import Dense
from keras.models import load_model
from keras.preprocessing import image as keras_image
from keras.utils import to_categorical

from yuha.util import HDF5_PATH, LABEL, TOP_LAYER_WEIGHT_PATH, DONE_FILE_PATH, INTERMEDIATE_MODEL_PATH


class Models(object):
    def __init__(self):
        self.intermediate_layer = load_model(INTERMEDIATE_MODEL_PATH)
        self.intermediate_layer._make_predict_function()
        self.label = LABEL
        self.top_layer = self.create_top_layer(len(self.label))
        self.top_layer._make_predict_function()
        self.top_layer._make_train_function()
        with h5py.File(HDF5_PATH, 'r') as h5:
            data, label = h5['features'], h5['label']
            self.all_label = label[:]
            self.data = data[:]

    @staticmethod
    def create_top_layer(num_label):
        input_prediction = Input(shape=(2048,))
        predictions = Dense(num_label, activation='softmax', name='predictions')(input_prediction)
        top_layer = Model(input_prediction, predictions)
        top_layer.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        top_layer.load_weights(TOP_LAYER_WEIGHT_PATH)
        return top_layer

    @staticmethod
    def create_similarity_layer(num_label):
        input_prediction = Input(shape=(2408,))
        x = Dense(128, activation='relu', name='fc2')(input_prediction)
        prediction = Dense(num_label, activation='softmax', name='predictions')(x)
        similarity_layer = Model(input_prediction, prediction)
        similarity_layer.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        return similarity_layer

    @staticmethod
    def preprocess(image):
        img = image.resize(tuple([244, 244]))
        x = keras_image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x /= 255.
        return x

    def similarity(self, x, name):
        data = self.data[np.where(self.all_label == name)]
        data = np.average(data, axis=0)
        return -K.eval(objectives.cosine_proximity(np.squeeze(x), data))

    def predict_intermediate_layer(self, image):
        image_np = self.preprocess(image)
        return self.intermediate_layer.predict(image_np)

    def predict(self, image):
        intermediate_layer_output = self.predict_intermediate_layer(image)
        prediction = self.top_layer.predict(intermediate_layer_output)
        max = np.argmax(prediction)
        name, score = self.label[max], np.squeeze(prediction)[max]
        cosine_distance = self.similarity(intermediate_layer_output, name)
        # TODO find another way to calculate similarity
        # if score < 0.7 or cosine_distance < 0.9:
        #     name = 'Unknwon'
        return name, str(score)

    def registration_train(self, images, labels):
        if not os.path.exists(HDF5_PATH):
            intermediate_layer_output = []
            for image in images:
                image = Image.open(image)
                intermediate_layer_output.append(np.squeeze(self.predict_intermediate_layer(image)))
            with h5py.File(HDF5_PATH, 'w') as h5:
                h5.create_dataset('features', data=np.array(intermediate_layer_output), maxshape=(None, None))
                h5.create_dataset('label', data=np.array(labels), maxshape=(None,))
                data = h5['features'][:]
                label = h5['label'][:]
        else:
            with h5py.File(HDF5_PATH, 'a') as h5:
                data, label = h5['features'], h5['label']
                for image, label_ in zip(images, labels):
                    image = Image.open(image)
                    data.resize(((data.shape[0] + 1), data.shape[1]))
                    label.resize(((label.shape[0] + 1),))
                    data[-1:] = np.squeeze(self.predict_intermediate_layer(image))
                    label[-1:] = str(label_)
                data = data[:]
                label = label[:]
                self.data = data
                self.all_label = label

        self.label = list(self.label)
        num_label = len(self.label)

        label = np.squeeze(np.array([self.label.index(x) for x in label]))
        label = to_categorical(label)

        indices = range(len(label))
        random.shuffle(indices)
        data = data[indices]
        label = label[indices]

        self.top_layer.layers.pop()
        avg_pool = self.top_layer.layers[-1].output
        predictions = Dense(num_label, activation='softmax', name='predictions')(avg_pool)
        self.top_layer = Model(inputs=self.top_layer.input, outputs=predictions)
        self.top_layer.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

        self.top_layer.fit(data, label, epochs=150, batch_size=10, validation_split=0.25)
        self.top_layer.save_weights(TOP_LAYER_WEIGHT_PATH)

    def registration(self, images, label):
        labels = [label] * len(images)
        if not os.path.exists(DONE_FILE_PATH):
            with open(DONE_FILE_PATH, 'w') as file_:
                file_.write(label)
        # TODO replace intermediate_layer_output
        files = glob.glob('images/production/{}/*'.format(label))
        if label not in self.label:
            with open('models/label.txt', 'a') as file_:
                file_.write('\n' + label)
        self.registration_train(files, labels)
        with open(DONE_FILE_PATH, 'a') as file_:
            file_.write('\n')
            file_.write(label)
