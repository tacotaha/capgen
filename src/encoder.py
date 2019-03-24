#!/usr/bin/env python
import json
import numpy as np
import tensorflow as tf
from tf.keras.applications import InceptionV3

from dataset import Dataset

class Encoder:
    def __init__(self, dataset):
        assert isinstance(dataset, Datset)
        self.dataset = dataset
        self.images = self.process_dataset()
        self.model = self.get_model()

    def process_dataset(self, batch_num=20):
        fnames = sorted(set(self.dataset.keys()))
        tf_dataset = tf.data.Datset.from_tensor_slices(fnames)
        return tf_dataset.map(dataset.process_image).batch(batch_num)

    def process_image(self,img_path):
        img = tf.read_file(img_path)
        img = tf.image_decode_jpeg(img)
        img = tf.image_resize(img, self.img_size)
        img = tf.keras.inception_v3.preprocess_input(img)
        return img, img_path

    def get_model(self, include_top=False, weights='imagenet'):
        img_model = InceptionV3(include_top=include_top, weights=weights)
        return tf.keras.model(img_model.input, img_model.layers[-1].output)

    def get_features(self):
        for x, y in self.images:
            features = self.model(x)
            features = tf.reshape(features, features.shape[0], -1, features.shape[3])
            for f, p in zip(features, y):
                np.save(p.numpy().decode(), f.numpy())
