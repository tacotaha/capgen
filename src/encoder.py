#!/usr/bin/env python

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import inception_v3

from dataset import Dataset

class Encoder:
    def __init__(self, dataset, batch_num=20):
        assert isinstance(dataset, Dataset)
        self.dataset = dataset
        self.batch_num = batch_num
        self.images = self.process_dataset()
        self.model = self.get_model()

    def process_dataset(self):
        fnames = sorted(set(self.dataset.images))
        tf_dataset = tf.data.Dataset.from_tensor_slices(fnames)
        return tf_dataset.map(self.process_image).batch(self.batch_num)

    def process_image(self,img_path):
        img = tf.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, self.dataset.img_size)
        return inception_v3.preprocess_input(img), img_path

    def get_model(self, include_top=False, weights='imagenet'):
        img_model = InceptionV3(include_top=include_top, weights=weights)
        return tf.keras.Model(img_model.input, img_model.layers[-1].output)

    def get_features(self):
        for x, y in self.images:
            feats = self.model(x)
            feats = tf.reshape(feats, (feats.shape[0], -1, feats.shape[3]))
            for f, p in zip(feats, y):
                path = p.numpy().decode()
                if not os.path.exists(path + ".npy"):
                    np.save(path, f.numpy())
