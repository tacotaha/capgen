#!/usr/bin/env python

from filepaths import *

import os
import json
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from multiprocessing import cpu_count
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import inception_v3
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class Dataset:
    def __init__(self, fname, batch_num=64,img_size=(299, 299), shuffle_data=True,
                 num_caps=None, num_words=5000):
        self.fname = fname
        self.pad_tok = "<pad>"
        self.start_tok = "<start>"
        self.end_tok = "<end>"
        self.unk_tok = "<unk>"
        self.num_words = num_words
        self.batch_num = batch_num
        self.img_size = img_size
        self.num_caps = num_caps
        self.shuffle_data = shuffle_data
        self.images = list()
        self.captions =  list()
        self.read_captions()
        punctuation = '!"#$%&()*+.,-/:;=?@[\]^_`{|}~ '
        self.tokenizer = Tokenizer(num_words=num_words, oov_token=self.unk_tok, 
                                   filters=punctuation)
        self.cap_toks = self.tokenize_captions()
        self.max_cap_len = max(len(t) for t in self.cap_toks)

    def read_captions(self):
        with open(self.fname, "r") as f:
            caps = json.load(f)

        counter = 0
        for cap in caps["annotations"]:
            if not self.num_caps or counter < self.num_caps:
                caption = self.start_tok + cap["caption"] + self.end_tok
                img_id = cap["image_id"]
                img_path = os.path.join(TRAIN_IMG_DIR, "%012d.jpg" % img_id)
                self.images.append(img_path)
                self.captions.append(caption)
            counter += 1

        if self.shuffle_data:
            self.captions, self.images = shuffle(self.captions, self.images)

    def tokenize_captions(self, pad_seqs=True):
        self.tokenizer.fit_on_texts(self.captions)
        self.tokenizer.word_index = {x:y for x, y in\
                self.tokenizer.word_index.items() if y <= self.num_words}
#        self.tokenizer.word_index[self.tokenizer.oov_token] = self.num_words + 1
        self.tokenizer.word_index[self.pad_tok] = 0
        toks = self.tokenizer.texts_to_sequences(self.captions)
        return pad_sequences(toks, padding="post") if pad_seqs else toks

    def process_dataset(self):
        fnames = sorted(set(self.images))
        tf_dataset = tf.data.Dataset.from_tensor_slices(fnames)
        self.images = tf_dataset.map(self.process_image).batch(self.batch_num)
        return self.images

    def process_image(self, img_path):
        img = tf.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, self.img_size)
        return inception_v3.preprocess_input(img), img_path

    def get_model(self, include_top=False, weights='imagenet'):
        img_model = InceptionV3(include_top=include_top, weights=weights)
        self.model = tf.keras.Model(img_model.input, img_model.layers[-1].output)
        return self.model

    def get_features(self):
        for x, y in self.images:
            feats = self.model(x)
            feats = tf.reshape(feats, (feats.shape[0], -1, feats.shape[3]))
            for f, p in zip(feats, y):
                path = p.numpy().decode()
                if not os.path.exists(path + ".npy"):
                    np.save(path, f.numpy())

    def load_feature(self, img, cap):
        return np.load(img.decode() + ".npy"), cap
    
    def load_features(self, imgs, caps):
        data = tf.data.Dataset.from_tensor_slices((imgs, caps))
        lam = lambda x, y: tf.py_func(self.load_feature, [x, y], [tf.float32, tf.int32])
        return data.map(lam, num_parallel_calls=cpu_count())

if __name__ == "__main__":
    fname = os.path.join(CAP_DIR, "captions_train2017.json")
    training_data = Dataset(fname, num_caps=50000)
    print("Read {} captions.".format(len(training_data.captions)))
    print(training_data.cap_toks[:5])
