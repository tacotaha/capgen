#!/usr/bin/env python

from filepaths import *

import os
import json
from sklearn.utils import shuffle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class Dataset:
    def __init__(self, fname, img_size=(299, 299), shuffle_data=True,
                 num_caps=None, num_words=50000):
        self.fname = fname
        self.img_size = img_size
        self.num_caps = num_caps
        self.shuffle_data = shuffle_data
        self.images = list()
        self.captions =  list()
        self.read_captions()
        self.tokenizer = Tokenizer(num_words=num_words, oov_token="")
        self.cap_toks = self.tokenize_captions()
        self.max_cap_len = max(len(t) for t in self.cap_toks)

    def read_captions(self):
        with open(self.fname, "r") as f:
            caps = json.load(f)

        counter = 0
        for cap in caps["annotations"]:
            if counter < self.num_caps:
                caption = cap["caption"]
                img_id = cap["image_id"]
                img_path = os.path.join(TRAIN_IMG_DIR, "%012d.jpg" % img_id)
                self.images.append(img_path)
                self.captions.append(caption)
            counter += 1

        if self.shuffle_data:
            self.captions, self.images = shuffle(self.captions, self.images)

    def tokenize_captions(self, pad_seqs=True):
        self.tokenizer.fit_on_texts(self.captions)
        toks = self.tokenizer.texts_to_sequences(self.captions)
        return pad_sequences(toks, padding="post") if pad_seqs else toks

if __name__ == "__main__":
    fname = os.path.join(CAP_DIR, "captions_train2017.json")
    training_data = Dataset(fname, num_caps=50000)
    print("Read {} captions.".format(len(training_data.captions)))
    print(training_data.cap_toks[:5])
