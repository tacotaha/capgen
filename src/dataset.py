#!/usr/bin/env python

from filepaths import *

import os
import json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class Dataset:
    def __init__(self, fname, img_size=(299, 299), num_words=50000):
        self.fname = fname
        self.img_size = img_size
        self.cap_dict = self.read_captions()
        self.tokenizer = Tokenizer(num_words=num_words, oov_token="")
        self.cap_toks = self.tokenize_captions()
        self.max_cap_len = max(len(t) for t in self.cap_toks)

    def read_captions(self):
        """
        fname: input filename containing annotations
        returns: dict: img_file_name -> list(captions)
        """
        cap_dict = {}
        with open(self.fname, "r") as f:
            caps = json.load(f)

        for cap in caps["annotations"]:
            captions = cap["caption"]
            img_id = cap["image_id"]
            img_path = os.path.join(TRAIN_IMG_DIR, "%012d.jpg" % img_id)
            cap_dict[img_path] = captions

        return cap_dict

    def tokenize_captions(self):
        self.tokenizer.fit_on_texts(self.cap_dict.values())
        toks = self.tokenizer.texts_to_sequences(self.cap_dict.values())
        return pad_sequences(toks, padding="post")

if __name__ == "__main__":
    fname = os.path.join(DATA_PATH, "annotations/captions_train2017.json")
    training_data = Dataset(fname)
    print("Read {} captions.".format(len(training_data.cap_dict)))
    print(training_data.cap_toks[0])
