#!/usr/bin/env python

from filepaths import *

import os
import json

class Dataset:
    def __init__(self, fname, img_size=(299, 299)):
        self.fname = fname
        self.img_size = img_size
        self.cap_dict = self.read_captions()
        self.model = self.get_model()

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

if __name__ == "__main__":
    fname = os.path.join(DATA_PATH, "annotations/captions_train2017.json")
    training_data = Dataset(fname)
    print(len(training_data.cap_dict))
