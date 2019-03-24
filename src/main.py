#!/usr/bin/env python

import os
import tensorflow as tf

from filepaths import *
from dataset import Dataset
from encoder import Encoder

def main():
    tf.enable_eager_execution()
    fname = os.path.join(DATA_PATH, "annotations/captions_train2017.json")
    training_data = Dataset(fname)
    encoder = Encoder(training_data)
    encoder.get_features()

if __name__ == "__main__":
    main()
