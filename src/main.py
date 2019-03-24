#!/usr/bin/env python

import os
import tensorflow as tf

from filepaths import *
from dataset import Dataset
from encoder import Encoder

def check_device():
    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())

def main():
    tf.enable_eager_execution()
    fname = os.path.join(CAP_DIR, "captions_train2017.json")
    training_data = Dataset(fname, num_caps=50000)
    encoder = Encoder(training_data)
    encoder.get_features()

if __name__ == "__main__":
    print("===========CHECKING DEVICE=============")
    check_device()
    print("======DONE CHECKING DEVICE=============")
    main()
