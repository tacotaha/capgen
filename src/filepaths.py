import os

"""
Global data path definitions
"""

DATA_PATH = os.path.abspath("../data")
MODEL_DIR = os.path.join(DATA_PATH, "model")
CAP_DIR = os.path.join(DATA_PATH, "annotations")

TRAIN_CAP_FILE = os.path.join(CAP_DIR, "captions_train2014.json")
TRAIN_IMG_DIR = os.path.join(DATA_PATH, "train2014")

VAL_CAP_FILE = os.path.join(CAP_DIR, "captions_val2014.json")
VAL_IMG_DIR = os.path.join(DATA_PATH, "val2014")

VOCAB_FILE = os.path.join(DATA_PATH, "vocab.pkl")
ENCODER_FILE = os.path.join(MODEL_DIR, "encoder.ckpt")
DECODER_FILE = os.path.join(MODEL_DIR, "decoder.ckpt")
