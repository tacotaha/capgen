#!/usr/bin/env python

import os
import torch
import pickle
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

from filepaths import *
from model_params import *
from encoder import Encoder
from decoder import Decoder
from vocab import Vocabulary


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([transforms.ToTensor(), 
                                transforms.Normalize(
                                    (0.485, 0.456, 0.406), 
                                    (0.229, 0.224, 0.225))])

with open(VOCAB_PATH, "rb") as f:
    vocab = pickle.load(f)
