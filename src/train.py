#!/usr/bin/env python

import torch
import pickle
import torchvision.datasets as dset
import torchvision.transforms as transforms

from filepaths import *
from decoder import Decoder
from encoder import Encoder
from vocab import Vocabulary
from utils import collate_fn

# Model Params
embed_size = 256
hidden_size = 512
num_layers = 1
num_epochs = 5
batch_size = 128
num_workers = 2
alpha = 0.001
crop_size = 224
criterion = torch.nn.CrossEntropyLoss()

with open(VOCAB_FILE, "rb") as f:
    vocab = pickle.load(f)

transform = transforms.Compose([ 
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
			     (0.229, 0.224, 0.225))])

cap = dset.CocoCaptions(root=TRAIN_IMG_DIR, 
                        annFile=CAP_FILE,
                        transform=transform)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


encoder = Encoder(embed_size).to(device)
decoder = Decoder(embed_size, hidden_size, len(vocab), num_layers).to(device)


params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
optimizer = torch.optim.Adam(params, lr=alpha)

total_step = len(cap)
print(total_step)

for epoch in range(num_epochs):
    for i, (img, cap) in enumerate(cap):
        print(i, img, cap)
