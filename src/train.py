#!/usr/bin/env python

import torch
import pickle
import torchvision.datasets as dset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.nn.utils.rnn import pack_padded_sequence

from filepaths import *
from decoder import Decoder
from encoder import Encoder
from vocab import Vocabulary
from utils import collate_fn
from coco_dataset import COCODataset

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

coco = COCODataset(root=TRAIN_IMG_DIR, 
                   json=CAP_FILE,
                   vocab = vocab,
                   transform=transform)

dl = DataLoader(dataset=coco, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, collate_fn=collate_fn)
                                    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


encoder = Encoder(embed_size).to(device)
decoder = Decoder(embed_size, hidden_size, len(vocab), num_layers).to(device)


params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
optimizer = torch.optim.Adam(params, lr=alpha)

total_step = len(dl)
print(total_step)

for epoch in range(num_epochs):
    for i, (imgs, caps, lens) in enumerate(dl):
        imgs = imgs.to(device)
        caps = caps.to(device)
        targets = pack_padded_sequence(caps, lens, batch_first=True)[0]

        feats = encoder(imgs)
        out = decoder(feats, caps, lens)
        loss = criterion(out, targets)
        decoder.zero_grad()
        encoder.zero_grad()
        loss.backward()
        optimizer.step()
        print("loss = {}".format(loss.item()))
    
    d_path = os.path.join(MODEL_PATH, "decoder-epoch{}.ckpt".format(epoch))
    e_path = os.path.join(MODEL_PATH, "encoder-epoch{}.ckpt".format(epoch))
    torch.save(decoder.state_dict(), d_path) 
    torch.save(encoder.state_dict(), d_path) 
