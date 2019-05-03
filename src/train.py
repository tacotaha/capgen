#!/usr/bin/env python

import torch
import pickle
import torchvision.datasets as dset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.nn.utils.rnn import pack_padded_sequence

from params import *
from filepaths import *
from decoder import Decoder
from encoder import Encoder
from vocab import Vocabulary
from utils import collate_fn
from coco_dataset import COCODataset

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
epoch_losses = []
for epoch in range(num_epochs):
    losses = []
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
        losses.append(loss.item())
        print("loss = {}".format(loss.item()))
    
    epoch_losses.append(sum(losses)/len(losses))
    with open("epoch_losses.pkl", "wb") as f:
        pickle.dump(epoch_losses, f)

    d_path = os.path.join(MODEL_PATH, "decoder-epoch{}.ckpt".format(epoch))
    e_path = os.path.join(MODEL_PATH, "encoder-epoch{}.ckpt".format(epoch))
    torch.save(decoder.state_dict(), d_path) 
    torch.save(encoder.state_dict(), d_path) 
