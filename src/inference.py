#!/usr/bin/env python

import os
import sys
import torch
import pickle
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

from filepaths import *
from params import *
from encoder import Encoder
from decoder import Decoder
from vocab import Vocabulary

def get_caption(img_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.ToTensor(), 
                                    transforms.Normalize(
                                        (0.485, 0.456, 0.406), 
                                        (0.229, 0.224, 0.225))])

    with open(VOCAB_FILE, "rb") as f:
        vocab = pickle.load(f)

    encoder = Encoder(embed_size).eval().to(device)
    decoder = Decoder(embed_size, hidden_size, len(vocab), num_layers).to(device)
    encoder.load_state_dict(torch.load(ENCODER_FILE))
    decoder.load_state_dict(torch.load(DECODER_FILE))

    img = Image.open(img_path).resize(img_size, Image.LANCZOS)
    img = transform(img).unsqueeze(0)
    img = img.to(device)

    feat = encoder(img)
    out = decoder.sample(feat)[0].cpu().numpy()
    caption = list()

    for i in out:
        word = vocab.idx2word[i]
        caption.append(word)
        if word == "<end>": break

    return " ".join(caption)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: $./inference.py <img_path>")
        exit(1)

    img_path = os.path.abspath(sys.argv[1])
    img = Image.open(img_path)
    
    caption = get_caption(img_path)
    print("Caption: ", caption)
    
    plt.title(caption)
    plt.imshow(img)
    plt.savefig("caption.png")
    plt.show()
