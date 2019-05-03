import os
import torch
from PIL import Image
from nltk import word_tokenize
import torch.utils.data as data
from pycocotools.coco import COCO
import torchvision.transforms as transforms

class COCODataset(data.Dataset):

    def __init__(self, root, json, vocab, transform=None):
        self.root = root
        self.coco = COCO(json)
        self.ids = list(self.coco.anns.keys())
        self.vocab = vocab
        self.transform = transform

    def __getitem__(self, index):
        ann_id = self.ids[index]
        cap = self.coco.anns[ann_id]["caption"]
        id = self.coco.anns[ann_id]["image_id"]
        path = self.coco.loadImgs(id)[0]["file_name"]

        path = os.path.join(self.root, path)
        img = Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        toks = word_tokenize(str(cap).lower())
        cap = list()
        cap.append(self.vocab("<start>"))
        cap.extend([self.vocab(t) for t in toks])
        cap.append(self.vocab("<end>"))

        return img, torch.Tensor(cap)

    def __len__(self):
        return len(self.ids)
