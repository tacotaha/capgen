#!/usr/bin/env python

import os
import pickle
import argparse
from progress.bar import Bar
from nltk import word_tokenize
from pycocotools.coco import COCO
from collections import Counter, defaultdict

from filepaths import *

class Vocabulary:
    def __init__(self, special_tokens=None):
        self.w2idx = {}
        self.idx2w = {}
        self.w2cnt = defaultdict(int)
        self.special_tokens = special_tokens
        if self.special_tokens is not None:
            self.add_tokens(special_tokens)

    def add_tokens(self, tokens):
        for token in tokens:
            self.add_token(token)
            self.w2cnt[token] += 1

    def add_token(self, token):
        if token not in self.w2idx:
            cur_len = len(self)
            self.w2idx[token] = cur_len
            self.idx2w[cur_len] = token

    def prune(self, min_cnt=2):
        to_remove = set([token for token in self.w2idx if self.w2cnt[token] < min_cnt])
        to_remove ^= set(self.special_tokens)

        for token in to_remove:
            self.w2cnt.pop(token)

        self.w2idx = {token: idx for idx, token in enumerate(self.w2cnt.keys())}
        self.idx2w = {idx: token for token, idx in self.w2idx.items()}

    def __contains__(self, item):
        return item in self.w2idx

    def __getitem__(self, item):
        if isinstance(item, str):
            return self.w2idx[item if item in self.w2idx else "<unk>"]
        elif isinstance(item , int):
            return self.idx2w[item if item in self.idx2w else 3]
        else:
            raise TypeError("Supported indices are int and str")
    
    def __call__(self, item):
        return self.__getitem__(item)

    def __len__(self):
        return(len(self.w2idx))

if __name__ == "__main__":
    out_file = os.path.join(DATA_PATH, "vocab.pkl")

    coco = COCO(CAP_FILE)
    vocab = Vocabulary(["<pad>", "<start>", "<end>", "<unk>"])

    bar = Bar("Extracting Captions", max=len(coco.anns))

    for i, id in enumerate(coco.anns.keys()):
        cap = str(coco.anns[id]["caption"])
        toks = word_tokenize(cap.lower())
        vocab.add_tokens(toks)
        bar.next()

    with open(out_file, "wb") as f:
        pickle.dump(vocab, f)

    bar.finish()
