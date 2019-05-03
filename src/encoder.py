#!/usr/bin/env python

import torch
import torch.nn as nn
import torchvision.models as models

class Encoder(torch.nn.Module):
    """
    Feed pre-trained InceptionV3 features through linear layer
    """
    def __init__(self, embedding_dim=256):
        super(Encoder, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embedding_dim)
        self.bn = nn.BatchNorm1d(embedding_dim, momentum=0.01)


    def forward(self, images):
        """
        Forard Pass: Extract Feature vectors from input images
        """
        with torch.no_grad():
            feats = self.resnet(images)
        feats = feats.reshape(feats.size(0), -1)
        return self.bn(self.linear(feats))
