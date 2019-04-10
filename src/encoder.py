#!/usr/bin/env python


from tensorflow.nn import relu
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense

class Encoder(Model):
    """
    Feed pre-trained InceptionV3 features through linear layer
    """
    def __init__(self, embedding_dim=256):
        super(Encoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.linear = Dense(self.embedding_dim) 

    def call(self, inputs):
        return relu(self.linear(inputs))
