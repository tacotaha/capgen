#!/usr/bin/env python

import os
import time
import numpy as np
import tensorflow as tf

from filepaths import *
from dataset import Dataset
from encoder import Encoder
from decoder import Decoder
from sklearn.model_selection import train_test_split
from tensorflow.nn import sparse_softmax_cross_entropy_with_logits as criterion

def check_device():
    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())

def loss_func(actual, pred):
    mask = 1 - np.equal(actual, 0)
    return tf.reduce_mean(criterion(labels=actual, logits=pred) * mask)

def main():
    units = 512
    epochs = 20
    batch_size = 1 #64
    num_caps = 40000
    buffer_size = 1000
    embedding_dim = 256 

    tf.enable_eager_execution()
    fname = os.path.join(CAP_DIR, "captions_train2017.json")
    
    training_data = Dataset(fname, num_caps=num_caps)
    
    img_train, img_test, cap_train, cap_test =\
    train_test_split(training_data.images, training_data.cap_toks, test_size=0.3) 

    data = training_data.load_features(img_train, cap_train)
    data = data.shuffle(buffer_size).batch(batch_size).prefetch(1)

    voc_len = len(training_data.tokenizer.word_index)
    num_feats, num_caps = (2048, 64)

    encoder = Encoder()
    decoder = Decoder(voc_len)
    optimizer = tf.train.AdamOptimizer()

    losses = []
    for epoch in range(epochs):
        start = time.time()
        total_loss = 0
        for (batch, (img, target)) in enumerate(data):
            loss = 0
            h = decoder.init_hidden_layer(batch_size=target.shape[0])
            start_dim = [training_data.tokenizer.word_index[training_data.start_tok]] 
            decoder_input = tf.expand_dims(start_dim * batch_size, 1)
            with tf.GradientTape() as t:
                feats = encoder(img)
                for i in range(1, target.shape[1]):
                    x, h, y = decoder(decoder_input, feats, h)
                    loss += loss_func(target[:,i], x)
                    decoder_input = tf.expand_dims(target[:, i], 1)
                total_loss += (loss / int(target.shape[1]))
                vars = encoder.variables + decoder.variables
                grads = t.gradient(loss, vars)
                optimizer.apply_gradients(zip(grads, vars), tf.train.get_or_create_global_step())
            #print(loss.numpy() / int(target.shape[1]))
            losses.append(total_loss / len(training_data.captions))
        print("EPOCH LOSS = {}".format(total_loss / len(training_data.cap_toks)))
        print("EPOCH TOOK: {}".format(start - time.time()))

if __name__ == "__main__":
#    print("===========CHECKING DEVICE=============")
#    check_device()
#    print("======DONE CHECKING DEVICE=============")
    main()
