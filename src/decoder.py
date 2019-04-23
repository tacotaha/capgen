import logging
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.nn import tanh, softmax
from tensorflow.test import is_gpu_available
from tensorflow.keras.layers import GRU, Dense, CuDNNGRU, Embedding

class Decoder(Model):
    def __init__(self, voc_len, embedding_dim=256, units=512):
        super(Decoder, self).__init__() 
        self.logger = logging.getLogger("capgen")
        self.units = units       
        self.voc_len = voc_len
        self.embedding_dim = embedding_dim
        self.embedding = Embedding(voc_len, embedding_dim) 
        self.gru = self.get_gru()
        self.linear = tf.keras.layers.Dense(self.units)
        self.out = tf.keras.layers.Dense(self.voc_len)
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)

    def call(self, inputs, features, hidden):
        context, atten = self.attention(features, hidden)
        inputs = self.embedding(inputs)
        inputs = tf.concat([tf.expand_dims(context, 1), inputs], axis=-1)
        out, state = self.gru(inputs)
        inputs = self.linear(out)
        inputs = tf.reshape(inputs, (-1, inputs.shape[2]))
        return self.out(inputs), state, atten
   
    def attention(self, features, hidden):
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        score = tanh(self.W1(features) + self.W2(hidden_with_time_axis))
        attention_weights = softmax(self.V(score), axis=1)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

    def get_gru(self, activation="sigmoid", init="glorot_uniform"):
        if is_gpu_available():
            self.logger.info("[decoder] USING GPU!")
            return CuDNNGRU(self.units, return_sequences=True,
                    return_state=True, recurrent_initializer=init)
        else:
            self.logger.info("[decoder] USING CPU!")
            return GRU(self.units, return_state=True, return_sequences=True, 
                    recurrent_activation=activation, recurrent_initializer=init)

    def init_hidden_layer(self, batch_size):
        return tf.zeros((batch_size, self.units))
