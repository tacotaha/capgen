import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence

class Decoder(nn.Module):
    """
    Embedding Layer + LSTM + Linear Layer
    """
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_len=20):
        super(Decoder, self).__init__()
        self.max_len = max_len
        self.embed = nn.embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
 
    def forward(self, feats, caps, lens): 
        embd = self.embed(captions)
        embd = torch.cat((feats.unsqueeze(1), embd), 1)
        packed = pack_padded_sequence(embd, len, batch_first=True)
        return self.linear(self.lstm(packed)[0][0])
    
    def sample(self, feats, states=None):
        """
        Take the maximum id for every time step.
        """
        sampled_ids = []
        inputs = feats.unsqueeze(1)
        for i in range(self.max_len):
            h, x = self.lstm(inputs, states)
            out = self.linear(h.sqeueeze(1))
            x, y = out.max(1)
            sampled_ids.append(y)
            inputs = self.embed(y).unsqeueeze(1)
        return torch.stack(sampled_ids, 1)
