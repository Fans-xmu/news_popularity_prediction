from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


from module.PositionEmbedding import get_sinusoid_encoding_table

WORD_PAD = "[PAD]"

#sentence encoding
#1.CNN
class sentEncoder(nn.Module):
    def __init__(self, hps, embed):
        """
        :param hps: 
                word_emb_dim: word embedding dimension
                sent_max_len: max token number in the sentence
                word_embedding: bool, use word embedding or not
                embed_train: bool, whether to train word embedding
                cuda: bool, use cuda or not
        """
        super(sentEncoder, self).__init__()
        self._hps = hps
        self.sent_max_len = hps.sent_max_len
        embed_size = hps.word_emb_dim
        input_channels = 1
        out_channels = 32
        min_kernel_size = 2
        max_kernel_size = 5
        width = embed_size

        # word embedding
        self.embed = embed

        # cnn
        self.convs = nn.ModuleList([nn.Conv2d(input_channels, out_channels, kernel_size=(height, width)) for height in
                                    range(min_kernel_size, max_kernel_size + 1)])
        #self.output = nn.Linear(hps.feat_embed_size * 6, embed_size)
        for conv in self.convs:
            init_weight_value = 4.0
            init.xavier_normal_(conv.weight.data, gain=np.sqrt(init_weight_value))

    def forward(self, input):

        enc_embed_input = self.embed(input)  # [s_nodes, L, D]
        #only cnn
        enc_conv_input = enc_embed_input
        enc_conv_input = enc_conv_input.unsqueeze(1)  # [s_nodes, 1, L, D]

        enc_conv_output = [F.relu(conv(enc_conv_input)).squeeze(3) for conv in
                           self.convs]  # kernel_sizes * [s_nodes, Co=50, W]
        enc_maxpool_output = [F.max_pool1d(x, x.size(2)).squeeze(2) for x in
                              enc_conv_output]  # kernel_sizes * [s_nodes, Co=32]
        sent_embedding = torch.cat(enc_maxpool_output, 1)  # [s_nodes, 32*4]

        return sent_embedding  # [s_nodes, 128]