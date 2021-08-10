# encoding: utf-8
'''
@author: Fans
@file: MLP_CNN.py
@time: 2021/7/2 16:34
@desc:
'''

import torch
from module.Encoder import sentEncoder
from module.GAT import WSWGAT
from module.PositionEmbedding import get_sinusoid_encoding_table
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
# network build
class MLP(nn.Module):
    def __init__(self, input_size, common_size):
        super(MLP, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.ReLU(inplace=True),
            nn.Linear(input_size // 2, input_size // 4),
            nn.ReLU(inplace=True),
            nn.Linear(input_size // 4, common_size)
        )
    def forward(self, x):
        out = self.linear(x)
        return out

class MLPCNN(nn.Module):
    def __init__(self, hps, embed):
        """
        :param hps:
        :param embed: word embedding
        """
        super().__init__()

        self._hps = hps
        self._n_iter = hps.n_iter
        self._embed = embed
        self.embed_size = hps.word_emb_dim
        self.cnn_proj = nn.Linear(self.embed_size, self._hps.n_feature_size)
        self.sentencode = sentEncoder(self._hps, self._embed)
        self.n_feature_proj = nn.Linear(hps.n_feature_size, hps.hidden_size, bias=False)

        # word -> sent
        embed_size = hps.word_emb_dim
        self.n_feature = hps.hidden_size
        # 改成预测数值
        self.mlp=MLP(self.embed_size,1)

    def forward(self, news_list):
        # word embedding intro
        # sent encode
        sent_feature = self.sentencode.forward(news_list)  # [snode, embed_size]
        #MLP                                              # [snode, n_feature_size]
        result = self.mlp(sent_feature)
        return result
