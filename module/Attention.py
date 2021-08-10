# encoding: utf-8
'''
@author: Fans
@file: Attention.py
@time: 2021/6/29 19:28
@desc:
'''
import torch

import dgl
from module.Encoder import sentEncoder
from module.GAT import WSWGAT
from module.PositionEmbedding import get_sinusoid_encoding_table
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import torch.nn.functional as F

class Attention_news(nn.Module):
    def __init__(self,hps):
        super(Attention_news, self).__init__()
        # embedding之后的shape: torch.Size([newsnum, newsembednum, featuresize])
        self.feature_size=hps.feat_embed_size
        self.w_omega = nn.Parameter(torch.Tensor(
            self.feature_size, self.feature_size))
        self.u_omega = nn.Parameter(torch.Tensor(self.feature_size, 1))

        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)

    def forward(self, inputs):
        x = inputs
        # x形状是(news_num, news_seq_len, feature_size)
        # Attention过程
        u = torch.tanh(torch.matmul(x, self.w_omega))
        # u形状是(news_num, news_seq_len, feature_size)
        att = torch.matmul(u, self.u_omega)
        # att形状是(news_num, news_seq_len, 1)
        att_score = F.softmax(att, dim=1)
        # att_score形状仍为(news_num, news_seq_len, 1)
        scored_x = x * att_score
        # scored_x形状是(news_num, news_seq_len, feature_size)
        # Attention过程结束
        feat = torch.sum(scored_x, dim=1)  # 加权求和
        # feat形状是(news_num, feature_size)
        return feat

