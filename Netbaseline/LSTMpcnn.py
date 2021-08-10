# encoding: utf-8
'''
@author: Fans
@file: LSTMpcnn.py
@time: 2021/5/9 12:51
@desc:
'''
import numpy as np

import torch
import torch

import dgl

# from module.GAT import GAT, GAT_ffn
from module.Encoder import sentEncoder
from module.GAT import WSWGAT
from module.PositionEmbedding import get_sinusoid_encoding_table
import torch.nn as nn
import torch.nn.utils.rnn as rnn

from module.Encoder_pcnn import sentEncoder_pcnn
from module.GAT import WSWGAT
from module.PositionEmbedding import get_sinusoid_encoding_table
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn

class LSTMpcnn(nn.Module):
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

        self.sent_pos_embed = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(self._hps.doc_max_timesteps + 1, self.embed_size, padding_idx=0),
            freeze=True)
        self.cnn_proj = nn.Linear(self.embed_size, self._hps.n_feature_size)
        self.lstm_hidden_state = self._hps.lstm_hidden_state
        self.lstm = nn.LSTM(self.embed_size, self.lstm_hidden_state, num_layers=self._hps.lstm_layers, dropout=0.1,
                            batch_first=True, bidirectional=self._hps.bidirectional)
        if self._hps.bidirectional:
            self.lstm_proj = nn.Linear(self.lstm_hidden_state * 2, self._hps.n_feature_size)
        else:
            self.lstm_proj = nn.Linear(self.lstm_hidden_state, self._hps.n_feature_size)

        self.sentencode = sentEncoder_pcnn(self._hps, self._embed)
        self.n_feature_proj = nn.Linear(hps.n_feature_size, hps.hidden_size, bias=False)

        # word -> sent
        embed_size = hps.word_emb_dim
        self.n_feature = hps.hidden_size
        # 改成预测数值
        self.wh = nn.Linear(self.n_feature, 1)

    def forward(self, news_list):
        # word embedding intro
        # sent encode
        ngram_feature = self.sentencode.forward(news_list)  # [snode, embed_size]
        #LSTM
        lstm_feature = self._sent_lstm_feature(ngram_feature)
        sent_feature = self.n_feature_proj(lstm_feature)  # [snode, n_feature_size]
        result = self.wh(sent_feature)
        return result

    def _addsqu(self, embed):
        results = embed.unsqueeze(1)
        return results

    def _delsqu(self, embed):
        results = embed.squeeze(1)
        return results


    def _sent_cnn_feature(self, graph, snode_id):
        ngram_feature = self.ngram_enc.forward(graph.nodes[snode_id].data["words"])  # [snode, embed_size]
        graph.nodes[snode_id].data["sent_embedding"] = ngram_feature
        snode_pos = graph.nodes[snode_id].data["position"].view(-1)  # [n_nodes]
        position_embedding = self.sent_pos_embed(snode_pos)
        cnn_feature = self.cnn_proj(ngram_feature + position_embedding)
        return cnn_feature

    def _sent_lstm_feature(self, features):
        pad_seq = rnn.pad_sequence(features, batch_first=True)
        lstm_input = self._addsqu(pad_seq)
        lstm_output, _ = self.lstm(lstm_input)
        lstm_embedding = self._delsqu(lstm_output)
        lstm_feature = self.lstm_proj(lstm_embedding)  # [n_nodes, n_feature_size]
        return lstm_feature
