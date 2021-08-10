# encoding: utf-8
'''
@author: Fans
@file: FHgnn_noentity.py
@time: 2021/6/17 13:13
@desc:
'''
import numpy as np
import torch

import dgl


from module.Encoder import sentEncoder
from module.GAT import WSWGAT
from module.PositionEmbedding import get_sinusoid_encoding_table
import torch.nn as nn
import torch.nn.utils.rnn as rnn



class FHgnn_allno(nn.Module):
    """ word&sentence encoding---->sentence---->news embedding--->prediction"""

    def __init__(self, hps, embed):
        """
        :param hps:
        :param embed: word embedding
        """
        super().__init__()

        self._hps = hps
        self._n_iter = hps.n_iter# GAT iter num
        self._embed = embed
        self.embed_size = hps.word_emb_dim#128

        # sent node feature
        self._init_sn_param()
        # 50
        self._TFembed = nn.Embedding(10, hps.feat_embed_size)  # box=10
        self.n_feature_proj = nn.Linear(hps.n_feature_size, hps.hidden_size, bias=False)
        # 128-->64
        # word -> sent
        embed_size = hps.word_emb_dim
        # 128----->64
        self.word2sent = WSWGAT(in_dim=embed_size,
                                out_dim=hps.hidden_size,
                                num_heads=hps.n_head,
                                attn_drop_out=hps.atten_dropout_prob,
                                ffn_inner_hidden_size=hps.ffn_inner_hidden_size,
                                ffn_drop_out=hps.ffn_dropout_prob,
                                feat_embed_size=hps.feat_embed_size,
                                layerType="W2S"
                                )
        #64----->128
        # sent -> word
        self.sent2word = WSWGAT(in_dim=hps.hidden_size,
                                out_dim=embed_size,
                                num_heads=8,  # mult-head attention
                                attn_drop_out=hps.atten_dropout_prob,
                                ffn_inner_hidden_size=hps.ffn_inner_hidden_size,
                                ffn_drop_out=hps.ffn_dropout_prob,
                                feat_embed_size=hps.feat_embed_size,
                                layerType="S2W"
                                )

        #sent -> news
        self.sent2news = WSWGAT(in_dim=hps.hidden_size,#64
                                out_dim=hps.feat_embed_size,#50
                                num_heads=8,  # mult-head attention
                                attn_drop_out=hps.atten_dropout_prob,
                                ffn_inner_hidden_size=hps.ffn_inner_hidden_size,
                                ffn_drop_out=hps.ffn_dropout_prob,
                                feat_embed_size=hps.feat_embed_size,
                                layerType="S2N"
                                )
        # word--->entity--->word
        self.word2word = WSWGAT(in_dim=embed_size,
                                out_dim=embed_size,
                                num_heads=8,  # mult-head attention
                                attn_drop_out=hps.atten_dropout_prob,
                                ffn_inner_hidden_size=hps.ffn_inner_hidden_size,
                                ffn_drop_out=hps.ffn_dropout_prob,
                                feat_embed_size=hps.feat_embed_size,
                                layerType="W2W"
                                )
        # node classification
        self.n_feature = hps.hidden_size
        # 改成预测数值
        self.wh = nn.Linear(hps.feat_embed_size, 1)
        #self.gru = GRUModel(self.n_feature, self.n_feature * 2, self.n_feature)

    def forward(self, graph):
        """
        :param graph: [batch_size] * DGLGraph
            node:
                word: unit=0, dtype=0, id=(int)wordid in vocab
                sentence: unit=1, dtype=1, words=tensor, position=int, label=tensor
            edge:
                word2sent, sent2word:  tffrac=int, type=0
        :return: result: [sentnum, 2]
        """

        # word node init
        word_feature = self.set_wnfeature(graph)  # [wnode, embed_size]
        sent_feature = self.n_feature_proj(self.set_snfeature(graph))  # [snode, n_feature_size]

        # the start state------>content graph
        word_state = word_feature
        sent_state = self.word2sent(graph, word_feature, sent_feature)

        # for i in range(self._n_iter):
        #     # sent -> word
        #     word_state = self.sent2word(graph, word_state, sent_state)
        #     # word -> sent
        #     sent_state = self.word2sent(graph, word_state, sent_state)
        Nnode_id = graph.filter_nodes(lambda nodes: nodes.data["dtype"] == 2)
        #print(len(Nnode_id))

        news_state=torch.zeros(len(Nnode_id),50).to(torch.device("cuda"))

        news_embedding=self.sent2news(graph,sent_state,news_state)
        # sent_state = self._addsqu(sent_state)
        # # print(word_state_gcn2.size())
        # sent_state, h = self.gru(sent_state)
        # sent_state = self._delsqu(sent_state)
        result = torch.sigmoid(self.wh(news_embedding))

        return result

    def _addsqu(self, embed):
        results = embed.unsqueeze(1)
        return results

    def _delsqu(self, embed):
        results = embed.squeeze(1)
        return results

    def _init_sn_param(self):
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

        self.ngram_enc = sentEncoder(self._hps, self._embed)

    def _sent_cnn_feature(self, graph, snode_id):
        ngram_feature = self.ngram_enc.forward(graph.nodes[snode_id].data["words"])  # [snode, embed_size]
        graph.nodes[snode_id].data["sent_embedding"] = ngram_feature
        snode_pos = graph.nodes[snode_id].data["position"].view(-1)  # [n_nodes]
        position_embedding = self.sent_pos_embed(snode_pos)
        cnn_feature = self.cnn_proj(ngram_feature + position_embedding)
        return cnn_feature

    def _sent_lstm_feature(self, features, glen):
        pad_seq = rnn.pad_sequence(features, batch_first=True)
        lstm_input = rnn.pack_padded_sequence(pad_seq, glen, batch_first=True)
        lstm_output, _ = self.lstm(lstm_input)
        unpacked, unpacked_len = rnn.pad_packed_sequence(lstm_output, batch_first=True)
        lstm_embedding = [unpacked[i][:unpacked_len[i]] for i in range(len(unpacked))]
        lstm_feature = self.lstm_proj(torch.cat(lstm_embedding, dim=0))  # [n_nodes, n_feature_size]
        return lstm_feature

    def set_wnfeature(self, graph):
        wnode_id = graph.filter_nodes(lambda nodes: nodes.data["unit"] == 0)
        wsedge_id = graph.filter_edges(lambda edges: (edges.data["tffrac"] <= 9) & (edges.data["tffrac"] >= 0))
        # wsedge_id = graph.filter_edges(lambda edges: edges.data["dtype"] ==0)   # for word to supernode(sent&doc)
        wid = graph.nodes[wnode_id].data["id"]  # [n_wnodes]
        w_embed = self._embed(wid)  # [n_wnodes, D]
        graph.nodes[wnode_id].data["embed"] = w_embed
        etf = graph.edges[wsedge_id].data["tffrac"]
        graph.edges[wsedge_id].data["tfidfembed"] = self._TFembed(etf)
        return w_embed

    def set_snfeature(self, graph):
        # node feature
        snode_id = graph.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
        ngram_feature = self.ngram_enc.forward(graph.nodes[snode_id].data["words"])  # [snode, embed_size]

        graph.nodes[snode_id].data["sent_embedding"] = ngram_feature
        features, glen = get_snode_feat(graph, feat="sent_embedding")
        lstm_feature = self._sent_lstm_feature(features, glen)
        # node_feature = torch.cat([cnn_feature, lstm_feature], dim=1)  # [n_nodes, n_feature_size * 2]
        return lstm_feature


def get_snode_feat(G, feat):
    glist = dgl.unbatch(G)
    feature = []
    glen = []
    for g in glist:
        snode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 1)
        feature.append(g.nodes[snode_id].data[feat])
        glen.append(len(snode_id))
    return feature, glen