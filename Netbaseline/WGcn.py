# encoding: utf-8
'''
@author: Fans
@file: WGcn.py
@time: 2021/6/20 17:17
@desc:
'''

import torch

import dgl


from module.Encoder import sentEncoder
from module.GAT import WSWGAT
from module.PositionEmbedding import get_sinusoid_encoding_table
import torch.nn as nn
import torch.nn.utils.rnn as rnn


class GRUModel(nn.Module):
    def __init__(self, input_num, hidden_num, output_num):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_num
        # 这里设置了 batch_first=True, 所以应该 inputs = inputs.view(inputs.shape[0], -1, inputs.shape[1])
        # 针对时间序列预测问题，相当于将时间步（seq_len）设置为 1。
        self.GRU_layer = nn.GRU(input_size=input_num, hidden_size=hidden_num, batch_first=True)
        self.output_linear = nn.Linear(hidden_num, output_num)
        self.hidden = None

    def forward(self, x):
        # h_n of shape (num_layers * num_directions, batch, hidden_size)
        # 这里不用显式地传入隐层状态 self.hidden
        x, self.hidden = self.GRU_layer(x)
        x = self.output_linear(x)
        return x, self.hidden
class WGcn(nn.Module):
    """ GCN------>word&entity------->sentence---->news embedding"""

    def __init__(self, hps, embed):
        """
        :param hps:
        :param embed: word embedding
        """
        super().__init__()

        self._hps = hps
        self._n_iter = hps.n_iter# graph 迭代次数
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
        self.feat_embed_size = hps.feat_embed_size
        self.wh = nn.Linear(hps.feat_embed_size, 1)
        self.gru = GRUModel(self.n_feature, self.n_feature * 2, self.n_feature)

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
        snode_id = graph.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
        sent_feature = torch.zeros(len(snode_id),self.n_feature).to(torch.device("cuda"))  # [snode, n_feature_size]

        # the start state------>content graph
        word_state = self.word2word(graph,word_feature,word_feature)
        sent_state = self.word2sent(graph, word_state, sent_feature)

        Nnode_id = graph.filter_nodes(lambda nodes: nodes.data["dtype"] == 2)
        #print(len(Nnode_id))
        news_state=torch.zeros(len(Nnode_id),self.feat_embed_size).to(torch.device("cuda"))
        news_embedding=self.sent2news(graph,sent_state,news_state)

        result = self.wh(news_embedding)

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


    # sentence encode cnn+(lstm cat cnn)
    # def set_snfeature(self, graph):
    #     # node feature
    #     snode_id = graph.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
    #     cnn_feature = self._sent_cnn_feature(graph, snode_id)
    #     features, glen = get_snode_feat(graph, feat="sent_embedding")
    #     lstm_feature = self._sent_lstm_feature(features, glen)
    #     node_feature = torch.cat([cnn_feature, lstm_feature], dim=1)  # [n_nodes, n_feature_size * 2]
    #     return node_feature

def get_snode_feat(G, feat):
    glist = dgl.unbatch(G)
    feature = []
    glen = []
    for g in glist:
        snode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 1)
        feature.append(g.nodes[snode_id].data[feat])
        glen.append(len(snode_id))
    return feature, glen