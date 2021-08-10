# encoding: utf-8
'''
@author: Fans
@file: FHgnn.py
@time: 2021/6/17 13:13
@desc:
'''
import numpy as np
import torch
import dgl
from module.Attention import Attention_news
from module.Encoder import sentEncoder
from module.GAT import WSWGAT
from module.PositionEmbedding import get_sinusoid_encoding_table
import torch.nn as nn
import torch.nn.utils.rnn as rnn

#gru seq
class GRUModel(nn.Module):
    def __init__(self, input_num, hidden_num, output_num):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_num
        self.GRU_layer = nn.GRU(input_size=input_num, hidden_size=hidden_num, batch_first=True)
        self.output_linear = nn.Linear(hidden_num, output_num)
        self.hidden = None

    def forward(self, x):
        x, self.hidden = self.GRU_layer(x)
        x = self.output_linear(x)
        return x, self.hidden


class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size, bias=False)
        self.sig = torch.sigmoid
        self.linear2 = nn.Linear(hidden_size, output_size, bias=False)

    def forward(self, x):
        y_pred = self.linear2(self.sig(self.linear1(x)))
        return y_pred


class FHgnn(nn.Module):
    """ GAT------>word2sent,sent2word-->sentence-->news embedding-->entity sequence gru-->attention-->prediction"""
    def __init__(self, hps, embed, entity_news_dict):
        """
        :param hps:
        :param embed: word embedding
        :param entity_news_dict: news occur in entity sequence dict
        """
        super().__init__()

        self._hps = hps
        self._n_iter = hps.n_iter# graph 迭代次数
        self._embed = embed
        self._entity_news_dict=entity_news_dict
        self.embed_size = hps.word_emb_dim
        #128

        # sent node feature
        self._init_sn_param()
        # 50
        self._TFembed = nn.Embedding(10, hps.feat_embed_size)  # box=10
        self.n_feature_proj = nn.Linear(hps.n_feature_size, hps.hidden_size, bias=False)
        # 128-->64
        # word -> sent
        embed_size = hps.word_emb_dim
        # news graph encoding module
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
        #64->50
        self.sent2news = WSWGAT(in_dim=hps.hidden_size,#64
                                out_dim=hps.feat_embed_size,#50
                                num_heads=8,  # mult-head attention
                                attn_drop_out=hps.atten_dropout_prob,
                                ffn_inner_hidden_size=hps.ffn_inner_hidden_size,
                                ffn_drop_out=hps.ffn_dropout_prob,
                                feat_embed_size=hps.feat_embed_size,
                                layerType="S2N"
                                )

        # node classification
        self.n_feature = hps.hidden_size
        self.feat_embed_size=hps.feat_embed_size

        #sequence gru module
        self.gru = GRUModel(self.feat_embed_size, self.feat_embed_size * 2,self.feat_embed_size)
        #MLP prediction module
        self.pred = MLP(self.feat_embed_size,self.feat_embed_size*2 ,1)

    def forward(self, graph , entity_map):
        """
        :param graph: [batch_size] * DGLGraph
            node:
                word: unit=0, dtype=0, id=(int)wordid in vocab
                sentence: unit=1, dtype=1, words=tensor, position=int, label=tensor
            edge:
                word2sent, sent2word:  tffrac=int, type=0
            news:
                sent2news: avg pooling,maxpooling,no merge,attention, entity attention
            entity_map:[entity_num,news seqlen]
        :return: result: [newsnum, 1]
        """

        # word node init
        word_feature = self.set_wnfeature(graph)  # [wnode, embed_size]
        sent_feature = self.n_feature_proj(self.set_snfeature(graph))  # [snode, n_feature_size]

        # the start state------>content graph
        word_state = word_feature
        sent_state = self.word2sent(graph, word_feature, sent_feature)

        for i in range(self._n_iter):
            # sent -> word
            word_state = self.sent2word(graph, word_state, sent_state)
            # word -> sent
            sent_state = self.word2sent(graph, word_state, sent_state)
        Nnode_id = graph.filter_nodes(lambda nodes: nodes.data["dtype"] == 2)
        news_state=torch.zeros(len(Nnode_id),self.feat_embed_size).to(torch.device("cuda"))

        news_embedding=self.sent2news(graph,sent_state,news_state)
        graph.nodes[Nnode_id].data["news_embedding"] = news_embedding
        newsid2nid,nid2newsid=self.news_id2nid(graph)

        entity_seq=self.newsid2embed(graph, entity_map, newsid2nid)
        entity_seq=entity_seq.to(torch.device("cuda"))
        entity_result,_=self.gru(entity_seq)

        '''
        news feature merge process
        no-merge,max pooling, pooling,attention, entity attention
        '''
        # average pooling
        news_embedding=self.en2news_avgpooling(graph, entity_result,
                                                self._entity_news_dict,entity_map,newsid2nid)
        # no merge
        # news_embedding=self.en2news_one(graph, entity_result,
        #                                self._entity_news_dict,entity_map,newsid2nid)
        # max pooling
        # news_embedding = self.en2news_maxpooling(graph, entity_result,
        #                                          self._entity_news_dict, entity_map, newsid2nid)
        # attention
        # news_embedding = self.en2news_attention(graph, entity_result,
        #                                          self._entity_news_dict, entity_map, newsid2nid)

        result = self.pred(news_embedding)
        return result
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
        self.attention_news=Attention_news(self._hps)
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
        # sentence encoding
        # 1.CNN
        ngram_feature = self.ngram_enc.forward(graph.nodes[snode_id].data["words"])  # [snode, embed_size]

        graph.nodes[snode_id].data["sent_embedding"] = ngram_feature
        features, glen = get_snode_feat(graph, feat="sent_embedding")
        # sentence encoding
        # 2.BiLSTM
        lstm_feature = self._sent_lstm_feature(features, glen)
        # node_feature = torch.cat([cnn_feature, lstm_feature], dim=1)  # [n_nodes, n_feature_size * 2]
        return lstm_feature

    def entityprofile(self,entity_new_dict):
        entity_idlist = []
        for ids in list(entity_new_dict.keys()):
            entity_idlist.append(entity_new_dict[ids])
        ##entityidlist,可以构建profile//pretrainembed,merge,concat
        entity_profile = self._embed(entity_idlist)
        return entity_profile
    def en2news_maxpooling(self,graph,entity_result,_entity_news_dict,entity_map,newsid2nid):
        entity_result_list=entity_result.tolist()
        entity_map_list = entity_map.tolist()
        ##此处可拓展为attention
        news_embedding_list=[]
        Nnode_id = graph.filter_nodes(lambda nodes: nodes.data["dtype"] == 2)
        Nnode_id_list=Nnode_id.tolist()
        for id in Nnode_id:
            news_embedding_list.append([])
        #构建newsembedding profile
        for i in range(len(entity_map_list)):
            for j in range(len(entity_map_list[i])):
                if entity_map_list[i][j]!=0:
                    nid=newsid2nid[entity_map_list[i][j]]
                    oid=Nnode_id_list.index(nid)
                    news_embedding_list[oid].append(entity_result_list[i][j])
        for i in range(len(news_embedding_list)):
            if len(news_embedding_list[i])!=0:
                nid=Nnode_id_list[i]
                narray=np.array(news_embedding_list[i])
                maxpool=np.max(narray, axis=0)
                graph.nodes[nid].data["news_embedding"] = torch.FloatTensor([maxpool]).to(torch.device("cuda"))

        news_embedding=graph.nodes[Nnode_id].data["news_embedding"]
        return news_embedding

    def en2news_avgpooling(self,graph,entity_result,_entity_news_dict,entity_map,newsid2nid):
        entity_result_list=entity_result.tolist()
        entity_map_list = entity_map.tolist()
        ##此处可拓展为attention
        news_embedding_list=[]
        Nnode_id = graph.filter_nodes(lambda nodes: nodes.data["dtype"] == 2)
        Nnode_id_list=Nnode_id.tolist()
        for id in Nnode_id:
            news_embedding_list.append([])
        #构建newsembedding profile
        for i in range(len(entity_map_list)):
            for j in range(len(entity_map_list[i])):
                if entity_map_list[i][j]!=0:
                    nid=newsid2nid[entity_map_list[i][j]]
                    oid=Nnode_id_list.index(nid)
                    news_embedding_list[oid].append(entity_result_list[i][j])
        for i in range(len(news_embedding_list)):
            if len(news_embedding_list[i])!=0:
                nid=Nnode_id_list[i]
                narray=np.array(news_embedding_list[i])
                avgpool=narray.mean(axis=0)
                graph.nodes[nid].data["news_embedding"] = torch.FloatTensor([avgpool]).to(torch.device("cuda"))

        news_embedding=graph.nodes[Nnode_id].data["news_embedding"]
        return news_embedding

    def en2news_attention(self,graph,entity_result,_entity_news_dict,entity_map,newsid2nid):
        entity_result_list=entity_result.tolist()
        entity_map_list = entity_map.tolist()
        ##此处可拓展为attention
        news_embedding_list=[]
        Nnode_id = graph.filter_nodes(lambda nodes: nodes.data["dtype"] == 2)
        Nnode_id_list = Nnode_id.tolist()
        for id in Nnode_id:
            news_embedding_list.append([])
        #构建newsembedding profile
        mask_noid=[0]*len(Nnode_id_list)
        for i in range(len(entity_map_list)):
            for j in range(len(entity_map_list[i])):
                if entity_map_list[i][j]!=0:
                    nid=newsid2nid[entity_map_list[i][j]]
                    oid=Nnode_id_list.index(nid)
                    mask_noid[oid]=1
                    news_embedding_list[oid].append(entity_result_list[i][j])

        news_embedding_list_new=pad_seq(news_embedding_list,self.feat_embed_size)
        news_embedding_list_new=torch.Tensor(news_embedding_list_new)
        news_attention_embed=self.attention_news.forward(news_embedding_list_new)
        #attention完
        news_attention_list=news_attention_embed.tolist()
        for i in range(len(Nnode_id)):
            if mask_noid[i]!=0:
                nid=Nnode_id_list[i]
                attention_newsiemb=np.array(news_attention_list[i])
                atensor=torch.FloatTensor([attention_newsiemb],device="cpu")
                atensor=atensor.to(torch.device("cuda:0"))
                graph.nodes[nid].data["news_embedding"] = atensor

        news_embedding=graph.nodes[Nnode_id].data["news_embedding"]
        return news_embedding

    def en2news_one(self,graph,entity_result,_entity_news_dict,entity_map,newsid2nid):
        entity_result_list=entity_result.tolist()
        entity_map_list = entity_map.tolist()
        ##此处可拓展为attention
        for i in range(len(entity_map_list)):
            for j in range(len(entity_map_list[i])):
                if entity_map_list[i][j]!=0:
                    nid=newsid2nid[entity_map_list[i][j]]
                    ten=torch.FloatTensor([entity_result_list[i][j]])
                    graph.nodes[nid].data["news_embedding"] = torch.FloatTensor([entity_result_list[i][j]]).to(torch.device("cuda"))
        Nnode_id = graph.filter_nodes(lambda nodes: nodes.data["dtype"] == 2)
        news_embedding=graph.nodes[Nnode_id].data["news_embedding"]
        return news_embedding
    def news_id2nid(self,graph):
        newsid2nid={}
        nid2newsid={}
        Nnode_id = graph.filter_nodes(lambda nodes: nodes.data["dtype"] == 2)
        for idx in Nnode_id:
            news_id = graph.nodes[idx].data["id"].item()
            newsid2nid[news_id]=idx
            nid2newsid[idx]=news_id
        return newsid2nid,nid2newsid

    def newsid2embed(self,graph,entity_map,newsid2nid):
        entity_map_list = entity_map.tolist()
        for i in range(len(entity_map_list)):
            for j in range(len(entity_map_list[i])):
                if entity_map_list[i][j]==0:
                    entity_map_list[i][j]=torch.zeros(self.feat_embed_size).tolist()
                else:
                    news_id=entity_map_list[i][j]
                    #entity_map[i][j] = torch.zeros(self.feat_embed_size).tolist()
                    entity_map_list[i][j]=graph.nodes[newsid2nid[news_id]].data["news_embedding"].squeeze().tolist()
        return torch.Tensor(entity_map_list)
    def _addsqu(self, embed):
        results = embed.unsqueeze(1)
        return results

    def _delsqu(self, embed):
        results = embed.squeeze(1)
        return results




def pad_seq(news_embedding_list,embed_size):
    news_embedding_list_new = news_embedding_list
    max_len=0
    for i in range(len(news_embedding_list_new)):
        if len(news_embedding_list_new[i]) >max_len:
            max_len=len(news_embedding_list_new[i])
    fea=embed_size*[0]
    for i in range(len(news_embedding_list_new)):
        if len(news_embedding_list_new[i]) < max_len:
            pad_len=max_len-len(news_embedding_list_new[i])
            news_embedding_list_new[i].extend([fea]*pad_len)
    return news_embedding_list_new #浅拷贝

def get_snode_feat(G, feat):
    glist = dgl.unbatch(G)
    feature = []
    glen = []
    for g in glist:
        snode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 1)
        feature.append(g.nodes[snode_id].data[feat])
        glen.append(len(snode_id))
    return feature, glen