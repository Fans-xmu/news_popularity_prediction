# encoding: utf-8
'''
@author: Fans
@file: data_loader_gnn.py.py
@time: 2021/6/11 10:51
@desc:
'''
#1.读取数据构建example，一个news是一个图，图由word、sentence构成，边由word2sentence或word2word的共现
"""This file contains code to read the train/eval/test data 
from file and process it, and read the vocab data from file and process it"""
from collections import Counter
import re
import glob
import copy
import random
import time
from itertools import combinations
from random import shuffle
import pickle
import os
from itertools import combinations
import numpy as np
from random import shuffle
from gensim.models import Word2Vec
import torch
import torch.utils.data
import torch.nn.functional as F
from tools.logger import *
from module.vocabulary import Vocab
from dgl.data.utils import save_graphs, load_graphs

class Examplefhgnn(object):  # 一个新闻--->若干句子&词--->构图，node-word&sentence，边word2sent，sent2word
    # 一条新闻一个图
    """Class representing a train/val/test example for news graph encoding."""
    def __init__(self, news, vocab, sent_max_len, label, time_list):
        """ Initializes the Example, performing tokenization and truncation to produce the encoder, decoder and target sequences, which are stored in self.
        #news 一个新闻文本，包含若干句子
        :param article_sents: list(strings) for single document or list(list(string)) for multi-document; one per article sentence. each token is separated by a single space.
        :param vocab: Vocabulary object
        :param sent_max_len: int, max length of each sentence
        :param label: float, the popularity of this news
        """
        self.sent_max_len = sent_max_len
        self.enc_sent_len = []
        self.enc_sent_input = []
        self.enc_sent_input_pad = []

        # Store the original strings
        self.original_article_sents = news

        # Process the mews
        for sent in self.original_article_sents:
            article_words = sent
            self.enc_sent_len.append(len(article_words))  # store the length before padding
            self.enc_sent_input.append([vocab.word2id(w) for w in
                                        article_words])  # list of word ids; OOVs are represented by the id for UNK token

        self._pad_encoder_input(vocab.word2id('[PAD]'))  # pad操作
        # Store the label
        self.label = label  # 值
        self.time_list = time_list  # time

    def _pad_encoder_input(self, pad_id):
        """
        :param pad_id: int; token pad id
        :return:
        """
        max_len = self.sent_max_len
        for i in range(len(self.enc_sent_input)):
            article_words = self.enc_sent_input[i].copy()
            if len(article_words) > max_len:
                article_words = article_words[:max_len]
            if len(article_words) < max_len:
                article_words.extend([pad_id] * (max_len - len(article_words)))
            self.enc_sent_input_pad.append(article_words)


######################################### ExampleSet #########################################

class ExamplefhgnnSet(torch.utils.data.Dataset):
    """ Constructor: Dataset of example(object) for single news popularity prediction"""

    def __init__(self, data_file, vocab, news_max_timesteps, sent_max_len, w2s_path):
        """ Initializes the ExampleSet with the path of data
        :param data_file:string; the name of pkl data ，str类型
        :param vocab: object;
        :param news_max_timesteps: int; the maximum sentence number of a news, each example should pad sentences to this length
        :param sent_max_len: int; the maximum token number of a sentence, each sentence should pad tokens to this length
        :param w2s_path: str; file path, each line in the file contain a json format data (which can refer to the format can refer to script/5_calw2sTFIDF.py)
        """
        self.vocab = vocab
        self.sent_max_len = sent_max_len
        self.news_max_timesteps = news_max_timesteps

        logger.info("[INFO] Start reading %s", self.__class__.__name__)
        start = time.time()

        with open(data_file, 'rb') as f:
            self.sum_news_list, self.sum_labels, self.time_lists = pickle.load(f)  # 数据集
        logger.info("[INFO] train_x, train_y,train_t读取成功！")
        #change time
        self.example_list = readexamplelist(self.sum_news_list, self.sum_labels, self.news_max_timesteps,
                                            self.time_lists)
        # 返回的是划分后的所有的example_list，每一个元素指的是一个字典，字典内是50条新闻的内容列表，label列表和时间列表

        logger.info("[INFO] Finish reading %s. Total time is %f, Total size is %d", self.__class__.__name__,
                    time.time() - start, len(self.example_list))

        self.size = len(self.example_list)
        logger.info("[INFO] Loading word2sent TFIDF file from %s!" % w2s_path)
        self.w2s_tfidf = readpkl(w2s_path)


    def get_example(self, index):
        e = self.example_list[index]
        news = e['text']
        label = e['label']
        time_list = e['time']
        example = Examplefhgnn(news, self.vocab, self.sent_max_len, label, time_list)
        return example

    def pad_label_m(self, label_matrix):
        label_m = label_matrix[:self.doc_max_timesteps, :self.doc_max_timesteps]
        N, m = label_m.shape
        if m < self.doc_max_timesteps:
            pad_m = np.zeros((N, self.doc_max_timesteps - m))
            return np.hstack([label_m, pad_m])
        return label_m

    def AddWordNode(self, G, inputid):
        wid2nid = {}
        nid2wid = {}
        nid = 0
        for sentid in inputid:
            for wid in sentid:
                if wid not in wid2nid.keys():
                    wid2nid[wid] = nid
                    nid2wid[nid] = wid
                    nid += 1

        w_nodes = len(nid2wid)

        G.add_nodes(w_nodes)
        G.set_n_initializer(dgl.init.zero_initializer)
        G.ndata["unit"] = torch.zeros(w_nodes)
        G.ndata["id"] = torch.LongTensor(list(nid2wid.values()))
        G.ndata["dtype"] = torch.zeros(w_nodes)
        return wid2nid, nid2wid

    def catWordNode(self, inputid):
        wid2nid = {}
        nid2wid = {}
        nid = 0
        for sentid in inputid:
            for wid in sentid:
                if wid not in wid2nid.keys():
                    wid2nid[wid] = nid
                    nid2wid[nid] = wid
                    nid += 1
        return wid2nid, nid2wid

    def get_entity_dict(self, inputid, entity_list, vocab):
        entity_dict = []
        for sentid in inputid:
            for wid in sentid:
                if vocab.id2word(wid) in entity_list:
                    entity_dict.append(wid)
        return entity_dict

    def Create_contentGraph(self, input_pad, w2s_w , label,time_list,index):
        G = dgl.DGLGraph()  # content graph
        wid2nid, nid2wid = self.AddWordNode(G, input_pad)
        w_nodes = len(nid2wid)
        # print(wid2nid,nid2wid)
        N = len(input_pad)
        word_list=[]
        for i in range(N):
            for j in range(len(input_pad[i])):
                if input_pad[i][j] not in word_list:
                    word_list.append(input_pad[i][j])
        if len(word_list)<50:
            for i in range(50-len(word_list)):
                word_list.append(0)
        #保存单个新闻node的所有word
        G.add_nodes(N)
        G.ndata["unit"][w_nodes:] = torch.ones(N)
        G.ndata["dtype"][w_nodes:] = torch.ones(N)
        sentid2nid = [i + w_nodes for i in range(N)]
        #creat doc node
        G.add_nodes(1)
        docid = w_nodes + N
        G.ndata["unit"][docid] = torch.ones(1) * 2
        G.ndata["dtype"][docid] = torch.ones(1) * 2
        G.set_e_initializer(dgl.init.zero_initializer)

        for i in range(N):
            c = Counter(input_pad[i])
            sent_nid = sentid2nid[i]
            sent_tfw = w2s_w[i]
            for wid in c.keys():
                if wid in wid2nid.keys() and self.vocab.id2word(wid) in sent_tfw.keys():
                    tfidf = sent_tfw[self.vocab.id2word(wid)]
                    tfidf_box = np.round(tfidf * 9)  # box = 10
                    G.add_edges(wid2nid[wid], sent_nid,
                                data={"tffrac": torch.LongTensor([tfidf_box]), "dtype": torch.Tensor([0])})
                    G.add_edges(sent_nid, wid2nid[wid],
                                data={"tffrac": torch.LongTensor([tfidf_box]), "dtype": torch.Tensor([0])})

        G.nodes[sentid2nid].data["words"] = torch.LongTensor(input_pad)  # [N, seq_len]
        G.nodes[sentid2nid].data["position"] = torch.arange(1, N + 1).view(-1, 1).long()  # [N, 1]

        #add edge
        for i in range(N):
            sent_nid = sentid2nid[i]
            G.add_edges(docid,sent_nid,
                        data={"tffrac": torch.LongTensor([0]), "dtype": torch.Tensor([1])})
            G.add_edges(sent_nid, docid,
                        data={"tffrac": torch.LongTensor([0]), "dtype": torch.Tensor([1])})

        G.nodes[docid].data["id"] = torch.LongTensor([index+1])
        G.nodes[docid].data["label"] = torch.FloatTensor([label])  # [1]
        G.nodes[docid].data["time"] = torch.LongTensor([time_list])
        G.nodes[docid].data["word"]=torch.LongTensor([word_list])
        return G

    def __getitem__(self, index):
        """
        :param index: int; the index of the example
        :return
            G: graph for the example
            index: int; the index of the example in the dataset
        """
        item = self.get_example(index)
        # logger.info("[INFO] Loading now example successfully!")
        input_pad = item.enc_sent_input_pad
        label = item.label
        time_list = item.time_list
        w2s_w = self.w2s_tfidf[index]#可以准确得到对应的tfidf

        G = self.Create_contentGraph(input_pad, w2s_w, label,time_list,index)

        return G,index

    def __len__(self):
        return self.size



######################################### Tools #########################################


import dgl


def takeSecond(elem):
    return elem[1]


def catDoc(textlist):
    res = []
    for tlist in textlist:
        res.extend(tlist)
    return res



def readexamplelist(sum_news, labels, doc_max_timesteps, time_lists):
    example_list = []
    for i in range(len(sum_news)):
        data=sum_news[i]
        if len(data)<doc_max_timesteps:
            m=doc_max_timesteps-len(data)
            #for i in range(m):
                #data.append([])
        else:
            data=data[0:doc_max_timesteps]
        example_dict = {}
        example_dict['text'] = data
        example_dict['label'] = labels[i]
        example_dict['time'] = time_lists[i]
        example_list.append(example_dict)
    return example_list


def readjson(filename):
    with open(filename, encoding="utf-8") as f:
        tfidfvector = json.load(f)
        return tfidfvector

def readpkl(filename):
    with open(filename, 'rb') as f:
        wsw_lists = pickle.load(f)  # 数据集
    return wsw_lists

def readText(fname):
    data = []
    with open(fname, encoding="utf-8") as f:
        for line in f:
            data.append(line.strip())
    return data


def extractwstfidf(tfidfvector, doc_max_timesteps):
    id_list = list(tfidfvector.keys())
    num = len(id_list)
    count = 0
    alltfidf = []
    tfidf = []
    for i in id_list:
        count += 1
        tfidf.append(tfidfvector[i])
        if count % doc_max_timesteps == 0:
            alltfidf.append(tfidf)
            tfidf = []
    return alltfidf
def graph_collate_fn(samples):
    '''
    :param batch: (G, input_pad)
    :return:
    '''
    graphs, index= map(list, zip(*samples))
    graph_len = [len(g.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)) for g in graphs]  # sent node of graph
    sorted_len, sorted_index = torch.sort(torch.LongTensor(graph_len), dim=0, descending=True)
    batched_graph = dgl.batch([graphs[idx] for idx in sorted_index])
    return batched_graph, [index[idx] for idx in sorted_index]
def creat_entitymap(G,vocab,entity_dict,entseqlen):
    Nnode_id = G.filter_nodes(lambda nodes: nodes.data["dtype"] == 2)
    entity_map = []
    newsid2nid={}
    for i in list(entity_dict.keys()):
        entity_map.append([])
    for idx in Nnode_id:
        word_list=G.nodes[idx].data["word"].squeeze().tolist()
        word_list = list(set(word_list))
        for word in word_list:
            if vocab.id2word(word) in list(entity_dict.keys()):
                news_id=int(G.nodes[idx].data["id"].item())
                newsid2nid[news_id]=idx
                entity_map[entity_dict[vocab.id2word(word)]].append(news_id)
    entity_map_pad=[]
    for i in range(len(entity_map)):
        seq_words = entity_map[i].copy()
        seqtimelist=[G.nodes[newsid2nid[ide]].data["time"].item() for ide in seq_words]
        data = list(zip(seq_words, seqtimelist))

        data.sort(key=takeSecond)
        seq_words=[]
        for seq,times in data:
            seq_words.append(seq)
        if entity_map[i]!=[]:
            if len(seq_words) > entseqlen:

                seq_words = seq_words[:entseqlen]
            if len(seq_words) < entseqlen:
                seq_words.extend([0] * (entseqlen - len(seq_words)))
        else:
            seq_words.extend([0] * entseqlen)
        entity_map_pad.append(seq_words)
    return entity_map_pad
def takeSecond(elem):
    return elem[1]

