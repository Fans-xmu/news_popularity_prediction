# encoding: utf-8
'''
@author: Fans
@file: data_loader_seq.py
own to seq2seq module
@time: 2021/6/9 14:06
@desc:
'''
from module.vocabulary import Vocab
import re
import os
#from nltk.corpus import stopwords
from collections import Counter
import glob
import copy
import random
import time
import json
import pickle

from collections import Counter
import re
import glob
import copy
import random
import time
from itertools import combinations

from random import shuffle
import pickle
import collections
import os
from itertools import combinations
import numpy as np
from random import shuffle
from gensim.models import Word2Vec
import torch
import torch.utils.data
import torch.nn.functional as F

from tools.logger import *

class Exampleseq(object):  # 选出num个按时间排序的新闻,构成一个example
    # num个新闻构成一个图
    # news_max_len是新闻文本中词的序列数量，pad补充
    # news=[list]
    # time_list是新闻的时间=[time]
    # 对组成的example做排序
    def __init__(self, news, vocab, news_max_len, label, time_list):
        self.sent_max_len = news_max_len
        self.enc_sent_len = []
        self.enc_sent_input = []
        self.enc_sent_input_pad = [] #最终生成pad之后的list

        # Store the original strings
        self.original_article_sents = news

        # Process the mews
        for sent in self.original_article_sents:
            article_words = sent
            self.enc_sent_len.append(len(article_words))  # store the length before padding
            self.enc_sent_input.append([vocab.word2id(w) for w in
                                        article_words])
            # list of word ids; OOVs are represented by the id for UNK token

        self._pad_encoder_input(vocab.word2id('[PAD]'))  # pad操作
        # Store the label
        self.label = label  # 热度label值
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

class ExampleseqSet(torch.utils.data.Dataset):

    def __init__(self, data_file, vocab, num_news, sent_max_len):
        """ Initializes the ExampleSet with the path of data
        :param data_file:string; the name of pkl data ，str类型
        :param vocab: object;
        :param num_news: int; the maximum sentence number of a example, each example should pad sentences to this length
        :param sent_max_len: int; the maximum token number of a sentence, each sentence should pad tokens to this length
        :param w2s_path: str; file path, each line in the file contain a json format data (which can refer to the format can refer to script/5_calw2sTFIDF.py)
        """
        self.vocab = vocab
        self.sent_max_len = sent_max_len
        self.num_news = num_news

        logger.info("[INFO] Start reading %s", self.__class__.__name__)
        start = time.time()

        with open(data_file, 'rb') as f:
            self.sum_news_list, self.sum_labels, self.time_lists = pickle.load(f)  # 数据集
        logger.info("[INFO] train_x, train_y,train_t读取成功！")
        self.sum_news_list=example2seq(self.sum_news_list)
        self.example_list = readexamplelist(self.sum_news_list, self.sum_labels, self.num_news,
                                            self.time_lists)
        #此时用字典保存了num_news的一批批的数据
        logger.info("[INFO] Finish reading %s. Total time is %f, Total size is %d", self.__class__.__name__,
                    time.time() - start, len(self.example_list))

        self.size = len(self.example_list)


    def get_example(self, index):
        e = self.example_list[index]
        news = e['text']
        label = e['label']
        time_list = e['time']
        data = list(zip(news, time_list, label))
        data.sort(key=takeSecond)
        news_sort = []
        label_sort = []
        time_list_sort = []
        for new, ntime,lab in data:
            news_sort.append(new)
            label_sort.append(lab)
            time_list_sort.append(ntime)
        example = Exampleseq(news_sort, self.vocab, self.sent_max_len, label_sort, time_list_sort)
        return example  # 一个example是一系列的新闻按时间顺序输入进来

    def __getitem__(self, index):
        """
        :param index: int; the index of the example
        :return
            G: graph for the example
            index: int; the index of the example in the dataset
        """
        item = self.get_example(index)
        news_list=item.enc_sent_input_pad
        label_list=item.label
        time_list=item.time_list
        return news_list,label_list

    def __len__(self):
        return self.size
def example2seq(sum_news):
    news_list=[]
    for news in sum_news:
        news_seq=[]
        for sent in news:
            news_seq.extend(sent)
        news_list.append(news_seq)
    return news_list

def readexamplelist(sum_news, labels, num_news, time_lists):
    n = num_news
    num = int(len(sum_news) // n) * n
    data = [sum_news[i:i + n] for i in range(0, num, n)]
    label = [labels[i:i + n] for i in range(0, num, n)]
    time_list = [time_lists[i:i + n] for i in range(0, num, n)]

    example_list = []
    for i in range(len(data)):
        example_dict = {}
        example_dict['text'] = data[i]
        example_dict['label'] = label[i]
        example_dict['time'] = time_list[i]
        example_list.append(example_dict)
    return example_list

def takeSecond(elem):
    return elem[1]
