#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: fans
# Time: 2020/4/22 9:43

import numpy as np
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pickle
import os
from tools.logger import *
import logging
from sklearn import svm
import math


# [ 1.5]
def get_x_features(x):
    # BOW feature
    bow_vecs = cnt_vectorizer.transform([' '.join(i) for i in x])
    x_features = bow_vecs.toarray()
    return x_features
def example2seq(sum_news):
    news_list=[]
    for news in sum_news:
        news_seq=[]
        for sent in news:
            news_seq.extend(sent)
        news_list.append(news_seq)
    return news_list
def setup_eval_RMSE(y_predict,y_label):
    RMSE = np.sqrt(np.mean(np.square(y_label - y_predict)))
    logger.info("RMSE:{:.6f}".format(RMSE))

def setup_eval_NDCG(y_predict, y_label,batch_size):
    MRR_loss=0
    NDCG_loss=0
    for i in range(len(y_predict/batch_size)):
        y_pre=y_predict[i*batch_size:i*batch_size+batch_size]
        y_lab=y_label[i*batch_size:i*batch_size+batch_size]

        MRR, NDCG=Seqevalute(y_pre,y_lab)
        print("MRR:",MRR)
        print("NDCG",NDCG)
        MRR_loss += MRR
        NDCG_loss += NDCG
    logger.info("NDCG:{:.3f}".format(NDCG_loss / len(y_predict/batch_size)))
    logger.info("MRR:{:.3f}".format(MRR_loss / len(y_predict/batch_size)))

def Seqevalute(rank_list,label_list):
    idlist=[]
    for i in range(len(rank_list)):
        idlist.append(i)
    rank_list_sort=list(zip(idlist,rank_list))
    label_list_sort=list(zip(idlist,label_list))
    rank_list_sort.sort(key=takeSecond,reverse=True)
    label_list_sort.sort(key=takeSecond,reverse=True)
    rankid_list=[]
    for ids,_ in rank_list_sort:
        rankid_list.append(ids)
    labelid_list = []
    for ids, _ in label_list_sort:
        labelid_list.append(ids)
    MRR = 0
    NDCG=0
    for new in rankid_list:
        distance=0
        for j in labelid_list:
            if j!=new:
                distance+=1
        MRR += 1 / (distance + 1)
        NDCG += 1/(math.log2(1+distance+1))
    MRR/=len(rank_list)
    NDCG/=len(rank_list)
    return MRR,NDCG

def takeSecond(elem):
    return elem[1]

if __name__ == '__main__':
    data_dir='../pkldata/'
    data_file = os.path.join(data_dir, "disa_train_event_data.pkl")
    with open(data_file, 'rb') as f:
        sum_news_list, sum_labels, time_lists = pickle.load(f)  # 数据集
    sum_news_list = example2seq(sum_news_list)
    texts = sum_news_list  # 评论分词后的文本
    labels = sum_labels

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    vocab = tokenizer.word_index

    x_train, x_test, y_train, y_test = train_test_split(texts, labels, test_size=0.1)
    # y_train = np.array([i[1] for i in y_train])
    # y_test = np.array([i[1] for i in y_test])

    # bag of words
    corpus = [' '.join(i) for i in x_train]
    stop_words = [w.strip() for w in open('../pkldata/停用词词典.txt').readlines()]
    cnt_vectorizer = CountVectorizer(max_features=500, stop_words=stop_words)
    cnt_vectorizer.fit(corpus)

    x_train_features = get_x_features(x_train)
    x_test_features = get_x_features(x_test)
    # svc = GridSearchCV(SVC(), param_grid={"kernel": ['rbf'],
    #                                       "C": np.logspace(-3, 3, 7),
    #                                       "gamma": np.logspace(-3, 3, 7)})
    #print(y_train)
    clf = svm.SVR()
    # clf.fit([[1,2],[2,3]], [0.5,0.6])
    # label_test = clf.predict([[2,3],[1,0.6]])

    clf.fit(x_train_features, y_train)
    label_test=list(clf.predict(np.array(x_test)))

    setup_eval_NDCG(label_test, y_test, 256)
    setup_eval_RMSE(label_test, y_test)
