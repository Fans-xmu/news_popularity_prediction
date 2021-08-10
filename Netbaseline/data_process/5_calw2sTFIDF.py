import os
import argparse
import json
# from tools.logger import *
# import logging
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import pickle
from collections import Counter
import math


# 计算word在当前news内的词频
def tfcaculate(word, news):
    result = Counter(news)
    num = result[word]
    return num/len(news)


# 计算word在全局的idf
def idfcaculate(word, news_list):
    num = 0
    for s in news_list:
        if word in s:
            num += 1
    return math.log(len(news_list) / (num + 1))

    # 按照news id列表创建每一个单词的在当前new下的tfidf


def creattfidf(news_lists):
    tfidf_list=[]
    count = 0
    for news_list in news_lists:
        w2s_w = {}
        countj = 0
        for new in news_list:
            sent_tfw = {}
            for word in new:
                tfidf = tfcaculate(word, new) * idfcaculate(word, news_list)
                sent_tfw[word] = tfidf
            w2s_w[countj] = sent_tfw
            countj+=1
        tfidf_list.append(w2s_w)
        count += 1
        if count % 5 == 0:
            print("step:", count)
    return tfidf_list

if __name__ == '__main__':
    # ---------------------------event chinese dataset process--------------------------------
    data_file = '../pkldata/train_event_data.pkl'
    data_file2 = '../pkldata/valid_event_data.pkl'
    data_file3 = '../pkldata/test_event_data.pkl'
    fname = "train.w2s.tfidf_event.pkl"
    fname2 = "val.w2s.tfidf_event.pkl"
    fname3 = "test.w2s.tfidf_event.pkl"
    save_dir = '../cache'
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    saveFile = os.path.join(save_dir, fname)
    saveFile2 = os.path.join(save_dir, fname2)
    saveFile3 = os.path.join(save_dir, fname3)
    with open(data_file, 'rb') as f:
        train_x, train_y, train_t = pickle.load(f)# train dataset
    logger.info("[INFO]  train_x, train_y ,train_t读取成功！")
    logger.info("[INFO] Save word2sent features of dataset to %s" % saveFile)
    w2s_w = creattfidf(train_x)
    with open(saveFile, 'wb') as f:
        pickle.dump(w2s_w, f, pickle.HIGHEST_PROTOCOL)  # train
    logger.info("[INFO] Save tfidf vec sucessfully!")

    with open(data_file2, 'rb') as f:
        train_x, train_y, train_t = pickle.load(f)# valid dataset
    logger.info("[INFO]  train_x, train_y ,train_t读取成功！")
    logger.info("[INFO] Save word2sent features of dataset to %s" % saveFile)
    w2s_w = creattfidf(train_x)
    with open(saveFile2, 'wb') as f:
        pickle.dump(w2s_w, f, pickle.HIGHEST_PROTOCOL)  # valid
    logger.info("[INFO] Save tfidf vec sucessfully!")

    with open(data_file3, 'rb') as f:
        train_x, train_y, train_t = pickle.load(f)# test dataset
    logger.info("[INFO]  train_x, train_y ,train_t读取成功！")
    logger.info("[INFO] Save word2sent features of dataset to %s" % saveFile)
    w2s_w = creattfidf(train_x)
    with open(saveFile3, 'wb') as f:
        pickle.dump(w2s_w, f, pickle.HIGHEST_PROTOCOL)  # test
    logger.info("[INFO] Save tfidf vec sucessfully!")

    # ---------------------------mind english dataset process--------------------------------
    data_file = '../pkldata3/mind_train_event_data.pkl'
    data_file2 = '../pkldata3/mind_valid_event_data.pkl'
    data_file3 = '../pkldata3/mind_test_event_data.pkl'
    fname = "mind_train.w2s.tfidf_event.pkl"
    fname2 = "mind_val.w2s.tfidf_event.pkl"
    fname3 = "mind_test.w2s.tfidf_event.pkl"
    save_dir = '../cache'
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    saveFile = os.path.join(save_dir, fname)
    saveFile2 = os.path.join(save_dir, fname2)
    saveFile3 = os.path.join(save_dir, fname3)

    with open(data_file, 'rb') as f:
        train_x, train_y, train_t, train_id = pickle.load(f)  # train dataset
    logger.info("[INFO]  train_x, train_y ,train_t读取成功！")
    logger.info("[INFO] Save word2sent features of dataset to %s" % saveFile)
    w2s_w = creattfidf(train_x)
    with open(saveFile, 'wb') as f:
        pickle.dump(w2s_w, f, pickle.HIGHEST_PROTOCOL)  # train
    logger.info("[INFO] Save tfidf vec sucessfully!")

    with open(data_file2, 'rb') as f:
        valid_x, valid_y, valid_t, valid_id = pickle.load(f)  # valid dataset
    logger.info("[INFO]  valid_x,valid_y 读取成功！")
    logger.info("[INFO] Save word2sent features of dataset to %s" % saveFile2)

    w2s_w2 = creattfidf(valid_x)
    with open(saveFile2, 'wb') as f:
        pickle.dump(w2s_w2, f, pickle.HIGHEST_PROTOCOL)  # valid
    logger.info("[INFO] Save tfidf vec sucessfully!")

    with open(data_file3, 'rb') as f:
        test_x, test_y, test_t, test_id = pickle.load(f)  # test dataset
    logger.info("[INFO]  test_x, test_y 读取成功！")
    logger.info("[INFO] Save word2sent features of dataset to %s" % saveFile3)

    w2s_w3 = creattfidf(test_x)
    with open(saveFile3, 'wb') as f:
        pickle.dump(w2s_w3, f, pickle.HIGHEST_PROTOCOL)  # test
    logger.info("[INFO] Save tfidf vec sucessfully!")