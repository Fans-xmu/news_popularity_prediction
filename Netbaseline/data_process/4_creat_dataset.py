# encoding: utf-8
'''
@author: Fans
@file: 4_creat_dataset.py
@time: 2020/12/15 14:21
@desc: create the train ,valid , test datasets
'''
import os
from openpyxl import load_workbook
import jieba
import random
import pickle
from tools.logger import *
import random
import numpy as np
import math
def getxy(data):
    #输入是一个元组列表
    #输出是两个列表
    x=[]
    y=[]
    for news, label in data:
        x.append(news)
        y.append(label)
    return x,y
def getxyt(data):
    #输入是一个元组列表
    #输出是两个列表
    x=[]
    y=[]
    t=[]
    for news,label,time_new in data:
        x.append(news)
        y.append(label)
        t.append(time_new)
    return x,y,t
def getxytd(data):
    #输入是一个元组列表
    #输出是两个列表
    x=[]
    y=[]
    t=[]
    d=[]
    for news,label,time_new,id in data:
        x.append(news)
        y.append(label)
        t.append(time_new)
        d.append(id)
    return x,y,t,d
def data_extract_seq(word_list,label_pop):
    #输入是一个新闻列表和一个label列表

    data=list(zip(word_list,label_pop))
    random.shuffle(data)
    news_num=len(data)
    #按照8 1 1 的比例划分训练集，验证集，以及测试集
    train_data=data[0:int(news_num*0.8)]
    valid_data=data[int(news_num*0.8):int(news_num*0.9)]
    test_data=data[int(news_num*0.9):int(news_num)]

    train_x,train_y=getxy(train_data)
    valid_x,valid_y=getxy(valid_data)
    test_x,test_y=getxy(test_data)
    return train_x,train_y,valid_x,valid_y,test_x,test_y

def data_extract_event(news_list,label_data,time_list):
    # 输入是一个新闻列表和一个label列表
    data = list(zip(news_list,label_data,time_list))
    random.shuffle(data)
    news_num = len(data)
    # 按照7 2 1 的比例划分训练集，验证集，以及测试集
    train_data = data[0:int(news_num * 0.8)]
    valid_data = data[int(news_num * 0.8):int(news_num * 0.9)]
    test_data = data[int(news_num * 0.9):int(news_num)]

    train_x, train_y, train_t = getxyt(train_data)
    valid_x, valid_y, valid_t = getxyt(valid_data)
    test_x, test_y, test_t = getxyt(test_data)

    return train_x, train_y, train_t, valid_x, valid_y, valid_t, test_x, test_y, test_t

def data_extract_mind(news_list,label_data,time_list,entiid_list):
    # 输入是一个新闻列表和一个label列表
    data = list(zip(news_list,label_data,time_list,entiid_list))
    random.shuffle(data)
    news_num = len(data)
    # 按照7 2 1 的比例划分训练集，验证集，以及测试集
    train_data = data[0:int(news_num * 0.8)]
    valid_data = data[int(news_num * 0.8):int(news_num * 0.9)]
    test_data = data[int(news_num * 0.9):int(news_num)]

    train_x, train_y, train_t ,train_id= getxytd(train_data)
    valid_x, valid_y, valid_t ,valid_id= getxytd(valid_data)
    test_x, test_y, test_t ,test_id= getxytd(test_data)

    return train_x, train_y, train_t ,train_id, valid_x, valid_y, valid_t ,valid_id, test_x, test_y, test_t ,test_id
def maxminnorm(array):
    maxcols=np.nanmax(array)
    mincols=np.nanmin(array)
    print("max,min:")
    print(maxcols,mincols)
    t=np.empty(array.size)
    for i in range(array.size):
        t[i]=(array[i]-mincols)/(maxcols-mincols)
    return t
def changetime_list(time_lists):
    timedict={}
    time2id=time_lists
    time2id.sort()
    for i in range(len(time2id)):
        timedict[time2id[i]]=i+1
    time_fin=[timedict[idx] for idx in time_lists]
    return time_fin

def data_cleangnn(news_list):
    for news in news_list:
        for sent in news:
            if len(sent)==0:
                news.remove(sent)

if __name__ == "__main__":
    # ---------------------------event chinese dataset process--------------------------------
    with open('../pkldata/disa_eventnews_data.pkl', 'rb') as f:
        news_list= pickle.load(f)#对所有新闻分词并且去除停用词之后的列表的列表
        label_list= pickle.load(f)#新闻的积极评论数量
        time_list = pickle.load(f)  # 所有新闻的label pop列表，评论数量
        print(len(news_listseq))
    logger.info("[INFO] news_list，time_list,label_list已读取！")
    #目前只用了一个label，预测热度pop
    #热度为点赞转发评论平均值或者求和。

    #max归一化label
    label_data = label_list
    label_y = np.array(label_data)
    label_data = maxminnorm(label_y)
    # 数据清洗--------------------------------
    for i in range(10):
        data_cleangnn(news_list)
    print(len(news_list))

    # 数据清洗判断--------------------------------
    new_x = []
    new_y = []
    new_t = []
    for i in range(len(news_list)):
        if news_list[i] != [] and not np.isnan(label_data[i]):
            new_x.append(news_list[i])
            new_y.append(label_data[i])
            new_t.append(time_list[i])
    #change time str to minmax-number
    new_t = changetime_list(new_t)

    #取前5W条新闻构建数据集
    new_x = new_x[0:50000]
    new_y = new_y[0:50000]
    new_t = new_t[0:50000]

    train_x, train_y, train_t , valid_x, valid_y, valid_t , test_x, test_y, test_t =data_extract_event(new_x,new_y,new_t)

    with open('../pkldata/disa_train_event_data.pkl', 'wb') as f:
        pickle.dump((train_x, train_y, train_t), f, pickle.HIGHEST_PROTOCOL)#训练集
    logger.info("[INFO] train_x, train_y，train_t已保存至pkldata/train_event_data.pkl文件！")
    with open('../pkldata/disa_valid_event_data.pkl', 'wb') as f:
        pickle.dump((valid_x, valid_y, valid_t), f, pickle.HIGHEST_PROTOCOL)#验证集
    logger.info("[INFO] valid_x, valid_y，valid_t已保存至pkldata/valid_event_data.pkl文件！")
    with open('../pkldata/disa_test_event_data.pkl', 'wb') as f:
        pickle.dump((test_x, test_y, test_t), f, pickle.HIGHEST_PROTOCOL)#测试集
    logger.info("[INFO] test_x, test_y，test_t已保存至pkldata/test_event_data.pkl文件！")


    # ---------------------------mind english dataset process--------------------------------
    with open('../pkldata3/mindnews_data.pkl', 'rb') as f:
        news_list= pickle.load(f)#对所有新闻分词并且去除停用词之后的列表的列表
        label_list= pickle.load(f)#新闻的label列表
        time_list = pickle.load(f)#所有新闻的时间列表
        id_list=pickle.load(f)#新闻的id_list

    logger.info("[INFO] news_list，time_list,label_list已读取！")
    # max归一化label
    label_data = label_list
    label_y = np.array(label_data)
    label_data = maxminnorm(label_y)
    # 数据清洗--------------------------------
    for i in range(10):
        data_cleangnn(news_list)
    print(len(news_list))
    # 数据清洗判断--------------------------------
    new_x = []
    new_y = []
    new_t = []
    new_id = []
    for i in range(len(news_list)):
        if news_list[i] != [] and not np.isnan(label_data[i]):
            new_x.append(news_list[i])
            new_y.append(label_data[i])
            new_t.append(time_list[i])
            new_id.append(id_list[i])
    new_t = changetime_list(new_t)

    # 取前5W条新闻构建数据集
    new_x = new_x[0:50000]
    new_y = new_y[0:50000]
    new_t = new_t[0:50000]
    new_id=new_id[0:50000]
    train_x, train_y, train_t,train_id, valid_x, valid_y, valid_t,valid_id,test_x, test_y, test_t,test_id = data_extract_mind(new_x, new_y,
                                                                                                      new_t,new_id)
    with open('../pkldata3/mind_train_event_data.pkl', 'wb') as f:
        pickle.dump((train_x, train_y, train_t ,train_id), f, pickle.HIGHEST_PROTOCOL)#训练集
    logger.info("[INFO] train_x, train_y, train_t ,train_id已保存至pkldata/mind_train_event_data.pkl文件！")
    with open('../pkldata3/mind_valid_event_data.pkl', 'wb') as f:
        pickle.dump((valid_x, valid_y, valid_t ,valid_id), f, pickle.HIGHEST_PROTOCOL)#验证集
    logger.info("[INFO] valid_x, valid_y, valid_t ,valid_id已保存至pkldata/mind_valid_event_data.pkl文件！")
    with open('../pkldata3/mind_test_event_data.pkl', 'wb') as f:
        pickle.dump((test_x, test_y, test_t ,test_id), f, pickle.HIGHEST_PROTOCOL)#测试集
    logger.info("[INFO] test_x, test_y,test_t ,test_id已保存至pkldata/mind_test_event_data.pkl文件！")



