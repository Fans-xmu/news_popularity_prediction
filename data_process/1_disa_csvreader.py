# encoding: utf-8
'''
#first process
@author: Fans
@file: 1_disa_csvreader.py
@time: 2021/1/12 14:51
@desc:change the event csv file into pkl data[time_list,news_list,label_list]
'''
from tools.logger import *
import csv
import pandas as pd
import numpy as np
import pickle
import re
import random
import os
def clean(text):
    text = re.sub(r"(回复)?(//)?\s*@\S*?\s*(:| |$)", " ", text)  #去除正文中的@和回复/转发中的用户名
    text = re.sub(r"[a-zA-Z]+", "", text)
    text = re.sub('[a-zA-Z’!"#$%&\'()*+,-./:;<=>@★、…【《》？“”‘’！[\\]^_`{|}~\s]+','',text)
    text = re.sub(
        '[\001\002\003\004\005\006\007\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a]+','', text)
    text = re.sub(r"\[\S+\]", "", text)      #去除表情符号
    URL_REGEX = re.compile(
        r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))',
        re.IGNORECASE)
    text = re.sub(URL_REGEX, "", text)       #去除网址

    for i in ["转发微博","网页链接","网页链接转发","网页链接如下","图片链接如下","图片链接转发","秒拍视频",'【','↓','#','&',]:
        text = text.replace(i, "")       #去除无意义的词语
    text = text.replace('】', "，")      #将【】转化为一句话
    text = re.sub(r"\s+", " ", text)    #合并正文中过多的空格

    return text.strip()

def weiboread():
    years = ['2017_blog_data.pkl', '2018_blog_data.pkl', '2019_blog_data.pkl', '2020_blog_data.pkl']
    time_list = []
    news_list = []
    label_list = []
    reviews_list = []
    videolink_list = []
    rootdir = 'pkldata/'
    for year in years:
        with open(rootdir + year, 'rb') as f:
            time_l = pickle.load(f)  # 所有新闻的时间列表，存储格式:['']
            news_l = pickle.load(f)  # 所有新闻的内容列表['']
            label_l = pickle.load(f)  # 所有新闻的shareNum，reviewNum，likeNum:,[[a,b,c]]
            reviews_l = pickle.load(f)  # 所有新闻的评论列表[[]]
            videolink_l = pickle.load(f)  # 所有新闻的视频链接['']
        time_list += time_l
        news_list += news_l
        label_list += label_l
        reviews_list += reviews_l
        videolink_list += videolink_l

    return time_list,news_list,label_list,reviews_list,videolink_list
def readnews(root_dir,events,organ_list,contlist):
    sample_list = []
    for event in events:
        filename=root_dir+event
        csv_reader = csv.reader(open(filename,'r', encoding="utf-8"))
        count=0
        for s in csv_reader:
            count += 1
            if count%1000==0:
                print("step:",count)
            if s[11]=='点赞数':
                continue
            logger.info("[INFO] begin clean news content")
            for i in contlist:
                if i in s[3]:
                    news_content = s[4]
                    clean_news_content=clean(news_content)
                    time = s[12]
                    likeNum = s[11]
                    reviewNum = s[10]
                    shareNum = s[9]
                    if likeNum=='0' or reviewNum=='0' or shareNum=='0':
                        pop=random.randint(1, 10)
                    else:
                        pop = int(likeNum) + int(reviewNum) + int(shareNum)
                    sample_list.append([clean_news_content, time, pop])

            if s[3] in organ_list:
                news_content = s[4]
                clean_news_content = clean(news_content)
                time = s[12]
                likeNum = s[11]
                reviewNum = s[10]
                shareNum = s[9]
                if likeNum == '0' or reviewNum == '0' or shareNum == '0':
                    pop = random.randint(1, 10)
                else:
                    pop = int(likeNum) + int(reviewNum) + int(shareNum)
                sample_list.append([clean_news_content, time, pop])

            #     continue
    return sample_list
import json
if __name__ == "__main__":
    root_dir='../pkldata/灾难/'
    organ_path='../pkldata/发表部门.txt'
    city_path='../pkldata/城市.txt'
    organ_list=[w[:-1] for w in open(organ_path, 'r', encoding='UTF-8').readlines()]
    contlist=[w[:-1] for w in open(city_path, 'r', encoding='UTF-8').readlines()]
    events = os.listdir(root_dir)
    sample_list=readnews(root_dir,events,organ_list,contlist)

    time_list = []
    news_list = []
    label_list = []
    for s in sample_list:
        news_list.append(s[0])
        time_list.append(s[1])
        label_list.append(s[2])

    logger.info("[INFO] Select the first 50000 news：")
    new_x = []
    new_y = []
    new_t = []
    for i in range(len(news_list)):
        if len(news_list[i]) >= 130:
            new_x.append(news_list[i])
            new_y.append(label_list[i])
            new_t.append(time_list[i])
    print(len(new_x))
    news_list=new_x[0:50000]
    time_list=new_t[0:50000]
    label_list=new_y[0:50000]
    with open('../pkldata/disa_event_data.pkl', 'wb') as f:
        pickle.dump(time_list, f, pickle.HIGHEST_PROTOCOL)  # news published time list, ['']
        pickle.dump(news_list, f, pickle.HIGHEST_PROTOCOL)  # news content list, ['']
        pickle.dump(label_list, f, pickle.HIGHEST_PROTOCOL)  # the sum of news's shareNum，reviewNum，likeNum, [int]
    logger.info("[INFO] news's time_list，news_list，label_list，already saved to disa_event_data.pkl！")


