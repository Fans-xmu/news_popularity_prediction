# encoding: utf-8
'''
@author: Fans
@file: mind_csvreader.py
@time: 2021/1/12 14:51
@desc:change the MIND Data into pkl data[time_list,news_list,label_list,id_list]
'''


import csv
import pandas as pd
import numpy as np
import pickle
import re
import random

def clean(text):
    text = re.sub(r"(回复)?(//)?\s*@\S*?\s*(:| |$)", " ", text)  # 去除正文中的@和回复/转发中的用户名
    text = re.sub(r"\[\S+\]", "", text)      # 去除表情符号
    # text = re.sub(r"#\S+#", "", text)      # 保留话题内容
    URL_REGEX = re.compile(
        r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))',
        re.IGNORECASE)
    text = re.sub(URL_REGEX, "", text)       # 去除网址

    for i in ["转发微博","网页链接","网页链接转发","网页链接如下","图片链接如下","图片链接转发","秒拍视频"]:
        text = text.replace(i, "")       # 去除无意义的词语
    text = re.sub(r"\s+", " ", text) # 合并正文中过多的空格

    return text.strip()

def readnews(filename):
    csv_reader=[]
    with open(filename,"rt",encoding='utf-8') as f:
        for line in f:
            line=line.split('\t')
            if line[-1][-1] == '\n':
                line[-1] = line[-1][:-1]
            csv_reader.append(line)

    sample_list = []
    for s in csv_reader:
        nid=s[0]
        title=s[3]
        abstract=s[4]
        title_entity=s[6]
        abs_entity=s[7]
        pop=0
        sample_list.append([nid,title,abstract,title_entity,abs_entity,pop])

    return sample_list

def readuser(filename):
    csv_reader = []
    with open(filename, "rt", encoding='utf-8') as f:
        for line in f:
            line = line.split('\t')
            if line[-1][-1]=='\n':
                line[-1] = line[-1][:-1]
            csv_reader.append(line)

    sample_list = []
    for s in csv_reader:
        click_time=s[2][6:10]+'/'+s[2][0:5]+s[2][len(s[2])-2]+s[2][10:len(s[2])-2]
        history=s[3]
        impressions=s[4]
        sample_list.append([history,impressions,click_time])
    return sample_list

def catculate_pop(sample_list,user_history):
    news_pop=[]  #news_pop，新闻id和文本sample
    news_entity = {} #实体字典
    entity_name2id={}
    news2id={}
    id2news={}
    count=0
    for news in sample_list:
        #---------------news----------------
        news_id = news[0]
        #将news_id与下标联系起来
        news2id[news_id]=count
        id2news[count]=news_id

        news_content=news[1]+'.'+news[2]
        pop=0
        pubtime='2050'
        Entitiy_id='Q'
        count+=1
        #---------------entity--------------
        # print(news[3])
        if news[3]!=[]:
            entity_num=news[3].split('{')
            for enname in entity_num:
                if enname.find('Label'):
                    head=enname.find('Label')
                    tail=enname.find('Type')
                    entity_name=enname[head+9:tail-4]
                    entity_id=enname[enname.find('WikidataId')+14:enname.find('Confidence')-4]
                    if entity_id!='' and entity_name!='' and 'Q' in entity_id:

                        news_entity[entity_id] = entity_name
                        entity_name2id[entity_name] = entity_id
                        Entitiy_id=entity_id
        news_pop.append([Entitiy_id, news_content, pop, pubtime])
    #-----------catculate_pop--------------
    for click in user_history:
        clicknum=click[0]
        clicknum=clicknum.split()
        for id in clicknum:
            news_pop[news2id[id]][2] += 1

        clickornone=click[1]
        clickornone=clickornone.split()
        for click_n in clickornone:
            id=click_n[0:-2]
            usein=int(click_n[-1])
            news_pop[news2id[id]][2]+=(usein+1)
        clicktime=click[2]
        for id in clicknum:
            if news_pop[news2id[id]][3]>clicktime:
                news_pop[news2id[id]][3] = clicktime
        for click_n in clickornone:
            id=click_n[0:-2]
            if news_pop[news2id[id]][3] > clicktime:
                news_pop[news2id[id]][3] = clicktime
    return news_pop,news_entity,entity_name2id,news2id,id2news
if __name__ == "__main__":
    filename='MINDsmall_train/news.tsv'
    userfilename='MINDsmall_train/behaviors.tsv'
    sample_list=readnews(filename)#所有news新闻

    user_history=readuser(userfilename)#所有用户浏览点击历史
    print(user_history[0:5])
    print(len(sample_list))   #新闻总数
    news_pop, news_entity, entity_name2id,news2id, id2news = catculate_pop(sample_list, user_history)
    for i in news_pop[0:5]:
         print(i)
    print(entity_name2id)
    print(news_entity)
    # 读取并存取所有新闻相关的信息
    id_list=[]
    news_list = []
    label_list = []
    time_list=[]
    for s in news_pop:
        id_list.append(s[0])
        news_list.append(s[1])
        label_list.append(s[2])
        time_list.append(s[3])
    print(id_list[0:5])
    with open('mind_data.pkl', 'wb') as f:
        pickle.dump(id_list, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(time_list, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(news_list, f, pickle.HIGHEST_PROTOCOL)  # 所有新闻的内容列表，存储格式:['']
        pickle.dump(label_list, f, pickle.HIGHEST_PROTOCOL)  # 所有新闻的popularity
    print(len(news_list))
    with open('entity_data.pkl', 'wb') as f:
        pickle.dump(news_entity, f, pickle.HIGHEST_PROTOCOL)  # 实体字典
        pickle.dump(entity_name2id, f, pickle.HIGHEST_PROTOCOL)  # 实体name2id list
    print(len(list(news_entity.keys())))
