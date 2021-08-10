# encoding: utf-8
'''
@author: Fans
@file: 3_entitiy_named.py
@time: 2021/1/8 15:29
@desc: Extract entities from news

'''
from tools.logger import *
import pickle
from LAC import LAC
import jieba

def creat_entitydict(entity_list,vocab):
    entity_dict = {}
    entity_new_dict = {}
    index=0
    for entity in entity_list:
        entity_dict[entity]=index
        entity_new_dict[index] = vocab.word2id(entity)
        index+=1
    return entity_dict,entity_new_dict


if __name__ == "__main__":
    #--------------chinese dataset entity name recognized----------------
    with open('../pkldata/disa_event_data.pkl', 'rb') as f:
        time_list = pickle.load(f)  # news published time list, ['']
        news_list = pickle.load(f)  # news content list, ['']
        label_list = pickle.load(f)  # the sum of news's shareNum，reviewNum，likeNum, [int]

    logger.info("[INFO] news的news_list，label_list，reviews_list，videolink_list already read successfully.")
    logger.info("[INFO] all news num：%d",len(news_list))


    # 装载ner模型------baidu LAC
    lac = LAC(mode='lac')

    # entity extract
    word_lists=[]
    ner_list=[]
    for text in news_list:
        if text=='':
            word_lists.append([])
            ner_list.append([])
        else:
            seg_result = lac.run(text)
            word_lists.append(seg_result[0])
            ner_list.append(seg_result[1])
    entity_dict = {}
    for i in range(len(ner_list)):
        for j in range(len(ner_list[i])):
            if ner_list[i][j] in ['PER', 'LOC', 'ORG']:
                entity_dict[word_lists[i][j]] = ner_list[i][j]
    entity_list = list(entity_dict.keys())

    stop_words = [w.strip() for w in open('../pkldata/停用词词典.txt', encoding='utf-8').readlines()]


    entity_list = [w for w in entity_list if w not in stop_words]

    with open('../pkldata/disa_entitylist_data.pkl', 'wb') as f:
        pickle.dump(entity_list, f, pickle.HIGHEST_PROTOCOL)  # 实体列表
        pickle.dump(entity_dict, f, pickle.HIGHEST_PROTOCOL)  # 实体字典
    entity_file = "../pkldata/disa_entitylist_data.pkl"

    # 筛选步骤二，仅选择有embedding的实体
    with open(vocab_file, 'rb') as f:
        vocab_list = pickle.load(f)  # 词库
        vocab_num = pickle.load(f)  # 词的数量
    vocab_size = vocab_num
    logger.info("[INFO] vocab_list,vocab_num读取成功！")
    vocab = Vocab(vocab_list, 100000)
    with open(entity_file, 'rb') as f:
        entity_list = pickle.load(f)  # 数据集
    entity_dict, entity_new_dict = creat_entitydict(entity_list, vocab)
    print(len(entity_list))

    with open('../pkldata/disa_entitylist_data.pkl', 'wb') as f:
        pickle.dump(entity_dict, f, pickle.HIGHEST_PROTOCOL)  # 实体列表
        pickle.dump(entity_new_dict, f, pickle.HIGHEST_PROTOCOL)  # 实体字典