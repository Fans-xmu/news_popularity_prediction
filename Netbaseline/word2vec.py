# encoding: utf-8
'''
@author: Fans
@file: word2vec.py
@time: 2021/1/17 14:05
@desc:
'''
from gensim.models import KeyedVectors

if __name__ == "__main__":

    #model = KeyedVectors.load_word2vec_format("pkldata/70000-small.txt")
    model = KeyedVectors.load_word2vec_format("pkldata/Tencent_AILab_ChineseEmbedding.txt")
    '''
    for content,sim in model.most_similar('特朗普', topn=10):
        print(content,sim)
    print('\n')
    for content,sim in model.most_similar(positive=['女', '国王'], negative=['男'], topn=5):
        print(content, sim)
    tensor_a=model['国王']
    '''
    vocabsum = list(model.wv.vocab.keys())
    print(len(vocabsum))
