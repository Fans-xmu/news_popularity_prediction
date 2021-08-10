#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: zzz_jq
# Time: 2020/4/22 9:44

from keras.layers.embeddings import Embedding
from keras.layers import AveragePooling1D, Lambda, Activation
from keras.layers import Conv1D, MaxPooling1D, Dropout, Dense, Input,BatchNormalization,Reshape
import keras.backend as K
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, classification_report
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr
from keras.models import Model
import numpy as np
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences


def mul(x):
    return K.batch_dot(x[0], x[1])


def attention(bn):
    avg_t = Reshape((bn.shape[2].value, bn.shape[1].value))(bn)
    out = Dense(units=1, use_bias=True, activation='tanh')(avg_t)
    weight = Activation('softmax')(out)
    output = Lambda(mul)([bn, weight])
    return output


def cnn_attention_model(x_train_padded_seqs, x_test_padded_seqs, y_train, y_test, embedding_matrix,embedding_length,count):
    main_input = Input(shape=(50,), dtype='float64')  # 输入的维度和数据类型
    # pre-trained word embedding
    embedder = Embedding(len(vocab) + 1, embedding_length, input_length=50, weights=[embedding_matrix], trainable=False)
    embed = embedder(main_input)  # (None,50,128)
    # cnn
    cnn = Conv1D(256, 3, padding='same', strides=1, activation='relu')(embed)
    cnn = MaxPooling1D(pool_size=48)(cnn)
    # cnn = Conv1D(256, 3, padding='same', strides=1, activation='relu')(cnn)
    # cnn = MaxPooling1D(pool_size=24)(cnn)
    drop_cnn = Dropout(0.2)(cnn)
    bn = BatchNormalization()(drop_cnn)
    out = attention(bn)
    out = Reshape([out.shape[1].value])(out)
    # classification
    y_pred = Dense(name='y_pred', units=1, activation='sigmoid')(out)
    # regression
    # out = Dense(units=128, activation='relu')(out)
    # y_pred = Dense(name='y_pred',units=1, activation='relu')(out)
    model = Model(inputs=main_input, outputs=y_pred)
    # classification
    losses = {'y_pred': 'binary_crossentropy'}
    # regression
    # losses = {'y_pred': 'mean_squared_error'}
    model.compile(loss=losses, optimizer='adam', metrics=['binary_crossentropy'])
    model.fit(x_train_padded_seqs, y_train, batch_size=128, epochs=50, shuffle=True)
    model.save("../model_save/model"+str(count)+".h5")
    test_pred = model.predict(x_test_padded_seqs)
    # classification
    test_pred[test_pred >= 0.5] = 1
    test_pred[test_pred < 0.5] = 0
    print(classification_report(y_test, test_pred))
    print('p: ', precision_score(y_test, test_pred))
    print('r: ', recall_score(y_test, test_pred))
    print('f1: ', f1_score(y_test, test_pred))
    # regression
    # print(np.sqrt(mean_squared_error(y_test, test_pred)))
    # print(spearmanr(y_test, test_pred))
    # print(r2_score(y_test, test_pred))
    # print(1 - ((1 - r2_score(y_test, test_pred)) * (len(x_test) - 1)) / (len(x_test) - len(x_test[0]) - 1))
    return f1_score(y_test, test_pred)


if __name__ == '__main__':
    # 通过两个RNN，然后一起训练。
    texts = Vocab.texts  # 评论分词后的文本
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    vocab = tokenizer.word_index
    labels= Vocab.labels
    x_train, x_test, y_train, y_test = train_test_split(texts, labels, test_size=0.1)
    y_train = [i[1] for i in y_train]
    y_test = [i[1] for i in y_test]
    x_train_word_ids = tokenizer.texts_to_sequences(x_train)
    x_test_word_ids = tokenizer.texts_to_sequences(x_test)  # 转为Index
    x_train_padded_seqs = pad_sequences(x_train_word_ids, maxlen=50)  # 将超过固定值的部分截掉，不足的在最前面用0填充
    x_test_padded_seqs = pad_sequences(x_test_word_ids, maxlen=50)

    # 情感值
    Senlines = open('../files/中文词向量训练模型/BosonNLP_sentiment_score/BosonNLP_sentiment_score.txt', 'r',
                    encoding='UTF-8').readlines()
    sendict = {}
    senwords = []
    for line in Senlines:
        word, sen = line.split(' ')
        senwords.append(word)
        sendict[word] = float(sen)

    model = gensim.models.Word2Vec.load('../files/中文词向量训练模型/word2vec/word_embedding')  # word2vec
    embedding_length = len(model.wv['平安'])  # 每一个embedding的维度
    embedding_matrix = np.zeros((len(vocab)+1, embedding_length))
    for word, i in vocab.items():
        try:
            embedding_vector = model.wv[str(word)]
            embedding_matrix[i] = embedding_vector
        except KeyError:
            continue

    # for word, i in vocab.items():
    #     try:
    #         embedding_vector = model.wv[str(word)]
    #         a = list(embedding_vector)
    #         if word in senwords:
    #             a.append(float(sendict[word]))
    #         else:
    #             a.append(0.0)
    #         embedding_vector = np.array(a)
    #         embedding_matrix[i] = embedding_vector
    #     except KeyError:
    #         continue
    sum = 0
    count = 0
    resultab = []
    for i in range(10):
        f1 = cnn_attention_model(x_train_padded_seqs, x_test_padded_seqs, y_train, y_test,
                             embedding_matrix, embedding_length,count)
        resultab.append(f1)
        sum += f1
