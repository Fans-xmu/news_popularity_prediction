# encoding: utf-8
'''
@author: Fans
@file: testseq.py
@time: 2021/6/12 15:14
@desc:
'''
import dgl
import numpy as np
import torch
import math
from LSTMlstm import LSTMlstm
from LSTMpcnn import LSTMpcnn
from sklearn.metrics import explained_variance_score,median_absolute_error
from module.data_loader_seq import ExampleseqSet
from module.embedding import Word_Embedding
from module.vocabulary import Vocab
from tools.logger import *
import logging
import os
import pickle
import argparse
from sklearn.metrics import r2_score

def setup_eval_MAE(hps,test_loader,new_model):
    logger.info("loss:MAE , the acc in test_data:")
    # new_model.eval()
    test_loss = 0
    criterion = torch.nn.L1Loss()
    criterion2 = torch.nn.MSELoss()
    for i, (news_list,label_list) in enumerate(test_loader):
        if hps.cuda:
            #print(news_list)
            #print(label_list)
            news_list = torch.LongTensor(news_list).to(torch.device("cuda"))
            label_list = torch.FloatTensor(label_list).to(torch.device("cuda"))
        outputs = new_model.forward(news_list)
        outputs = outputs.squeeze(1)

        label = label_list # [n_nodes]
        # print(label)
        loss = criterion(outputs.float(), label.float())
        test_loss += loss.item()


    logger.info("MAE:{:.6f}".format(test_loss / len(test_loader)))
def setup_eval_MSE(hps,test_loader,new_model):
    # new_model.eval()
    test_loss = 0
    criterion2 = torch.nn.MSELoss()
    for i, (news_list,label_list) in enumerate(test_loader):
        if hps.cuda:
            news_list = torch.LongTensor(news_list).to(torch.device("cuda"))
            label_list = torch.FloatTensor(label_list).to(torch.device("cuda"))
        outputs = new_model.forward(news_list)
        outputs = outputs.squeeze(1)

        label = label_list  #[n_nodes]
        # print(label)
        loss = criterion2(outputs.float(), label.float())
        test_loss += loss.item()


    logger.info("MSE:{:.6f}".format(test_loss / len(test_loader)))

def setup_eval_RMSE(hps, test_loader, new_model):
    criterion2 = torch.nn.MSELoss()
    test_loss = 0
    for i, (news_list,label_list) in enumerate(test_loader):
        if hps.cuda:
            news_list = torch.LongTensor(news_list).to(torch.device("cuda"))
            label_list = torch.FloatTensor(label_list).to(torch.device("cuda"))
        outputs = new_model.forward(news_list)
        outputs = outputs.squeeze(1)

        label = label_list  # [n_nodes]
        # print(label)
        loss = torch.sqrt(criterion2(outputs.float(), label.float()))
        test_loss += loss.item()

    logger.info("RMSE:{:.6f}".format(test_loss / len(test_loader)))


def smape(y_true, y_pred):
    return 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))) * 100
def setup_eval_MedAE(hps, test_loader, new_model):
    smape_loss = 0
    evs_loss = 0
    meadae_loss=0
    for i, (news_list,label_list) in enumerate(test_loader):
        if hps.cuda:
            news_list = torch.LongTensor(news_list).to(torch.device("cuda"))
            label_list = torch.FloatTensor(label_list).to(torch.device("cuda"))
        outputs = new_model.forward(news_list)
        outputs = outputs.squeeze(1)
        label = label_list  # [n_nodes]
        # print(label)
        rank_list = outputs.cpu().detach().tolist()
        label_list = label.cpu().detach().tolist()

        label_list=changezero(label_list,rank_list)
        y_pred = np.array(rank_list)
        y_true = np.array(label_list)
        evs_loss += explained_variance_score(y_true, y_pred)
        smape_loss += smape(y_true, y_pred)
        meadae_loss+=median_absolute_error(y_true, y_pred)

    logger.info("SMAPE:{:.3f}".format(smape_loss / len(test_loader)))
    logger.info("explained_variance_score:{:.3f}".format(evs_loss / len(test_loader)))
    logger.info("meadAE:{:.4f}".format(meadae_loss / len(test_loader)))
def changezero(label_list,rank_list):
    label_ls=[]
    for i in range(len(label_list)):
        key = label_list[i]
        if key==0 and rank_list[i]==0:
            key=1
            rank_list[i]=1
        label_ls.append(key)
    return label_ls

def takeSecond(elem):
    return elem[1]


def eval_entitysequence_metric(hps,new_model,entity_id2index,entity_index2id,entity_idlist,test_loader,K):
    news_all_list=[]
    label_all_list=[]
    pred_all_list=[]
    for i, (news_list, label_list) in enumerate(test_loader):
        news_batch_list=torch.LongTensor(news_list).tolist()
        label_batch_list=torch.FloatTensor(label_list).tolist()
        news_batch_list_new=[news[0:30] for news in news_batch_list]
        news_all_list.extend(news_batch_list_new)
        label_all_list.extend(label_batch_list)

        if hps.cuda:
            news_list = torch.LongTensor(news_list).to(torch.device("cuda"))
        outputs = new_model.forward(news_list)
        outputs = outputs.squeeze(1)
        pred_list = outputs.cpu().detach().tolist()
        pred_all_list.extend(pred_list)
    #all news include
    entity_map=[]
    for i in entity_idlist:
        entity_map.append([])
    for i in range(len(news_all_list)):
        for word in news_all_list[i]:
            if word in entity_idlist:
                entity_map[entity_id2index[word]].append(i)
    seq_eval=[]
    for seq in entity_map:#只考虑大于K的序列排序
        if len(seq)>K and len(seq)<3*K:
            seq_eval.append(seq)
        elif len(seq)>3*K:
            seq_eval.append(seq)
    #对每一个实体下的news序列的热度计算topK排序
    NDCG_all=0
    HR_all=0
    for pre in seq_eval:
        lab_list=[label_all_list[i] for i in pre]
        pre_list = [pred_all_list[i] for i in pre]
        idlist = [i for i in range(len(lab_list))]

        pre_list_sort = list(zip(idlist, pre_list))
        lab_list_sort = list(zip(idlist, lab_list))

        pre_list_sort.sort(key=takeSecond, reverse=True)
        lab_list_sort.sort(key=takeSecond, reverse=True)

        pre_list = []
        for ids, _ in pre_list_sort:
            pre_list.append(ids)
        lab_list = []
        for ids, _ in lab_list_sort:
            lab_list.append(ids)
        pre_list=pre_list[0:K]
        lab_list=lab_list[0:K]

        NDCG=getNDCG(lab_list,pre_list)
        HR=getHR(lab_list,pre_list)
        NDCG_all+=NDCG
        HR_all+=HR

    logger.info("NDCG@"+str(K)+":{:.4f}".format(NDCG_all / len(seq_eval)))
    logger.info("HR@"+str(K)+":{:.4f}".format(HR_all / len(seq_eval)))
    logger.info(">2*Klength:{:.3f}".format( len(seq_eval)))
def getHR(rank_list, pred_list):
    ar=1/len(rank_list)
    HR=0
    for news in rank_list:
        if news in pred_list:
            HR+=ar
    return HR
def getDCG(scores):
    return np.sum(
        np.divide(np.power(2, scores) - 1, np.log2(np.arange(scores.shape[0], dtype=np.float32) + 2)),
        dtype=np.float32)
def getNDCG(rank_list, pos_items):
    relevance = np.ones_like(pos_items)
    it2rel = {it: r for it, r in zip(pos_items, relevance)}
    rank_scores = np.asarray([it2rel.get(it, 0.0) for it in rank_list], dtype=np.float32)
    idcg = getDCG(relevance)

    dcg = getDCG(rank_scores)
    if dcg == 0.0:
        return 0.0
    ndcg = dcg / idcg
    return ndcg
def creat_entitydict(entity_list,vocab):
    entity_id2index = {}
    entity_index2id = {}
    index=0
    entity_idlist=[]
    for entity in entity_list:
        entity_id2index[vocab.word2id(entity)]=index
        entity_index2id[index] = vocab.word2id(entity)
        entity_idlist.append(vocab.word2id(entity))
        index+=1
    return entity_id2index,entity_index2id,entity_idlist
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LSTMlstm Model')

    # Where to find data
    parser.add_argument('--data_dir', type=str, default='pkldata/', help='The dataset directory.')
    parser.add_argument('--cache_dir', type=str, default='cache/', help='The processed dataset directory')
    parser.add_argument('--embedding_path', type=str, default='./pkldata',
                        help='Path expression to external word embedding.')
    # data处理

    # Important settings
    parser.add_argument('--model', type=str, default='LSTMpcnn', help='model structure[HSG|HDSG]')
    parser.add_argument('--restore_model', type=str, default='None',
                        help='Restore model for further training. [bestmodel/bestFmodel/earlystop/None]')

    # Where to save output
    parser.add_argument('--save_root', type=str, default='save/', help='Root directory for all model.')
    parser.add_argument('--log_root', type=str, default='log/', help='Root directory for all logging.')

    # Hyperparameters
    parser.add_argument('--gpu', type=str, default='1', help='GPU ID to use. [default: 0]')
    parser.add_argument('--cuda', action='store_true', default=True, help='GPU or CPU [default: False]')
    parser.add_argument('--vocab_size', type=int, default=100000, help='Size of vocabulary. [default: 3000]')
    # parser.add_argument('--vocab_size', type=int, default=50000,help='Size of vocabulary. [default: 50000]')
    parser.add_argument('--n_epochs', type=int, default=50, help='Number of epochs [default: 20]')

    parser.add_argument('--batch_size', type=int, default=1, help='Mini batch size [default: 1]')
    parser.add_argument('--n_iter', type=int, default=1, help='iteration hop [default: 1]')

    parser.add_argument('--word_embedding', action='store_true', default=True,
                        help='whether to use Word embedding [default: True]')
    parser.add_argument('--word_emb_dim', type=int, default=128, help='Word embedding size [default: 128]')
    parser.add_argument('--embed_train', action='store_true', default=False,
                        help='whether to train Word embedding [default: False]')
    parser.add_argument('--feat_embed_size', type=int, default=50, help='feature embedding size [default: 50]')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of GAT layers [default: 1]')
    parser.add_argument('--lstm_hidden_state', type=int, default=128, help='size of lstm hidden state [default: 128]')
    parser.add_argument('--lstm_layers', type=int, default=3, help='Number of lstm layers [default: 2]')
    parser.add_argument('--bidirectional', action='store_true', default=True,
                        help='whether to use bidirectional LSTM [default: True]')
    parser.add_argument('--n_feature_size', type=int, default=128, help='size of node feature [default: 128]')
    parser.add_argument('--hidden_size', type=int, default=64, help='hidden size [default: 64]')
    parser.add_argument('--ffn_inner_hidden_size', type=int, default=512,
                        help='PositionwiseFeedForward inner hidden size [default: 512]')
    parser.add_argument('--n_head', type=int, default=2, help='multihead attention number [default: 8]')
    parser.add_argument('--recurrent_dropout_prob', type=float, default=0.1,
                        help='recurrent dropout prob [default: 0.1]')
    parser.add_argument('--atten_dropout_prob', type=float, default=0.1, help='attention dropout prob [default: 0.1]')
    parser.add_argument('--ffn_dropout_prob', type=float, default=0.1,
                        help='PositionwiseFeedForward dropout prob [default: 0.1]')
    parser.add_argument('--use_orthnormal_init', action='store_true', default=True,
                        help='use orthnormal init for lstm [default: True]')
    parser.add_argument('--sent_max_len', type=int, default=20,
                        help='max length of sentences (max source text sentence tokens)')
    parser.add_argument('--doc_max_timesteps', type=int, default=256,
                        help='max length of documents (max timesteps of documents)')

    # Training
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--lr_descent', action='store_true', default=True, help='learning rate descent')
    parser.add_argument('--grad_clip', action='store_true', default=False, help='for gradient clipping')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='for gradient clipping max gradient normalization')

    # parser.add_argument('-m', type=int, default=3, help='decode summary length')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.set_printoptions(threshold=50000)
    LOG_PATH = args.log_root

    data_file = os.path.join(args.data_dir, "disa_train_event_data.pkl")
    valid_file = os.path.join(args.data_dir, "disa_valid_event_data.pkl")
    vocab_file = os.path.join(args.data_dir, "disa_event_vocab.pkl")
    train_w2s_path = os.path.join(args.cache_dir, "disa_train.w2s.tfidf_event.pkl")
    val_w2s_path = os.path.join(args.cache_dir, "disa_val.w2s.tfidf_event.pkl")
    embedding_path = os.path.join(args.data_dir, "word_embedding")
    test_file = os.path.join(args.data_dir, "disa_test_event_data.pkl")
    test_w2s_path = os.path.join(args.cache_dir, "disa_test.w2s.tfidf_event.pkl")
    entity_file = os.path.join(args.data_dir, "disa_entitylist_data.pkl")

    logger.info("Pytorch %s", torch.__version__)
    logger.info("[INFO] Create Vocab, vocab file is %s", vocab_file)
    with open(vocab_file, 'rb') as f:
        vocab_list = pickle.load(f)  # 词库
        vocab_num = pickle.load(f)  # 词的数量
    vocab_size = vocab_num
    logger.info("[INFO] vocab_list,vocab_num读取成功！")
    #vocab = Vocab(vocab_list, vocab_size)
    vocab = Vocab(vocab_list, args.vocab_size)
    hps = args
    logger.info(hps)


    with open(entity_file, 'rb') as f:
        entity_list = pickle.load(f)  # 数据集
    # ----------------------------------test-----------------------------------
    test_dataset = ExampleseqSet(test_file, vocab, hps.doc_max_timesteps, hps.sent_max_len)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=hps.batch_size, shuffle=True)

    entity_id2index,entity_index2id,entity_idlist = creat_entitydict(entity_list, vocab)

    modelname='bestmodel_MLPCNN'
    #modelname='earlystop_LSTMlstm_3_BiLSTM_False'
    new_model = torch.load('savemodel_disa/'+modelname)
    #new_model = torch.load('savemodel/earlystop_LSTMlstm_1')
    print("The model :" + modelname)
    K=5
    eval_entitysequence_metric(hps, new_model, entity_id2index, entity_index2id, entity_idlist, test_loader, K)
    setup_eval_MAE(hps, test_loader, new_model)
    # setup_eval_MSE(hps, test_loader, new_model)
    setup_eval_RMSE(hps, test_loader, new_model)

    setup_eval_MedAE(hps, test_loader, new_model)