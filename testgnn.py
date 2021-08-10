# encoding: utf-8
'''
@author: Fans
@file: testgnn.py
@time: 2021/6/18 13:25
@desc:
'''
import dgl
import numpy as np
import torch
from module.data_loader_gnn import ExamplegnnSet
from module.embedding import Word_Embedding
from module.vocabulary import Vocab
from tools.logger import *
import logging
import os
import pickle
import argparse
from sklearn.metrics import explained_variance_score,median_absolute_error
import math
def graph_collate_fn(samples):
    '''
    :param batch: (G, input_pad)
    :return:
    '''
    graphs, index= map(list, zip(*samples))
    graph_len = [len(g.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)) for g in graphs]  # sent node of graph
    sorted_len, sorted_index = torch.sort(torch.LongTensor(graph_len), dim=0, descending=True)
    batched_graph = dgl.batch([graphs[idx] for idx in sorted_index])
    return batched_graph, [index[idx] for idx in sorted_index]
def setup_eval_MAE(hps,test_loader,new_model):
    logger.info("loss:MAE , the acc in test_data:")
    # new_model.eval()
    test_loss = 0
    criterion = torch.nn.L1Loss()
    criterion2 = torch.nn.MSELoss()
    for i, (G,index) in enumerate(test_loader):
        if hps.cuda:
            G = G.to(torch.device("cuda"))
            index = torch.LongTensor(index).to(torch.device("cuda"))
        outputs = new_model.forward(G)
        outputs = outputs.squeeze(1)

        Nnode_id = G.filter_nodes(lambda nodes: nodes.data["dtype"] == 2)
        label = G.ndata["label"][Nnode_id]
        label = label.float()
        label = label.to(torch.device("cuda"))
        # print(label)
        loss = criterion(outputs.float(), label.float())
        test_loss += loss.item()

    logger.info("MAE:{:.6f}".format(test_loss / len(test_loader)))
def setup_eval_MSE(hps,test_loader,new_model):
    # new_model.eval()
    test_loss = 0
    criterion2 = torch.nn.MSELoss()
    for i, (G,index) in enumerate(test_loader):
        if hps.cuda:
            G = G.to(torch.device("cuda"))
            index = torch.LongTensor(index).to(torch.device("cuda"))
        outputs = new_model.forward(G)
        outputs = outputs.squeeze(1)

        Nnode_id = G.filter_nodes(lambda nodes: nodes.data["dtype"] == 2)
        label = G.ndata["label"][Nnode_id]
        label = label.float()
        label = label.to(torch.device("cuda"))
        loss = criterion2(outputs.float(), label.float())
        test_loss += loss.item()

    logger.info("MSE:{:.6f}".format(test_loss / len(test_loader)))
def smape(y_true, y_pred):
    return 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))) * 100
def setup_eval_MedAE(hps, test_loader, new_model):
    smape_loss = 0
    evs_loss = 0
    meadae_loss = 0
    for i, (G,index) in enumerate(test_loader):
        if hps.cuda:
            G = G.to(torch.device("cuda"))
            index = torch.LongTensor(index).to(torch.device("cuda"))
        outputs = new_model.forward(G)
        outputs = outputs.squeeze(1)

        Nnode_id = G.filter_nodes(lambda nodes: nodes.data["dtype"] == 2)
        label = G.ndata["label"][Nnode_id]
        label = label.float()
        label = label.to(torch.device("cuda"))
        rank_list = outputs.cpu().detach().tolist()
        label_list = label.cpu().detach().tolist()

        label_list = changezero(label_list, rank_list)
        y_pred = np.array(rank_list)
        y_true = np.array(label_list)
        evs_loss += explained_variance_score(y_true, y_pred)
        smape_loss += smape(y_true, y_pred)
        meadae_loss += median_absolute_error(y_true, y_pred)

    logger.info("SMAPE:{:.3f}".format(smape_loss / len(test_loader)))
    logger.info("explained_variance_score:{:.3f}".format(evs_loss / len(test_loader)))
    logger.info("meadAE:{:.3f}".format(meadae_loss / len(test_loader)))


def changezero(label_list, rank_list):
    label_ls = []
    for i in range(len(label_list)):
        key = label_list[i]
        if key == 0 and rank_list[i] == 0:
            key = 1
            rank_list[i] = 1
        label_ls.append(key)
    return label_ls
def setup_eval_RMSE(hps, test_loader, new_model):
    criterion2 = torch.nn.MSELoss()
    test_loss = 0
    for i, (G,index) in enumerate(test_loader):
        if hps.cuda:
            G = G.to(torch.device("cuda"))
            index = torch.LongTensor(index).to(torch.device("cuda"))
        outputs = new_model.forward(G)
        outputs = outputs.squeeze(1)

        Nnode_id = G.filter_nodes(lambda nodes: nodes.data["dtype"] == 2)
        label = G.ndata["label"][Nnode_id]
        label = label.float()
        label = label.to(torch.device("cuda"))
        loss = torch.sqrt(criterion2(outputs.float(), label.float()))
        test_loss += loss.item()

    logger.info("RMSE:{:.6f}".format(test_loss / len(test_loader)))
def setup_eval_R2(hps, test_loader, new_model):
    MRR_loss = 0
    NDCG_loss = 0
    for i, (G, index) in enumerate(test_loader):
        if hps.cuda:
            G = G.to(torch.device("cuda"))
            index = torch.LongTensor(index).to(torch.device("cuda"))
        outputs = new_model.forward(G)
        outputs = outputs.squeeze(1)

        Nnode_id = G.filter_nodes(lambda nodes: nodes.data["dtype"] == 2)
        label = G.ndata["label"][Nnode_id]
        label = label.float()
        label = label.to(torch.device("cuda"))
        rank_list = outputs.cpu().detach().tolist()
        label_list = label.cpu().detach().tolist()
        MRR, NDCG = Seqevalute(rank_list, label_list)
        # print("MRR:", MRR)
        # print("NDCG", NDCG)
        MRR_loss += MRR
        NDCG_loss += NDCG
    logger.info("NDCG:{:.3f}".format(NDCG_loss / len(test_loader)))
    logger.info("MRR:{:.3f}".format(MRR_loss / len(test_loader)))
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
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ShuaiGat Model')

    # Where to find data
    parser.add_argument('--data_dir', type=str, default='pkldata/', help='The dataset directory.')
    parser.add_argument('--cache_dir', type=str, default='cache/', help='The processed dataset directory')
    parser.add_argument('--embedding_path', type=str, default='./pkldata',
                        help='Path expression to external word embedding.')
    # data处理

    # Important settings
    parser.add_argument('--model', type=str, default='ShuaiGat', help='model structure[HSG|HDSG]')
    parser.add_argument('--restore_model', type=str, default='None',
                        help='Restore model for further training. [bestmodel/bestFmodel/earlystop/None]')

    # Where to save output
    parser.add_argument('--save_root', type=str, default='save/', help='Root directory for all model.')
    parser.add_argument('--log_root', type=str, default='log/', help='Root directory for all logging.')

    # Hyperparameters
    parser.add_argument('--gpu', type=str, default='5', help='GPU ID to use. [default: 0]')
    parser.add_argument('--cuda', action='store_true', default=True, help='GPU or CPU [default: False]')
    parser.add_argument('--vocab_size', type=int, default=60000, help='Size of vocabulary. [default: 3000]')
    # parser.add_argument('--vocab_size', type=int, default=50000,help='Size of vocabulary. [default: 50000]')
    parser.add_argument('--n_epochs', type=int, default=10, help='Number of epochs [default: 20]')

    parser.add_argument('--batch_size', type=int, default=128, help='Mini batch size [default: 1]')
    parser.add_argument('--n_iter', type=int, default=1, help='iteration hop [default: 1]')

    parser.add_argument('--word_embedding', action='store_true', default=True,
                        help='whether to use Word embedding [default: True]')
    parser.add_argument('--word_emb_dim', type=int, default=128, help='Word embedding size [default: 128]')
    parser.add_argument('--embed_train', action='store_true', default=False,
                        help='whether to train Word embedding [default: False]')
    parser.add_argument('--feat_embed_size', type=int, default=50, help='feature embedding size [default: 50]')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of GAT layers [default: 1]')
    parser.add_argument('--lstm_hidden_state', type=int, default=128, help='size of lstm hidden state [default: 128]')
    parser.add_argument('--lstm_layers', type=int, default=2, help='Number of lstm layers [default: 2]')
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
    parser.add_argument('--sent_max_len', type=int, default=10,
                        help='max length of sentences (max source text sentence tokens)')
    parser.add_argument('--doc_max_timesteps', type=int, default=5,
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

    # File paths

    LOG_PATH = args.log_root

    data_file = os.path.join(args.data_dir, "train_event_data.pkl")
    valid_file = os.path.join(args.data_dir, "valid_event_data.pkl")
    vocab_file = os.path.join(args.data_dir, "event_vocab.pkl")
    train_w2s_path = os.path.join(args.cache_dir, "train.w2s.tfidf_event.pkl")
    val_w2s_path = os.path.join(args.cache_dir, "val.w2s.tfidf_event.pkl")
    embedding_path = os.path.join(args.data_dir, "word_embedding")
    test_file = os.path.join(args.data_dir, "test_event_data.pkl")
    test_w2s_path = os.path.join(args.cache_dir, "test.w2s.tfidf_event.pkl")
    entity_file = os.path.join(args.data_dir, "entitylist_data.pkl")

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
    # ----------------------------------test-----------------------------------
    test_dataset = ExamplegnnSet(test_file, vocab, hps.doc_max_timesteps, hps.sent_max_len, test_w2s_path)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=hps.batch_size, shuffle=False,
                                              collate_fn=graph_collate_fn)
    modelname='bestmodel_weibo_MAE_FHgnn_noentity'

    #modelname='earlystop_ShuaiGat_2'
    new_model = torch.load('savemodel_entain/'+modelname)
    print("The model :" + modelname)
    setup_eval_MAE(hps, test_loader, new_model)
    # setup_eval_MSE(hps, test_loader, new_model)
    setup_eval_RMSE(hps, test_loader, new_model)
    setup_eval_MedAE(hps, test_loader, new_model)
    #setup_eval_R2(hps, test_loader, new_model)
