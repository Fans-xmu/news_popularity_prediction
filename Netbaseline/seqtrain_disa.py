# encoding: utf-8
'''
@author: Fans
@file: seqtrain.py
@time: 2021/6/10 17:28
@desc:
'''

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import pickle
import argparse
import datetime
import os
import shutil
import time

import dgl
import numpy as np
import torch
from MLP_CNN import MLPCNN
from LSTMlstm import LSTMlstm
from LSTMpcnn import LSTMpcnn
from module.data_loader_seq import ExampleseqSet
from module.embedding import Word_Embedding
from module.vocabulary import Vocab
from tools.logger import *
import logging

def save_model(model, save_file):
    with open(save_file, 'wb') as f:
        torch.save(model, f)
    logger.info('[INFO] Saving model to %s', save_file)



def setup_training_MAE(model, train_loader, valid_loader, valset, hps):
    """ Does setup before starting training (run_training)

        :param model: the model
        :param train_loader: train dataset loader
        :param valid_loader: valid dataset loader
        :param valset: valid dataset which includes text and summary
        :param hps: hps for model
        :return:
    """
    train_loss_list = []
    val_loss_list = []
    train_dir = os.path.join(hps.save_root, "Seqdisa")
    if os.path.exists(train_dir) and hps.restore_model != 'None':
        logger.info("[INFO] Restoring %s for training...", hps.restore_model)
        bestmodel_file = os.path.join(train_dir, hps.restore_model)
        model.load_state_dict(torch.load(bestmodel_file))
        hps.save_root = hps.save_root + "_reload"
    else:
        logger.info("[INFO] Create new model for training...")
        if os.path.exists(train_dir): shutil.rmtree(train_dir)
        os.makedirs(train_dir)

    try:
        model.to(torch.device("cuda:0"))
        run_training_MAE(model, train_loader, valid_loader, valset, hps, train_dir, train_loss_list, val_loss_list)
    except KeyboardInterrupt:
        logger.error("[Error] Caught keyboard interrupt on worker. Stopping supervisor...")
        save_model(model, os.path.join(train_dir, "earlystop"+'_'+hps.model+'_'+str(hps.lstm_layers)+'BiLSTM_'+str(hps.bidirectional)))

    return train_loss_list, val_loss_list




def run_training_MAE(model, train_loader, valid_loader, valset, hps, train_dir, train_loss_list, val_loss_list):
    '''  Repeatedly runs training iterations, logging loss to screen and log files

        :param model: the model
        :param train_loader: train dataset loader
        :param valid_loader: valid dataset loader
        :param valset: valid dataset which includes text and summary
        :param hps: hps for model
        :param train_dir: where to save checkpoints
        :return:
    '''
    logger.info("[INFO] Starting run_training")

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=hps.lr)

    # if hps.cuda:
    #     criterion = torch.nn.L1Loss().to(torch.device("cuda"))
    # else:
    #     criterion = torch.nn.L1Loss()
    if hps.cuda:
        criterion = torch.nn.MSELoss().to(torch.device("cuda"))
    else:
        criterion =torch.nn.MSELoss()

    best_train_loss = None
    best_loss = None
    best_F = None
    non_descent_cnt = 0
    saveNo = 0

    for epoch in range(1, hps.n_epochs + 1):
        train_loss = 0.0
        epoch_start_time = time.time()
        for i, (news_list,label_list) in enumerate(train_loader):
            iter_start_time = time.time()
            model.train()
            if hps.cuda:
                news_list = torch.LongTensor(news_list).to(torch.device("cuda"))
                label_list = torch.FloatTensor(label_list).to(torch.device("cuda"))

            outputs = model.forward(news_list)  # [n_snodes, 1]
            outputs = outputs.squeeze(1)
            outputs=outputs.float()

            label = label_list # [n_nodes]
            label=label.float()
            label=label.to(torch.device("cuda"))

            loss = criterion(outputs, label)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 记录误差
            train_loss += loss.item()

            if i != 0:
                print("[INFO] The batch step is :", i)
                print("train_loss:", train_loss / i)

        epoch_avg_loss = train_loss / len(train_loader)
        train_loss_list.append(epoch_avg_loss)
        #学习率是否decay
        if hps.lr_descent:
            new_lr = max(5e-6, hps.lr / (epoch + 1))
            for param_group in list(optimizer.param_groups):
                param_group['lr'] = new_lr
            logger.info("[INFO] The learning rate now is %f", new_lr)
        logger.info('   | end of epoch {:3d} | time: {:5.2f}s | epoch train loss {:5.6f} | '
                    .format(epoch, (time.time() - epoch_start_time), float(epoch_avg_loss)))

        if not best_train_loss or epoch_avg_loss < best_train_loss:
            save_file = os.path.join(train_dir, "bestmodel"+'_'+hps.model+'_'+"epoch:"+str(epoch))
            logger.info('[INFO] Found new best model with %.3f running_train_loss. Saving to %s', float(epoch_avg_loss),
                        save_file)
            save_model(model, save_file)
            best_train_loss = epoch_avg_loss

        elif epoch_avg_loss >= best_train_loss:
            logger.error("[Error] training loss does not descent. Stopping supervisor...")
            save_model(model, os.path.join(train_dir, "bestmodel"+'_'+hps.model+'_'+"epoch:"+str(epoch)))


        val_avg_loss, best_loss, non_descent = run_eval_MAE(model, valid_loader, hps, best_loss, non_descent_cnt)
        val_loss_list.append(val_avg_loss)
        print('end of epoch: {} \t average training Loss: {:.6f} \t valid loss: {:.6f} \t'.format(
            epoch,
            epoch_avg_loss,
            val_avg_loss
        ))

        if non_descent_cnt >= 3:
            logger.error("[Error] val loss does not descent for three times. Stopping supervisor...")
            save_model(model, os.path.join(train_dir, "earlystop"+'_'+hps.model+'_'+str(hps.lstm_layers)))
            break



def run_eval_MAE(model, loader, hps, best_loss, non_descent_cnt):
    '''
        Repeatedly runs eval iterations, logging to screen and writing summaries. Saves the model with the best loss seen so far.
        :param model: the model
        :param loader: valid dataset loader
        :param hps: hps for model
        :return:
    '''
    logger.info("[INFO] Starting eval for this model ...")
    # eval_dir = os.path.join(hps.save_root, "eval")  # make a subdir of the root dir for eval data
    # if not os.path.exists(eval_dir): os.makedirs(eval_dir)

    model.eval()
    # criterion=torch.nn.MSELoss()
    if hps.cuda:
        criterion = torch.nn.L1Loss().to(torch.device("cuda"))
    else:
        criterion = torch.nn.L1Loss()
    # iter_start_time = time.time()

    with torch.no_grad():
        val_loss = 0

        for i, (news_list,label_list) in enumerate(loader):
            if hps.cuda:
                news_list = torch.LongTensor(news_list).to(torch.device("cuda"))
                label_list = torch.FloatTensor(label_list).to(torch.device("cuda"))

            outputs = model.forward(news_list)
            outputs = outputs.squeeze(1)

            label = label_list # [n_nodes]
            label = label.float()
            label = label.to(torch.device("cuda"))
            loss = criterion(outputs.float(), label.float())

            val_loss += loss.item()

        val_avg_loss = val_loss / len(loader)

    if best_loss is None or val_avg_loss < best_loss:
        best_loss = val_avg_loss
        non_descent_cnt = 0
    else:
        non_descent_cnt += 1

    return val_avg_loss, best_loss, non_descent_cnt



def setup_eval_MAE(hps,test_loader,new_model):
    print("loss:MAE , the acc in test_data:")
    logger.info("loss:MAE , the acc in test_data:")
    # new_model.eval()
    test_loss = 0
    criterion = torch.nn.L1Loss()
    criterion2 = torch.nn.MSELoss()
    for i, (news_list,label_list) in enumerate(test_loader):
        if hps.cuda:
            news_list = torch.LongTensor(news_list).to(torch.device("cuda"))
            label_list = torch.FloatTensor(label_list).to(torch.device("cuda"))
        outputs = new_model.forward(news_list)
        outputs = outputs.squeeze(1)

        label = label_list # [n_nodes]
        # print(label)
        loss = criterion(outputs.float(), label.float())
        test_loss += loss.item()

    print("MAE:{:.6f}".format(test_loss / len(test_loader)))
    logger.info("MAE:{:.6f}".format(test_loss / len(test_loader)))
def setup_eval_MSE(hps,test_loader,new_model):
    print("loss:MAE , the acc in test_data:")
    logger.info("loss:MAE , the acc in test_data:")
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

    print("MSE:{:.6f}".format(test_loss / len(test_loader)))
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

    print("RMSE:{:.6f}".format(test_loss / len(test_loader)))
    logger.info("RMSE:{:.6f}".format(test_loss / len(test_loader)))




#---------------------------------------mind----------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='LSTMlstm Model')

    # Where to find data
    parser.add_argument('--data_dir', type=str, default='pkldata/', help='The dataset directory.')
    parser.add_argument('--cache_dir', type=str, default='cache/', help='The processed dataset directory')
    parser.add_argument('--embedding_path', type=str, default='./pkldata',
                        help='Path expression to external word embedding.')
    # data处理

    # Important settings
    parser.add_argument('--model', type=str, default='LSTMlstm', help='model structure[HSG|HDSG]')
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

    parser.add_argument('--word_embedding', action='store_true', default=True, help='whether to use Word embedding [default: True]')
    parser.add_argument('--word_emb_dim', type=int, default=128, help='Word embedding size [default: 128]')
    parser.add_argument('--embed_train', action='store_true', default=False,
                        help='whether to train Word embedding [default: False]')
    parser.add_argument('--feat_embed_size', type=int, default=50, help='feature embedding size [default: 50]')
    parser.add_argument('--n_layers', type=int, default=1, help='Number of GAT layers [default: 1]')
    parser.add_argument('--lstm_hidden_state', type=int, default=128, help='size of lstm hidden state [default: 128]')
    parser.add_argument('--lstm_layers', type=int, default=3, help='Number of lstm layers [default: 2]')
    parser.add_argument('--bidirectional', action='store_true', default=False,
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
    parser.add_argument('--doc_max_timesteps', type=int, default=3200,
                        help='max length of documents (max timesteps of documents)')

    # Training
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
    parser.add_argument('--lr_descent', action='store_true', default=True, help='learning rate descent')
    parser.add_argument('--grad_clip', action='store_true', default=False, help='for gradient clipping')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='for gradient clipping max gradient normalization')

    #parser.add_argument('-m', type=int, default=3, help='decode summary length')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.set_printoptions(threshold=50000)

    # File paths

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



    # train_log setting
    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH)
    nowTime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(LOG_PATH, "train_" + nowTime)
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info("Pytorch %s", torch.__version__)
    logger.info("[INFO] Create Vocab, vocab file is %s", vocab_file)
    with open(vocab_file, 'rb') as f:
        vocab_list= pickle.load(f)#词库
        vocab_num= pickle.load(f)#词的数量
    vocab_size = vocab_num
    logger.info("[INFO] vocab_list,vocab_num读取成功！")
    with open(data_file, 'rb') as f:
        sum_news_list, sum_labels,time_list = pickle.load(f)  # 数据集
    logger.info("[INFO] train_x, train_y读取成功！")
    print(sum_news_list[0:10])
    print(sum_labels[0:10])
    #vocab = Vocab(vocab_list, vocab_size)
    vocab = Vocab(vocab_list, args.vocab_size)
    #使用预训练的词embedding修改随机初始化的embedding

    embed = torch.nn.Embedding(vocab.size(), args.word_emb_dim, padding_idx=0)
    if args.word_embedding:
        embed_loader = Word_Embedding(embedding_path, vocab)
        vectors = embed_loader.load_my_vecs()
        pretrained_weight = embed_loader.add_unknown_words_by_uniform(vectors, args.word_emb_dim)
        embed.weight.data.copy_(torch.Tensor(pretrained_weight))
        embed.weight.requires_grad = args.embed_train

    hps = args
    logger.info(hps)


    train_dataset = ExampleseqSet(data_file, vocab, hps.doc_max_timesteps, hps.sent_max_len)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=hps.batch_size, shuffle=False)
    del train_dataset
    valid_dataset = ExampleseqSet(valid_file, vocab, hps.doc_max_timesteps, hps.sent_max_len)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=hps.batch_size, shuffle=False)

    if hps.model == "LSTMlstm":
        model = LSTMlstm(hps, embed)
        logger.info("[MODEL] LSTMlstm ")
    elif hps.model == "LSTMpcnn":
        model = LSTMpcnn(hps, embed)
        logger.info("[MODEL] LSTMpcnn ")
    elif hps.model == "MLPCNN":
        model = MLPCNN(hps, embed)
        logger.info("[MODEL]MLPCNN ")
    # elif hps.model == "LSTMcnn":
    #     model = LSTMcnn(hps, embed)
    #     logger.info("[MODEL]LSTMcnn ")
    else:
        logger.error("[ERROR] Invalid Model Type!")
        raise NotImplementedError("Model Type has not been implemented")
    if args.cuda:
        model.to(torch.device("cuda:0"))
        logger.info("[INFO] Use cuda")

    train_loss_list,val_loss_list=setup_training_MAE(model, train_loader, valid_loader, valid_dataset, hps)

    #----------------------------------test-----------------------------------
    # test_dataset = ExampleseqSet(test_file, vocab, hps.doc_max_timesteps, hps.sent_max_len)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=hps.batch_size, shuffle=False)
    # new_model = torch.load('./save/Seq/bestmodel_weibo_MAE'+'_'+hps.model+'_'+str(hps.lstm_layers)+'BiLSTM_'+str(hps.bidirectional))
    # print("The model :" + hps.model + ' with ' + str(hps.lstm_layers) + 'layers')
    # setup_eval_MAE(hps, test_loader, new_model)
    # setup_eval_MSE(hps, test_loader, new_model)
    # setup_eval_RMSE(hps, test_loader, new_model)
if __name__ == '__main__':
    main()
