import time
import pickle
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import classification_report, precision_recall_fscore_support

from logger import log
from model import DialogueGCNModel
from dataloader import IEMOCAPDataset
from loss import MaskedNLLLoss
from utils import seed_everything


def get_train_valid_sampler(trainset, valid=0.1):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid * size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])  # 随机采样


def get_IEMOCAP_loaders(batch_size=32, valid=0.1, num_workers=0, pin_memory=False):
    trainset = IEMOCAPDataset(train=True)
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = IEMOCAPDataset(train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader


def train_or_eval_model(model, loss_func, dataloader, epoch, optimizer=None, train=False):
    losses, preds, labels, masks = [], [], [], []
    alphas, alphas_f, alphas_b, vids = [], [], [], []
    max_sequence_len = []

    assert not train or optimizer != None
    if train:
        model.train()
    else:
        model.eval()

    seed_everything(seed=args.seed)
    for data in dataloader:
        if train:
            optimizer.zero_grad()

        textf, visualf, audiof, qmask, umask, label = [d.cuda() for d in data[:-1]] if cuda else data[:-1]
        max_sequence_len.append(textf.size(0))

        log_prob, alpha, alpha_f, alpha_b, _ = model(textf, qmask, umask)
        lp_ = log_prob.transpose(0, 1).contiguous().view(-1, log_prob.size()[2])
        labels_ = label.view(-1)  # batch*seq_len
        loss = loss_func(lp_, labels_, umask)

        pred_ = torch.argmax(lp_, 1)  # batch*seq_len
        preds.append(pred_.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())
        masks.append(umask.view(-1).cpu().numpy())

        losses.append(loss.item() * masks[-1].sum())
        if train:
            loss.backward()
            if args.tensorboard:
                for param in model.named_parameters():
                    writer.add_histogram(param[0], param[1].grad, epoch)
            optimizer.step()
        else:
            alphas += alpha
            alphas_f += alpha_f
            alphas_b += alpha_b
            vids += data[-1]

    if preds is not []:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks = np.concatenate(masks)
    else:
        return float('nan'), float('nan'), [], [], [], float('nan'), []

    avg_loss = round(np.sum(losses) / np.sum(masks), 4)
    avg_accuracy = round(accuracy_score(labels, preds, sample_weight=masks) * 100, 2)
    avg_fscore = round(f1_score(labels, preds, sample_weight=masks, average='weighted') * 100, 2)
    return avg_loss, avg_accuracy, labels, preds, masks, avg_fscore, [alphas, alphas_f, alphas_b, vids]


def train_or_eval_graph_model(model, loss_func, dataloader, epoch, cuda, optimizer=None, train=False):
    losses, preds, labels = [], [], []
    scores, vids = [], []

    ei, et, en, el = torch.empty(0).type(torch.LongTensor), torch.empty(0).type(torch.LongTensor), torch.empty(0), []

    if cuda:
        ei, et, en = ei.cuda(), et.cuda(), en.cuda()

    assert not train or optimizer is not None
    if train:
        model.train()
    else:
        model.eval()

    seed_everything(seed=args.seed)
    for data in dataloader:
        if train:
            optimizer.zero_grad()
        # Text feature, Visual feature, Audio feature, Speaker mask, Utterance mask, Label
        textf, visualf, audiof, qmask, umask, label = [d.cuda() for d in data[:-1]] if cuda else data[:-1]
        lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in
                   range(len(umask))]  # 每个 batch 中真实的 utterance 数量

        # 只用到了 text feature
        log_prob, e_i, e_n, e_t, e_l = model(textf, qmask, umask, lengths)  # what is e_i, e_n, e_t, e_l??
        label = torch.cat([label[j][:lengths[j]] for j in range(len(label))])
        loss = loss_func(log_prob, label)

        # 这里不理解，到底是什么东西呢？
        ei = torch.cat([ei, e_i], dim=1)
        et = torch.cat([et, e_t])
        en = torch.cat([en, e_n])
        el += e_l

        preds.append(torch.argmax(log_prob, 1).cpu().numpy())
        labels.append(label.cpu().numpy())
        losses.append(loss.item())

        if train:
            loss.backward()
            if args.tensorboard:
                for param in model.named_parameters():
                    writer.add_histogram(param[0], param[1].grad, epoch)
            optimizer.step()

    if preds != []:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
    else:
        return float('nan'), float('nan'), [], [], float('nan'), [], [], [], [], []

    vids += data[-1]
    ei = ei.data.cpu().numpy()
    et = et.data.cpu().numpy()
    en = en.data.cpu().numpy()
    el = np.array(el)
    labels = np.array(labels)
    preds = np.array(preds)
    vids = np.array(vids)

    avg_loss = round(np.sum(losses) / len(losses), 4)
    avg_accuracy = round(accuracy_score(labels, preds) * 100, 2)
    avg_fscore = round(f1_score(labels, preds, average='weighted') * 100, 2)

    return avg_loss, avg_accuracy, labels, preds, avg_fscore, vids, ei, et, en, el


if __name__ == "__main__":

    path = './saved/IEMOCAP/'
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='does not use GPU')
    parser.add_argument('--base-model', default='LSTM',
                        help='base recurrent model, must be one of DialogRNN/LSTM/GRU')
    parser.add_argument('--graph-model', action='store_true', default=False,
                        help='whether to use graph model after recurrent encoding')
    parser.add_argument('--nodal-attention', action='store_true', default=False,
                        help='whether to use nodal attention in graph model: Equation 4,5,6 in Paper')  # Emotion classifier
    parser.add_argument('--windowp', type=int, default=10,
                        help='context window size for constructing edges in graph model for past utterances')
    parser.add_argument('--windowf', type=int, default=10,
                        help='context window size for constructing edges in graph model for future utterances')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate')
    parser.add_argument('--l2', type=float, default=0.00001,
                        help='L2 regularization weight')
    parser.add_argument('--rec-dropout', type=float, default=0.1,
                        help='rec_dropout rate')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='batch size')
    parser.add_argument('--epochs', type=int, default=60,
                        help='number of epochs')
    parser.add_argument('--class-weight', action='store_true', default=False,
                        help='use class weights')
    parser.add_argument('--active-listener', action='store_true', default=False,
                        help='activate listener')
    parser.add_argument('--attention', default='general',
                        help='Attention type in DialogueRNN model')
    parser.add_argument('--tensorboard', action='store_true', default=False,
                        help='Enables tensorboard log')
    parser.add_argument('--seed', type=int, default=100, help='random seed')

    args = parser.parse_args()
    log.info(args)

    args.cuda = torch.cuda.is_available() and not args.no_cuda
    if args.cuda:
        log.info("Running on GPU")
    else:
        log.info("Running on CPU")

    if args.tensorboard:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter()

    n_classes = 6  # 6个情感分类，针对 IEMOCAP
    cuda = args.cuda
    n_epochs = args.epochs
    batch_size = args.batch_size

    D_m = 100  # utterance feature size
    D_g = 150
    D_p = 150
    D_e = 100
    D_h = 100
    D_a = 100
    graph_h = 100

    if args.graph_model:
        seed_everything(seed=args.seed)
        model = DialogueGCNModel(args.base_model,
                                 D_m, D_g, D_p, D_e, D_h, D_a, graph_h,
                                 n_speakers=2,
                                 max_seq_len=110,  # 这个是指 utterances 的最大条数吗？
                                 window_past=args.windowp,
                                 window_future=args.windowf,
                                 n_classes=n_classes,
                                 listener_state=args.active_listener,
                                 context_attention=args.attention,
                                 dropout=args.dropout,
                                 nodal_attention=args.nodal_attention,
                                 no_cuda=args.no_cuda)
        log.info("Graph NN with %s as base model" % args.base_model)
        name = 'Graph'
    else:
        if args.base_model == 'DialogueRNN':
            pass
            log.info("Basic Dialogue RNN Model.")
        elif args.base_model == 'GRU':
            pass
            log.info("Basic GRU Model.")
        elif args.base_model == 'LSTM':
            pass
            log.info("Basic LSTM Model.")
        else:
            log.error("Base model must be one of DialogueRNN/LSTM/GRU")
            raise NotImplementedError
        name = 'Base'

    if cuda:
        model.cuda()

    loss_weights = torch.FloatTensor(
        [1 / 0.086747, 1 / 0.144406, 1 / 0.227883, 1 / 0.160585, 1 / 0.127711, 1 / 0.252668])

    if args.class_weight:
        if args.graph_model:
            loss_function = nn.NLLLoss(loss_weights.cuda() if cuda else loss_weights)
            # 用于多分类的负对数似然损失函数(negative log likelihood loss)
            # weight权重是一个一维张量，为每个类分配权重；当训练集不平衡时，这是特别有用的。
        else:
            loss_function = MaskedNLLLoss(loss_weights.cuda() if cuda else loss_weights)
    else:
        if args.graph_model:
            loss_function = nn.NLLLoss()
        else:
            loss_function = MaskedNLLLoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    train_loader, valid_loader, test_loader = get_IEMOCAP_loaders(valid=0.0,
                                                                  batch_size=batch_size,
                                                                  num_workers=0)

    best_fscore, best_loss, best_label, best_pred, best_mask = None, None, None, None, None
    all_fscore, all_acc, all_loss = [], [], []

    for e in range(n_epochs):
        start_time = time.time()

        if args.graph_model:
            train_loss, train_acc, _, _, train_fscore, _, _, _, _, _ = \
                train_or_eval_graph_model(model, loss_function, train_loader, e, cuda, optimizer, True)
            valid_loss, valid_acc, _, _, valid_fscore, _, _, _, _, _ = \
                train_or_eval_graph_model(model, loss_function, valid_loader, e, cuda)
            test_loss, test_acc, test_label, test_pred, test_fscore, _, _, _, _, _ = \
                train_or_eval_graph_model(model, loss_function, test_loader, e, cuda)
            all_fscore.append(test_fscore)
            # torch.save({'model_state_dict': model.state_dict()},
            #            path + name + args.base_model + '_' + str(e) + '.pkl')
        else:
            pass

        if args.tensorboard:
            writer.add_scalar('test: accuracy/loss', test_acc / test_loss, e)
            writer.add_scalar('train: accuracy/loss', train_acc / train_loss, e)

        log.info(
            'epoch: {}, train_loss: {}, train_acc: {}, train_fscore: {}, valid_loss: {}, valid_acc: {}, \
            valid_fscore: {}, test_loss: {}, test_acc: {}, test_fscore: {}, time: {}s'. \
                format(e + 1, train_loss, train_acc, train_fscore, valid_loss, valid_acc, valid_fscore, test_loss,
                       test_acc, test_fscore, round(time.time() - start_time, 2)))

    if args.tensorboard:
        writer.close()

    log.info('Test performance...')
    log.info('F-Score: %f' % max(all_fscore))
