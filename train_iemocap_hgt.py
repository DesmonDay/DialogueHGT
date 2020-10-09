import time
import argparse
import ipdb
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.metrics import classification_report, precision_recall_fscore_support

from logger import log
from model import DialogueHGTModel
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


def train_or_eval_graph_model(model, loss_func, dataloader, cuda, optimizer=None, train=False):
    losses, preds, labels = [], [], []
    scores, vids = [], []

    assert not train or optimizer is not None
    if train:
        model.train()
    else:
        model.eval()

    seed_everything(args.seed)
    for data in dataloader:
        if train:
            optimizer.zero_grad()
        # Text feature, Visual feature, Audio feature, Speaker mask, Utterance mask, Label
        textf, visualf, audiof, qmask, umask, label = [d.cuda() for d in data[:-1]] if cuda else data[:-1]
        lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in
                   range(len(umask))]  # 每个 batch 中真实的 utterance 数量

        # 只用到了 text feature, need to change.
        
        # ipdb.set_trace()
        log_prob = model(textf, qmask, umask, lengths)
        label = torch.cat([label[j][:lengths[j]] for j in range(len(label))])
        loss = loss_func(log_prob, label)

        preds.append(torch.argmax(log_prob, 1).cpu().numpy())
        labels.append(label.cpu().numpy())
        losses.append(loss.item())

        if train:
            loss.backward()
            optimizer.step()
        else:
            vids += data[-1]

    if preds != []:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
    else:
        return float('nan'), float('nan'), [], [], float('nan'), []

    labels = np.array(labels)
    preds = np.array(preds)
    vids = np.array(vids)

    avg_loss = round(np.sum(losses) / len(losses), 4)
    avg_accuracy = round(accuracy_score(labels, preds) * 100, 2)
    avg_fscore = round(f1_score(labels, preds, average='weighted') * 100, 2)
    return avg_loss, avg_accuracy, labels, preds, avg_fscore, vids


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=100, help='random seed')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='does not use GPU')
    parser.add_argument('--base-model', default='LSTM',
                        help='base recurrent model, must be one of DialogRNN/LSTM/GRU')
    parser.add_argument('--nodal-attention', action='store_true', default=False,
                        help='whether to use nodal attention in graph model: Equation 4,5,6 in Paper')  # Emotion classifier
    parser.add_argument('--windowp', type=int, default=10,
                        help='context window size for constructing edges in graph model for past utterances')
    # 实验点：可用于探究过去的影响大还是未来的影响大
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
    parser.add_argument('--num_layers', type=int, default=2, help='number of gnn layers')
    parser.add_argument('--num_heads', type=int, default=4, help='number of heads in DialogueHGT')
    parser.add_argument('--tensorboard', action='store_true', default=False,
                        help='Enables tensorboard log')

    args = parser.parse_args()
    log.info(args)
    
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    if args.cuda:
        log.info("Running on GPU")
    else:
        log.info("Running on CPU")

    # I can use VisualDL instead.
    if args.tensorboard:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter()

    # 针对 IEMOCAP 数据集特定不变的部分
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

    seed_everything(args.seed)
    model = DialogueHGTModel(args,
                             D_m, D_g, D_p, D_e, D_a, graph_h,
                             n_speakers=2,
                             n_classes=n_classes)
    log.info("Graph NN with %s as base model" % args.base_model)
    name = 'Graph'

    if cuda:
        model.cuda()

    total_params = sum(p.numel() for p in model.parameters())
    log.info('Total parameters: %d' % total_params)
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info('Training parameters: %d' % total_trainable_params) 

    loss_weights = torch.FloatTensor(
        [1 / 0.086747, 1 / 0.144406, 1 / 0.227883, 1 / 0.160585, 1 / 0.127711, 1 / 0.252668])

    if args.class_weight:
        loss_function = nn.NLLLoss(loss_weights.cuda() if cuda else loss_weights)
            # 用于多分类的负对数似然损失函数(negative log likelihood loss)
            # weight权重是一个一维张量，为每个类分配权重；当训练集不平衡时，这是特别有用的。
    else:
        loss_function = nn.NLLLoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    train_loader, valid_loader, test_loader = get_IEMOCAP_loaders(valid=0.0,
                                                                  batch_size=batch_size,
                                                                  num_workers=0)
    best_fscore, best_loss, best_label, best_pred, best_mask = None, None, None, None, None
    all_fscore, all_acc, all_loss = [], [], []

    for e in range(n_epochs):
        start_time = time.time()
        train_loss, train_acc, _, _, train_fscore, _ = \
            train_or_eval_graph_model(model, loss_function, train_loader, cuda, optimizer, True)
        valid_loss, valid_acc, _, _, valid_fscore, _, = \
            train_or_eval_graph_model(model, loss_function, valid_loader, cuda)
        test_loss, test_acc, test_label, test_pred, test_fscore, _ = \
            train_or_eval_graph_model(model, loss_function, test_loader, cuda)
        all_fscore.append(test_fscore)

        if args.tensorboard:
            writer.add_scalar('test: accuracy/loss', test_acc / test_loss, e)
            writer.add_scalar('train: accuracy/loss', train_acc / train_loss, e)

        log.info(
            'epoch: {}, train_loss: {}, train_acc: {}, train_fscore: {}, valid_loss: {}, valid_acc: {}, \
            valid_fscore: {}, test_loss: {}, test_acc: {}, test_fscore: {}, time: {}s'. \
                format(e + 1, train_loss, train_acc, train_fscore, valid_loss, valid_acc, valid_fscore, test_loss,
                       test_acc, test_fscore, round(time.time() - start_time, 2)))
        log.info(classification_report(test_label, test_pred, digits=4)) # 用于记录分类具体结果
        log.info(confusion_matrix(test_label, test_pred))

    if args.tensorboard:
        writer.close()

    log.info('Test performance...')
    log.info('F-Score: %f' % max(all_fscore))
