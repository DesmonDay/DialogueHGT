import time
import ipdb
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr

from logger import log
from model import DialogueHGTModel
from dataloader import AVECDataset
from utils import seed_everything

# TODO: If we add word-level graph, we need to add `masks`.


def get_train_valid_sampler(trainset, valid=0.1):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid * size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])  # 随机采样


def get_AVEC_loaders(path, batch_size=32, valid=0.1, num_workers=0, pin_memory=False):
    trainset = AVECDataset(path=path, train=True)
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

    testset = AVECDataset(path=path, train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader


def train_or_eval_graph_model(model, loss_func, dataloader, cuda, optimizer=None, train=False):
    losses, preds, labels, masks = [], [], [], []
    assert not train or optimizer != None

    if train:
        model.train()
    else:
        model.eval()

    for data in dataloader:
        if train:
            optimizer.zero_grad()
        textf, visualf, audiof, qmask, umask, label = [d.cuda() for d in data] if cuda else data
        lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in
                   range(len(umask))]  # 每个 batch 中真实的 utterance 数量
        # TODO: 如果增加词级别 level，可以增加一个新的 lengths，表示添加词的 features 之后lengths的长度

        # 只用到了 text feature, need to change.
        # ipdb.set_trace()  
        pred = model(textf, qmask, umask, lengths) # True number of utterances
        pred = pred.squeeze()
        labels_ = torch.cat([label[j][:lengths[j]] for j in range(len(label))])
        loss = loss_func(pred, labels_)

        preds.append(pred.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())
        losses.append(loss.item())

        if train:
            loss.backward()
            optimizer.step()

    # ipdb.set_trace()
    if preds != []:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
    else:
        return float('nan'), float('nan'), float('nan'), [], []
    avg_loss = round(np.sum(losses) / len(losses), 4)
    mae = round(mean_absolute_error(labels, preds), 4)
    pred_lab = pd.DataFrame(list(zip(labels, preds)))
    pear = round(pearsonr(pred_lab[0], pred_lab[1])[0], 4)
    return avg_loss, mae, pear, labels, preds


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
    parser.add_argument('--attribute', type=int, default=1, help='AVEC attribute for regression')

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

    # 针对 AVEC 数据集特定部分
    n_classes = 1  # 回归问题
    cuda = args.cuda
    n_epochs = args.epochs
    batch_size = args.batch_size

    D_m = 100  # utterance feature size
    D_g = 150
    D_p = 150
    D_e = 100
    D_a = 100
    graph_h = 100

    seed_everything(args.seed)
    model = DialogueHGTModel(args,
                             D_m, D_g, D_p, D_e, D_a, graph_h,
                             n_speakers=2,
                             n_classes=n_classes,
                             avec=True)
    log.info("Graph NN with %s as base model" % args.base_model)
    name = 'Graph'

    if cuda:
        model.cuda()

    total_params = sum(p.numel() for p in model.parameters())
    log.info('Total parameters: %d' % total_params)
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info('Training parameters: %d' % total_trainable_params)

    # 不需要 class weights, 因为这是回归任务
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    train_loader, valid_loader, test_loader = \
        get_AVEC_loaders('./dataset/AVEC_features/AVEC_features_{}.pkl'.format(args.attribute),
                         valid=0.0,
                         batch_size=batch_size,
                         num_workers=2)
    best_loss, best_label, best_pred, best_pear = None, None, None, None

    for e in range(n_epochs):
        start_time = time.time()
        train_loss, train_mae, train_pear, _, _ = train_or_eval_graph_model(model, loss_function, train_loader, cuda,
                                                                            optimizer, True)
        valid_loss, valid_mae, valid_pear, _, _ = train_or_eval_graph_model(model, loss_function, valid_loader, cuda)
        test_loss, test_mae, test_pear, test_label, test_pred = train_or_eval_graph_model(model, loss_function,
                                                                                          test_loader, cuda)
        if best_loss == None or best_loss > test_loss:
            best_loss, best_label, best_pred, best_pear = \
                test_loss, test_label, test_pred, test_pear

        log.info('epoch: {}, train_loss: {}, train_mae: {}, train_pear: {}, valid_loss: {}, valid mae: {}, '
                 'valid_pear: {}, test_loss: {}, test_mae: {}, test_pear: {}, time: {}'.
                 format(e, train_loss, train_mae, train_pear, valid_loss, valid_mae,
                        valid_pear, test_loss, test_mae, test_pear, round(time.time() - start_time, 2)))

    log.info('Test performance...')
    log.info('MSE: {}, MAE: {}, r: {}'.format(best_loss,
                                              round(mean_absolute_error(best_label, best_pred), 4),
                                              best_pear))


