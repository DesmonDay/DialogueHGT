import numpy as np
import itertools
import random
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.nn import RGCNConv, GraphConv

from logger import log
from conv import HGTConv
from hgt_utils import hgt_batch_graphify
from utils import pad, edge_perms, batch_graphify, attentive_node_features, classify_node_features
from attention import SimpleAttention, MatchingAttention, MaskedEdgeAttention


######### DialogueRNN #########
class DialogueRNNCell(nn.Module):
    def __init__(self, D_m, D_g, D_p, D_e, listener_state=False,
                 context_attention='simple', D_a=100, dropout=0.5):
        super(DialogueRNNCell, self).__init__()

        self.D_m = D_m
        self.D_g = D_g
        self.D_p = D_p
        self.D_e = D_e

        self.listener_state = listener_state
        # input_size：输入数据X的特征值的数目。
        # hidden_size：隐藏层的神经元数量，也就是隐藏层的特征数量。
        self.g_cell = nn.GRUCell(D_m + D_p, D_g)
        self.p_cell = nn.GRUCell(D_m + D_g, D_p)
        self.e_cell = nn.GRUCell(D_p, D_e)

        if listener_state:  # Listener Update part in paper.
            self.l_cell = nn.GRUCell(D_m + D_p, D_p)

        self.dropout = nn.Dropout(dropout)

        if context_attention == 'simple':
            self.attention = SimpleAttention(D_g)
        else:
            self.attention = MatchingAttention(D_g, D_m, D_a, context_attention)

    def _select_parties(self, X, indices):
        pass

    def forward(self, U, qmask, g_hist, q0, e0):
        pass


class DialogueRNN(nn.Module):
    def __init__(self, D_m, D_g, D_p, D_e, listener_state=False,
                 context_attention='simple', D_a=100, dropout=0.5):
        super(DialogueRNN, self).__init__()
        # D_m: utterance representation size
        # D_g: size of global state vector
        # D_p: size of party state vector
        # D_e: size of emotion representation vector
        self.D_m = D_m
        self.D_g = D_g
        self.D_p = D_p
        self.D_e = D_e
        self.dropout = nn.Dropout(dropout)

        self.dialogue_cell = DialogueRNNCell(D_m, D_g, D_p, D_e,
                                             listener_state, context_attention, D_a, dropout)


######### DialogueGCN #########
class GraphNet(nn.Module):
    def __init__(self, num_features, num_classes, num_relations, max_seq_len,
                 hidden_size=64, dropout=0.5, no_cuda=False):
        """
        The Speaker-level context encoder in the form of a 2 layer GCN.
        """
        super(GraphNet, self).__init__()
        self.conv1 = RGCNConv(num_features, hidden_size, num_relations, num_bases=30)
        self.conv2 = GraphConv(hidden_size, hidden_size)
        self.matchatt = MatchingAttention(num_features + hidden_size, num_features + hidden_size, att_type='general2')
        self.linear = nn.Linear(num_features + hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.softmax_fc = nn.Linear(hidden_size, num_classes)
        self.no_cuda = no_cuda

    def forward(self, x, edge_index, edge_norm, edge_type, seq_lengths, umask, nodal_attn, avec):
        out = self.conv1(x, edge_index, edge_type, edge_norm)
        out = self.conv2(out, edge_index)
        emotions = torch.cat([x, out], dim=-1)
        log_prob = classify_node_features(emotions, seq_lengths, umask, self.matchatt, self.linear, self.dropout,
                                          self.softmax_fc, nodal_attn, avec, self.no_cuda)
        return log_prob


class DialogueGCNModel(nn.Module):
    # D_m: 文本特征维度
    # D_e: 通过 sequential context encoder 后的维度，双向，2 * D_e
    def __init__(self, base_model, D_m, D_g, D_p, D_e, D_h, D_a,
                 graph_hidden_size, n_speakers, max_seq_len,
                 window_past, window_future, n_classes=7,
                 listener_state=False, context_attention='simple',
                 dropout_rec=0.5, dropout=0.5, nodal_attention=True,
                 avec=False, no_cuda=False):

        super(DialogueGCNModel, self).__init__()
        self.base_model = base_model
        self.avec = avec
        self.no_cuda = no_cuda

        # The base model is the sequential context encoder.
        if self.base_model == 'DialogRNN':
            self.dialog_rnn_f = DialogueRNN(D_m, D_g, D_p, D_e, listener_state, context_attention, D_a, dropout_rec)
            self.dialog_rnn_r = DialogueRNN(D_m, D_g, D_p, D_e, listener_state, context_attention, D_a, dropout_rec)
        elif self.base_model == 'LSTM':
            self.lstm = nn.LSTM(input_size=D_m, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)
        elif self.base_model == 'GRU':
            self.gru = nn.GRU(input_size=D_m, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)
        elif self.base_model == 'None':
            self.base_linear = nn.Linear(D_m, 2 * D_e)
        else:
            log.info('Base model must be one of DialogRNN/LSTM/GRU')
            raise NotImplementedError

        n_relations = 2 * (n_speakers ** 2)  # 例如，n_speakers=2，则 n_relations=8. 表示 relation types.
        self.window_past = window_past
        self.window_future = window_future

        self.att_model = MaskedEdgeAttention(2 * D_e, max_seq_len, self.no_cuda)
        self.nodal_attention = nodal_attention
        self.graph_net = GraphNet(2 * D_e, n_classes, n_relations, max_seq_len,
                                  graph_hidden_size, dropout, self.no_cuda)

        edge_type_mapping = {}  # 边类型
        for j in range(n_speakers):
            for k in range(n_speakers):
                edge_type_mapping[str(j) + str(k) + '0'] = len(edge_type_mapping)
                edge_type_mapping[str(j) + str(k) + '1'] = len(edge_type_mapping)
        self.edge_type_mapping = edge_type_mapping

    def _reverse_seq(self, X, mask):  ## 这个函数需要理解，主要是用于 DialogueRNN 当中
        """
        X -> seq_len, batch, dim
        mask -> batch, seq_len
        """
        X_ = X.transpose(0, 1)
        mask_sum = torch.sum(mask, 1).int()

        xfs = []
        for x, c in zip(X_, mask_sum):
            xf = torch.flip(x[:c], [0])
            xfs.append(xf)

        return pad_sequence(xfs)

    def forward(self, U, qmask, umask, seq_lengths):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """

        # 这里的U是文本特征。需要添加与语音特征的交互模块。
        """
        U -> U + audiof
        """

        if self.base_model == "DialogRNN":
            if self.avec:
                emotions, _ = self.dialog_rnn_f(U, qmask)
            else:
                emotions_f, alpha_f = self.dialog_rnn_f(U, qmask)
                rev_U = self._reverse_seq(U, umask)
                rev_qmask = self._reverse_seq(qmask, umask)
                emotions_b, alpha_b = self.dialog_rnn_r(rev_U, rev_qmask)
                emotions_b = self._reverse_seq(emotions_b, umask)
                emotions = torch.cat([emotions_f, emotions_b], dim=-1)

        elif self.base_model == "LSTM":
            emotions, hidden = self.lstm(U)  # seq_len, batch, D_e

        elif self.base_model == "GRU":
            emotions, hidden = self.gru(U)

        elif self.base_model == "None":
            emotions = self.base_linear(U)

        features, edge_index, edge_norm, edge_type, edge_index_lengths = \
            batch_graphify(emotions, qmask, seq_lengths, self.window_past,
                           self.window_future, self.edge_type_mapping,
                           self.att_model, self.no_cuda)
        log_prob = self.graph_net(features, edge_index, edge_norm, edge_type, seq_lengths, umask,
                                  self.nodal_attention, self.avec)

        return log_prob, edge_index, edge_norm, edge_type, edge_index_lengths


class HGTNet(nn.Module):
    def __init__(self, in_dim, n_hid, num_types, num_relations, n_heads, n_layers, dropout=0.2, pre_norm=False,
                 last_norm=False, use_RTE=False):
        super(HGTNet, self).__init__()
        self.hgts = nn.ModuleList()
        self.num_types = num_types  # 相当于 speakers 数量
        self.in_dim = in_dim
        self.n_hid = n_hid
        self.adapt_ws = nn.ModuleList()
        self.drop = nn.Dropout(dropout)
        for t in range(num_types):
            self.adapt_ws.append(nn.Linear(in_dim, n_hid))
        for _ in range(n_layers - 1):
            self.hgts.append(HGTConv(n_hid, n_hid, num_types, num_relations, n_heads, dropout,
                                     use_norm=pre_norm, use_RTE=use_RTE))
        self.hgts.append(HGTConv(n_hid, n_hid, num_types, num_relations, n_heads, dropout,
                                 use_norm=last_norm, use_RTE=use_RTE))

    def forward(self, node_features, node_type, edge_index, edge_type):
        res = torch.zeros(node_features.size(0), self.n_hid).to(node_features.device)
        for t_id in range(self.num_types):
            idx = (node_type == int(t_id))
            if idx.sum() == 0:
                continue
            res[idx] = torch.tanh(self.adapt_ws[t_id](node_features[idx]))
        meta_xs = self.drop(res)
        del res
        for hgt in self.hgts:
            meta_xs = hgt(meta_xs, node_type, edge_index, edge_type)
        return meta_xs


class DialogueHGTModel(nn.Module):
    """
    Implementation of DialogueHGT Model.
    """
    def __init__(self, base_model, D_m, D_g, D_p, D_e, D_h, D_a,
                 graph_hidden_size, n_speakers, max_seq_len,
                 window_past, window_future, n_classes=7,
                 listener_state=False, context_attention='simple',
                 dropout_rec=0.5, dropout=0.5, nodal_attention=True,
                 avec=False, no_cuda=False):

        super(DialogueHGTModel, self).__init__()
        self.base_model = base_model
        self.avec = avec
        self.no_cuda = no_cuda

        # Sequential Encoder
        if self.base_model == 'DialogRNN':
            self.dialog_rnn_f = DialogueRNN(D_m, D_g, D_p, D_e, listener_state, context_attention, D_a, dropout_rec)
            self.dialog_rnn_r = DialogueRNN(D_m, D_g, D_p, D_e, listener_state, context_attention, D_a, dropout_rec)
        elif self.base_model == 'LSTM':
            self.lstm = nn.LSTM(input_size=D_m, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)
        elif self.base_model == 'GRU':
            self.gru = nn.GRU(input_size=D_m, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)
        elif self.base_model == 'None':
            self.base_linear = nn.Linear(D_m, 2 * D_e)
        else:
            log.error('Base model must be one of DialogRNN/LSTM/GRU')
            raise NotImplementedError

        self.nodal_attention = nodal_attention
        self.window_past = window_past
        self.window_future = window_future

        # 暂不添加其他 attention 结构，只将图结构经过 HGT模型
        # Graph Encoder
        self.graphnet = HGTNet(2 * D_e, graph_hidden_size, n_speakers, num_relations=2,
                               n_heads=4, n_layers=2, dropout=0.2)
        self.matchatt = MatchingAttention(2 * D_e + graph_hidden_size, 2 * D_e + graph_hidden_size, att_type='general2')
        self.linear = nn.Linear(2 * D_e + graph_hidden_size, graph_hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.softmax_fc = nn.Linear(graph_hidden_size, n_classes)

    def _reverse_seq(self, X, mask):  ## 这个函数需要理解，主要是用于 DialogueRNN 当中
        """
        X -> seq_len, batch, dim
        mask -> batch, seq_len
        """
        X_ = X.transpose(0, 1)
        mask_sum = torch.sum(mask, 1).int()

        xfs = []
        for x, c in zip(X_, mask_sum):
            xf = torch.flip(x[:c], [0])
            xfs.append(xf)

        return pad_sequence(xfs)

    def forward(self, U, qmask, umask, seq_lengths):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """
        if self.base_model == 'DialogRNN':
            if self.avec:
                emotions, _ = self.dialog_rnn_f(U, qmask)
            else:
                emotions_f, alpha_f = self.dialog_rnn_f(U, qmask)
                rev_U = self._reverse_seq(U, umask)
                rev_qmask = self._reverse_seq(qmask, umask)
                emotions_b, alpha_b = self.dialog_rnn_r(rev_U, rev_qmask)
                emotions_b = self._reverse_seq(emotions_b, umask)
                emotions = torch.cat([emotions_f, emotions_b], dim=-1)
        elif self.base_model == 'LSTM':
            emotions, hidden = self.lstm(U)  # seq_len, batch, D_e
        elif self.base_model == 'GRU':
            emotions, hidden = self.gru(U)
        elif self.base_model == 'None':
            emotions = self.base_linear(U)

        # Construct graph.
        features, node_type, edge_index, edge_type, edge_index_lengths = \
            hgt_batch_graphify(emotions, qmask, seq_lengths, self.window_past,
                           self.window_future, self.no_cuda)
        out = self.graphnet(features, node_type, edge_index, edge_type)
        # Classification layer.
        emotions = torch.cat([features, out], dim=-1)
        log_prob = classify_node_features(emotions, seq_lengths, umask, self.matchatt, self.linear, self.dropout,
                                          self.softmax_fc, self.nodal_attention, self.avec, self.no_cuda)
        return log_prob

