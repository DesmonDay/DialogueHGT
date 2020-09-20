import torch
import torch.nn as nn


class MaskedNLLLoss(nn.Module):
    # 用于多分类的负对数似然损失函数(negative log likelihood loss)
    # NLLLoss 函数输入 input 之前，需要对 input 进行 log_softmax 处理, 即将 input 转换成概率分布的形式
    def __init__(self, weight=None):
        super(MaskedNLLLoss, self).__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight, reduction='sum')

    def forward(self, pred, target, mask):
        """
        pred -> batch*seq_len, n_classes
        target -> batch*seq_len
        mask -> batch, seq_len
        """
        mask_ = mask.view(-1, 1)  # batch*seq_len, 1
        if type(self.weight) == type(None):
            loss = self.loss(pred * mask_, target) / torch.sum(mask)
        else:
            loss = self.loss(pred * mask_, target) \
                   / torch.sum(self.weight[target] * mask_.squeeze())
        return loss


class MaskedMSELoss(nn.Module):

    def __init__(self):
        super(MaskedMSELoss, self).__init__()
        self.loss = nn.MSELoss(reduction='sum')

    def forward(self, pred, target, mask):
        """
        pred -> batch*seq_len
        target -> batch*seq_len
        mask -> batch*seq_len
        """
        loss = self.loss(pred * mask, target) / torch.sum(mask)
        return loss


class UnMaskedWeightedNLLLoss(nn.Module):

    def __init__(self, weight=None):
        super(UnMaskedWeightedNLLLoss, self).__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight, reduction='sum')

    def forward(self, pred, target):
        """
        pred -> batch * seq_len, n_classes
        target -> batch * seq_len
        """
        if type(self.weight) == type(None):
            loss = self.loss(pred, target)
        else:
            loss = self.loss(pred, target) / torch.sum(self.weight[target])
        return loss