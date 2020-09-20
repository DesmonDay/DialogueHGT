import pickle
import pandas as pd

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class IEMOCAPDataset(Dataset):
    """
    IEMOCAP Dataset.
    """
    def __init__(self, train=True):
        # 其中，self.videoSentence 是原始的文本数据，可以用于做第一个改进点——构图有问题，需要多思考思考
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText, \
        self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid, \
        self.testVid = pickle.load(open('./dataset/IEMOCAP_features.pkl', 'rb'), encoding='latin1')

        # label index mapping = {'hap':0, 'sad':1, 'neu':2, 'ang':3, 'exc':4, 'fru':5}

        self.keys = [x for x in (self.trainVid if train else self.testVid)]
        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.videoText[vid]),\
               torch.FloatTensor(self.videoVisual[vid]), \
               torch.FloatTensor(self.videoAudio[vid]), \
               torch.FloatTensor([[1, 0] if x == 'M' else [0,1] for x in self.videoSpeakers[vid]]), \
               torch.FloatTensor([1] * len(self.videoLabels[vid])), \
               torch.LongTensor(self.videoLabels[vid]), \
               vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):  # 对 Tensor 进行 padding，使得每个 batch 对应的 utterance 数目相同
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i < 4 else pad_sequence(dat[i], True) if i < 6 else dat[i].tolist() for i in dat]


class MELDDataset(Dataset):
    """
    MELD Dataset.
    """
    def __init__(self, path, classify, train=True):
        self.videoIDs, self.videoSpeakers, self.videoLabelsEmotion, self.videoText, \
        self.videoAudio, self.videoSentence, self.trainVid, \
        self.testVid, self.videoLabelsSentiment = pickle.load(open(path, 'rb'))

        if classify == 'emotion':
            self.videoLabels = self.videoLabelsEmotion
        else:
            self.videoLabels = self.videoLabelsSentiment
        '''
        label index mapping = {'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3, 'joy': 4, 'disgust': 5, 'anger':6}
        '''
        self.keys = [x for x in (self.trainVid if train else self.testVid)]

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        # 少了 videoVisual 的特征
        return torch.FloatTensor(self.videoText[vid]), \
               torch.FloatTensor(self.videoAudio[vid]), \
               torch.FloatTensor(self.videoSpeakers[vid]), \
               torch.FloatTensor([1] * len(self.videoLabels[vid])), \
               torch.LongTensor(self.videoLabels[vid]), \
               vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i < 3 else pad_sequence(dat[i], True) if i < 5 else dat[i].tolist() for i in
                dat]


class AVECDataset(Dataset):
    """
    AVEC Dataset
    """
    def __init__(self, path, train=True):
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText, \
        self.videoAudio, self.videoVisual, self.videoSentence, \
        self.trainVid, self.testVid = pickle.load(open(path, 'rb'), encoding='latin1')

        self.keys = [x for x in (self.trainVid if train else self.testVid)]

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.videoText[vid]), \
               torch.FloatTensor(self.videoVisual[vid]), \
               torch.FloatTensor(self.videoAudio[vid]), \
               torch.FloatTensor([[1, 0] if x == 'user' else [0, 1] for x in \
                                  self.videoSpeakers[vid]]), \
               torch.FloatTensor([1] * len(self.videoLabels[vid])), \
               torch.FloatTensor(self.videoLabels[vid])

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i < 4 else pad_sequence(dat[i], True) for i in dat]





