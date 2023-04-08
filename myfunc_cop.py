import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torch.nn.functional as F
import json
from collections import Counter
import jieba
from tqdm import tqdm


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_size, num_classes):
        super(TextCNN, self).__init__()
        self.vocab_size = None
        self.W = nn.Embedding(vocab_size, embedding_size)
        output_channel = 20
        self.conv = nn.Sequential(nn.Conv2d(1, output_channel, (2, embedding_size)),
                                  # inpu_channel, output_channel, 卷积核高和宽 n-gram 和 embedding_size
                                  nn.ReLU(),
                                  nn.MaxPool2d((output_channel-1, 1)))
        self.fc = nn.Linear(output_channel, num_classes)

    def forward(self, X):
        '''
        X: [batch_size, sequence_length]
        '''
        batch_size = X.shape[0]
        embedding_X = self.W(X)  # [batch_size, sequence_length, embedding_size]
        embedding_X = embedding_X.unsqueeze(1)  # add channel(=1) [batch, channel(=1), sequence_length, embedding_size]
        conved = self.conv(embedding_X)  # [batch_size, output_channel,1,1]
        flatten = conved.view(batch_size, -1)  # [batch_size, output_channel*1*1]
        output = self.fc(flatten)
        return output


def data_loader(is_train=True):
    test_dataset = load_data_from_json("./tnews_public/test.json")
    train_dataset = load_data_from_json("./tnews_public/train.json")
    # print(len(test_dataset))
    # print(test_dataset[:3])

    # print(len(train_dataset))
    # print(train_dataset[:3])

    dataset = train_dataset + test_dataset

    word2idx, idx2word = make_data(dataset)
    vocab_size = len(word2idx)
    # targets = []
    # for out in labels:
    # targets.append(out) return inputs, targets
    batch_size = 4
    if is_train:
        inputs, labels = serialization(train_dataset, word2idx)
    else:
        inputs, labels = serialization(test_dataset, word2idx)
    # print(len(labels))
    # print(len(inputs))

    input_batch, target_batch = torch.LongTensor(inputs), torch.LongTensor(labels)
    dataset = Data.TensorDataset(input_batch, target_batch)
    loader = Data.DataLoader(dataset, batch_size, True)
    return loader, vocab_size


def padding_and_cut(idx_seq, max_length):
    # padding 补零
    if len(idx_seq) < max_length:
        idx_seq += [0]*(max_length-len(idx_seq))
    else:
        # 截取
        idx_seq = idx_seq[:max_length]
    return idx_seq


def load_data_from_json(file_path):
    dataset = []
    for line in open(file_path, "r" , encoding='utf-8'):
        instance = json.loads(line)
        sent = [x for x in jieba.cut(instance['sentence'])]
        label = instance['label']
        dataset.append((sent, label))
    return dataset


def make_data(dataset):
    counter = Counter()
    for (sent, label) in dataset:
        counter.update(sent)
    # sorted(iterable, cmp=None, key=None, reverse=False)
    vocab = [(word, count) for (word, count) in counter.items() if count > 1]
    sorted_vocab = sorted(vocab, key=lambda x: x[1], reverse=False)
    word2idx = {'unk': 0}
    word2idx.update({word: i + 1 for i, (word, count) in enumerate(sorted_vocab)})
    idx2word = {i: word for (word, i) in word2idx.items()}

    return word2idx, idx2word
    # inputs.appendword2idx[n]([ for n in sen.split()])


def serialization(dataset, word2idx, max_length=20):
    input_ids = []
    label_ids = []
    for sent, label in dataset:
        idx_seq = []
        for word in sent:
            if word in word2idx:
                idx = word2idx[word]
            else:
                idx = word2idx["unk"]
            idx_seq.append(idx)
        input_ids.append(padding_and_cut(idx_seq, max_length))
        label_ids.append(int(label) - 100)
    return input_ids, label_ids


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division
    acc = correct.sum() / len(correct)
    return acc
