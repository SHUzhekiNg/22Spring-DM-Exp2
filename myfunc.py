import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self):
        super(TextCNN, self).__init__()
        self.num_classes = None
        self.vocab_size = None
        self.embedding_size = 2

        self.data_loader()
        self.W = nn.Embedding(self.vocab_size, self.embedding_size)
        output_channel = 3
        self.conv = nn.Sequential(nn.Conv2d(1, output_channel, (2, self.embedding_size)),
                                  # inpu_channel, output_channel, 卷积核高和宽 n-gram 和 embedding_size
                                  nn.ReLU(),
                                  nn.MaxPool2d((2, 1)))
        self.fc = nn.Linear(output_channel, self.num_classes)

    def data_loader(self):
        # 3 words sentences (=sequence_length is 3)
        sentences = ["i love you", "he loves me", "she likes baseball", "i hate you", "sorry for that", "this is awful"]
        labels = [1, 1, 1, 0, 0, 0]  # 1 is good, 0 is not good.

        sequence_length = len(sentences[0])
        self.num_classes = len(set(labels))
        batch_size = 3

        word_list = " ".join(sentences).split()
        vocab = list(set(word_list))
        self.word2idx = {w: i for i, w in enumerate(vocab)}
        self.vocab_size = len(vocab)

        inputs = []
        for sen in sentences:
            inputs.append([self.word2idx[n] for n in sen.split()])
        targets = []
        for out in labels:
            targets.append(out)

        print(inputs)
        print(targets)
        input_batch, target_batch = torch.LongTensor(inputs), torch.LongTensor(targets)
        dataset = Data.TensorDataset(input_batch, target_batch)
        loader = Data.DataLoader(dataset, batch_size, True)
        return loader

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


