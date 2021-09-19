import torch
import torch.nn as nn


class TextRNN(nn.Module):
    def __init__(self, args):
        super(TextRNN, self).__init__()
        # 类别数
        label_num = args.label_num
        # 用于embedding的权重矩阵
        weight = args.weight
        # 字典的词数和Embedding维数
        embedding_dim = weight.size(1)
        num_embeddings = weight.size(0)
        # 循环网络的种类
        self.rnn_type = args.rnn_type
        # 隐层的维度
        self.hidden_size = args.hidden_size
        # 循环神经网络的层数
        self.num_layers = args.num_layers
        # 是否使用双向
        self.bidirectional = args.bidirectional

        # Embedding层, 使用预训练的词向量进行word Embedding
        self.embedding = nn.Embedding.from_pretrained(weight, freeze=False)
        # LSTM
        if self.rnn_type == 'rnn':
            self.rnn = nn.RNN(input_size=embedding_dim,
                              hidden_size=self.hidden_size,
                              num_layers=self.num_layers,
                              batch_first=True,
                              bidirectional=self.bidirectional)
        elif self.rnn_type == 'lstm':
            self.lstm = nn.LSTM(input_size=embedding_dim,
                                hidden_size=self.hidden_size,
                                num_layers=self.num_layers,
                                batch_first=True,
                                bidirectional=self.bidirectional)
        # 全连接层
        if self.bidirectional:
            self.fullconnection = nn.Linear(self.hidden_size * 2, label_num)
        else:
            self.fullconnection = nn.Linear(self.hidden_size, label_num)

    def forward(self, x):
        # word embedding x: (batch_size, max_len) to x: (batch_size, max_len, embedding_dim)
        x = self.embedding(x)

        # 隐层
        if self.rnn_type == 'rnn':
            if self.bidirectional:
                h0 = torch.randn(self.num_layers * 2, x.size(0), self.hidden_size)
            else:
                h0 = torch.randn(self.num_layers, x.size(0), self.hidden_size)
            out, hn = self.rnn(x, h0)
        elif self.rnn_type == 'lstm':
            if self.bidirectional:
                h0 = torch.randn(self.num_layers * 2, x.size(0), self.hidden_size)
                c0 = torch.randn(self.num_layers * 2, x.size(0), self.hidden_size)
            else:
                h0 = torch.randn(self.num_layers, x.size(0), self.hidden_size)
                c0 = torch.randn(self.num_layers, x.size(0), self.hidden_size)
            out, (hn, cn) = self.lstm(x, (h0, c0))

        # 全连接层
        logits = self.fullconnection(out[:, -1, :])

        return logits