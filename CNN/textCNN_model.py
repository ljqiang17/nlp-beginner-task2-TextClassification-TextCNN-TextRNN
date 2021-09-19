import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, args):
        super(TextCNN, self).__init__()
        self.args = args

        # 输入通道数
        in_chaneels = 1
        # 类别数
        label_num = args.label_num
        # 卷积核的数量和尺寸
        kernel_num = args.kernel_num  # 等价于输出通道数
        kernel_size = args.kernel_size
        # 用于Embedding的权重矩阵
        weight = args.weight
        # embedding向量维度和embedding词典大小
        self.embedding_dim = weight.size(1)
        num_embeddings = weight.size(0)

        # embedding层，用预训练模型进行Embedding
        # self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding = nn.Embedding.from_pretrained(weight, freeze=False)  # 是否微调
        # 如果随机embedding
        # self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        # 卷积层 输入通道数、输出通道数/每个卷积核的数量，卷积核的大小(卷积核的尺寸，嵌入维度)
        self.convs = nn.ModuleList([nn.Conv2d(in_chaneels, kernel_num, (ks, self.embedding_dim)) for ks in kernel_size])
        # drop层
        self.dropout = nn.Dropout(args.dropout)
        # 全连接层
        self.fullconnection = nn.Linear(len(kernel_size) * kernel_num, label_num)

    def forward(self, x):
        # print("x.shape", x.size())
        # x: (batch_size * max_length)
        # embedding操作，x: (batch_size * max_length * embedding_dim)
        x = self.embedding(x)
        # 在第二个维度增加一个维度 x: (batch_size, channel_num, max_length, embedding_dim)
        # x = x.unsqueeze(1)
        x = x.view(x.size(0), 1, x.size(1), self.embedding_dim)
        # 卷积操作， x:(batch_size, out_channel, width, height=1), width为卷积运算后的向量宽度
        x = [F.relu(conv(x)) for conv in self.convs]
        # 最大池化 x:(batch, out_channel, 1, 1) width经过最大池化为1
        x = [F.max_pool2d(input=x_item, kernel_size=(x_item.size(2), x_item.size(3))) for x_item in x]
        # 展平 x:(batch, (out_channel * 1 * 1 ))
        x = [x_item.view(x_item.size(0), -1) for x_item in x]
        # 特征组合
        x = torch.cat(x, 1)
        # dropout层
        x = self.dropout(x)
        # 全连接层
        logits = self.fullconnection(x)
        # 输出
        return logits

