from torchtext.legacy import data
from torchtext.vocab import Vectors
import torch


def dataLoad(train_data, val_data, test_data, batch_size):
    # print("batch_size=", batch_size)
    tokenize = lambda x: x.split()
    # 这里一定要指定一个fix_length进行padding，因为数据集中有非常短的phrase，比如一个单词，如果不padding，会出现卷积核的h维度大于token数，导致runtimeerror
    TEXT = data.Field(sequential=True, tokenize=tokenize, stop_words='english', fix_length=60)
    LABEL = data.Field(sequential=False, use_vocab=False)

    # 载入数据
    fields = [('PhraseId', None), ('SentenceId', None), ('Phrase', TEXT), ('Sentiment', LABEL)]
    train_data, val_data, test_data = data.TabularDataset.splits(
        path='../data',
        skip_header=True,
        train=train_data,
        validation=val_data,
        test=test_data,
        format='tsv',
        fields=fields
    )

    # 加载glove预训练的词向量进行word Embedding
    vectors = Vectors(name='../glove/glove.6B.200d.txt')
    TEXT.build_vocab(train_data, val_data, test_data, vectors=vectors)
    LABEL.build_vocab(train_data, val_data, test_data)
    weights = TEXT.vocab.vectors  # 嵌入矩阵的初始权重
    # 如果不使用预训练模型进行word embedding, 在搭建网络的时候，指明word_num和embedding_dim进行随机初始化
    # TEXT.build_vocab(train_data, val_data, test_data）
    # LABEL.build_vocab(train_data, val_data, test_data)

    # 设置迭代器
    train_itr, val_itr = data.Iterator.splits(
        (train_data, val_data),
        batch_sizes=(batch_size, batch_size),
        sort_key=lambda x: len(x.Phrase),
        device=-1
    )
    test_itr = data.Iterator(
        test_data,
        batch_size=batch_size,
        sort=False,
        device=-1
    )

    return train_itr, val_itr, test_itr, weights