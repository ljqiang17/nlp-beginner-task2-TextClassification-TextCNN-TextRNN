# nlp-beginner-task2-TextClassification-TextCNN-TextRNN
基于深度学习的文本分类，实现基于CNN和RNN的文本分类

### 一、问题描述

1. 实现基于深度学习的文本分类，使用卷积神经网络CNN和循环神经网络RNN

2. 数据集：https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews

3. 分类要求，将sentence进行情感分类，一共五类

   0: negative

   1: somewhat negative

   2:  neutral

   3: somewhat positive

   4: positive

### 二、数据处理

#### 1. 数据集划分

​        将数据集按train: crossValidation: test=6:2:2的比例划分，将train.tsv分为train_data.tsv, val_data.tsv, test_data.tsv三个文件，代码见CNN/split.py

#### 2. 文本处理

使用torchtext进行文本处理

1. 设置field

   ```
   TEXT = data.Field(sequential=True, tokenize=tokenize, stop_words='english', fix_length=60)
   LABEL = data.Field(sequential=False, use_vocab=False)
   fields = [('PhraseId', None), ('SentenceId', None), ('Phrase', TEXT), ('Sentiment', LABEL)]
   ```

   这里一定要指定fix_length，因为数据集中有非常短的的phrase，比如一个单词，如果不padding，会出现卷积核的h维度大于token数，导致runtimeerror. 此处我设置的是60，原因是数据集中最长的phrase为52个token.

2. 数据载入

   ```
   data.TabularDataset.splits（）
   ```

3. 构建词表

   ```python
   vectors = Vectors(name='../glove/glove.6B.200d.txt')
   TEXT.build_vocab(train_data, val_data, test_data, vectors=vectors)
   LABEL.build_vocab(train_data, val_data, test_data)
   weights = TEXT.vocab.vectors  # 嵌入矩阵的初始权重
   ```

   使用glove预训练的词向量构建，需要保存weight，用于网络中进行embedding

   也可以不使用预训练向量

4. 构建迭代器

   ```
   data.Iterator.splits()
   ```

### 三、模型搭建

#### 1. TextCNN

embedding层：进行WordEmbedding，可以选择随机embedding和使用预训练的词向量进行embedding

卷积层：进行卷积操作，需要指定卷积每个核的尺寸和数量

全连接层：线性操作

#### 2. TextRNN

用两种模型来定义TextRNN，RNN和LSTM，可以选择是否使用双向，

网络结构由embedding层和循环层和全连接层

### 四、训练脚本

1. train() 进行模型的训练

在训练时，要注意将text进行转置，将batch_size作为第一个维度，否则在网络中进行embedding时，会出现尺寸不匹配的情况

```
text, label = batch.Phrase, batch.Sentiment
text.t_()
```

2. eval()进行交叉验证和测试

### 五、main()

设置两种不同网络需要的各种参数，载入数据，进行模型的训练、保存和测试
