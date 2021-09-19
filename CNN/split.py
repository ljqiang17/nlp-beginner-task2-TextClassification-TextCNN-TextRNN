import csv
import pandas as pd
data = pd.read_csv('../data/train.tsv', delimiter='\t')

# 数据分割 train:val:test = 6:2:2
splitNum1 = int(len(data) * 0.6)
splitNum2 = int(len(data) * 0.8)
train_data = data.head(splitNum1)
cv_data = data.loc[splitNum1: splitNum2]
test_data = data.loc[splitNum2: len(data)]

with open('../data/train_data.tsv', 'w') as fw1:
    fw1.write(train_data.to_csv(sep='\t', index=False))

with open('../data/val_data.tsv', 'w') as fw2:
    fw2.write(cv_data.to_csv(sep='\t', index=False))

with open('../data/test_data.tsv', 'w') as fw3:
    fw3.write(test_data.to_csv(sep='\t', index=False))

