import dataLoad
import textCNN_model
import train
import argparse
import torch


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--kernel_size', type=str, default='2,3,4')
    parser.add_argument('--kernel_num', type=int, default=100)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--device', type=int, default=-1)
    parser.add_argument('--test', action='store_true', default=False)
    args = parser.parse_args()

    train_data, val_data, test_data = 'train_data.tsv', 'val_data.tsv', 'test_data.tsv'
    train_itr, val_itr, test_itr, weight = dataLoad.dataLoad(train_data, val_data, test_data, args.batch_size)

    args.weight = weight
    args.label_num = 5
    args.kernel_size = [int(k) for k in args.kernel_size.split(',')]
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    textCNN = textCNN_model.TextCNN(args)
    textCNN.to(args.device)

    if args.test:
        print('-----Loading Model...-----')
        textCNN = torch.load('model/textCNN_model.pt')
        print('-----Testing...-----')
        train.eval(test_itr, textCNN, args)
    else:
        train.train(train_itr, val_itr, textCNN, args)
