import dataLoad
import textRNN_model
import train
import argparse
import torch


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--rnn_type', type=str, default='rnn')
    parser.add_argument('--hidden_size', type=int, default=20)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--bidirectional', type=bool, default=False)
    parser.add_argument('--device', type=int, default=-1)
    parser.add_argument('--test', action='store_true', default=False)
    args = parser.parse_args()

    print('-----Data Loading...-----')
    train_data, val_data, test_data = 'train_data.tsv', 'val_data.tsv', 'test_data.tsv'
    train_itr, val_itr, test_itr, weight = dataLoad.dataLoad(train_data, val_data, test_data, args.batch_size)
    print('------Data Loaded.-------')

    args.weight = weight
    args.label_num = 5
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    textRNN = textRNN_model.TextRNN(args)
    textRNN.to(args.device)

    if args.test:
        save_path = ''
        if args.rnn_type == 'rnn':
            if args.bidirectional:
                save_path = 'model/rnn/textRNN_rnn2_model.pt'
            else:
                save_path = 'model/rnn/textRnn_rnn1_model.pt'
        elif args.rnn_type == 'lstm':
            if args.bidirectional:
                save_path = 'model/lstm/textRNN_lstm2_model.pt'
            else:
                save_path = 'model/lstm/textRnn_lstm1_model.pt'
        print('-----Loading Model...-----')
        textRNN = torch.load(save_path)
        print('-----Testing by textRNN...-----')
        train.eval(test_itr, textRNN, args)
    else:
        train.train(train_itr, val_itr, textRNN, args)
