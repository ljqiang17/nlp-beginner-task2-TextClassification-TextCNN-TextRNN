import torch
import torch.nn.functional as F


def train(train_itr, val_itr, model, args):

    model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  # 优化参数，指定learning rate

    step = 0
    best_acc = 0
    save_path = 'model/textCNN_model.pt'

    model.train()
    print('-----Training...-----')
    for epoch in range(1, args.epochs+1):
        print('Epoch: {}/{}'.format(epoch, args.epochs))
        for batch in train_itr:
            text, label = batch.Phrase, batch.Sentiment
            text.t_()
            # label.sub_(1)
            text = text.to(args.device)
            label = label.to(args.device)

            optimizer.zero_grad()  # 将模型梯度置零
            predict = model(text)  # 预测值
            loss = F.cross_entropy(predict, label)  # 交叉熵作为损失函数
            loss.backward()  # 反向传播
            optimizer.step()  # 更新

            step += 1
            if step % 1000 == 0:
                predict_y = torch.max(predict, 1)[1].view(label.size())
                accuracy = (predict_y.data == label.data).sum() / batch.batch_size
                print('\rBatch[{}] - loss: {:.6f} acc: {:.4f}'.format(step, loss.data.item(), accuracy))
                # sys.stdout.write('\rBatch[{}] - loss: {:.6f} acc: {:.4f}'.format(step, loss.data.item(), accuracy))

            if step % 1000 == 0:
                val_acc = eval(val_itr, model, args)
                if val_acc > best_acc:
                    best_acc = val_acc
                    torch.save(model, save_path)
            elif step % 500 == 0:
                torch.save(model, save_path)


def eval(val_itr, model, args):
    model.eval()
    val_loss = 0
    val_correct = 0
    for batch in val_itr:
        text, label = batch.Phrase, batch.Sentiment
        text.t_()
        text = text.to(args.device)
        label = label.to(args.device)

        predict = model(text)  # 预测值
        loss = F.cross_entropy(predict, label)  # 交叉熵作为损失函数
        val_loss += loss.data.item()

        predict_y = torch.max(predict, 1)[1].view(label.size())
        val_correct += (predict_y.data == label.data).sum()

    data_size = len(val_itr.dataset)
    val_loss /= data_size
    val_acc = val_correct / data_size
    print('Evaluation - loss: {:.6f} acc: {:.4f}\n'.format(val_loss, val_acc))
    # sys.stdout.write('\n Evaluation - loss: {:.6f} acc: {:.4f}'.format(val_loss, val_acc))

    return val_acc



