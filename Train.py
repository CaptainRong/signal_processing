
import os
import torch
import torch.nn as nn
import torch.optim as optim
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from dataset import create_datasets
# from keras import callbacks
from Hype import *
from model import FCNModel, RNN_model, RNN

if __name__ == '__main__':
    device = torch.device('cuda')
    TrainDataLoader, TestDataLoader = create_datasets(root_path='wav/')

    # 构建模型
    # FCN: input_size = time_length*features else = features
    model = FCNModel(input_size=32*44, classes=CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))

    LossRate = 0
    CorrectTimes = 0
    Total = 0
    for epoch in range(EPOCHS):
        LossRate = 0
        Total = 0
        CorrectTimes = 0
        for BatchTimes, batch in enumerate(TrainDataLoader):
            wavs = batch['input'].to(device)
            # print(wavs.shape)
            labels = batch['label'].to(device)

            optimizer.zero_grad(),
            outputs = model(wavs)
            # print(outputs, labels)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()  # targets为真实结果
            # print(f"{predicted[0:5]}\n{testlabels.argmax(dim=1).view(-1, 1)[0:5]}")
            LossRate += loss.item()
            a, predicted = outputs.max(1)
            # print(predicted, labels)
            Total += labels.size(0)
            CorrectTimes += torch.all(torch.eq(predicted.view(-1, 1), labels.argmax(dim=1).view(-1, 1)), dim=1).sum().item()

            # if BatchTimes % 100 == 0 or BatchTimes == (len(TrainDataLoader) - 1):
            #     print(epoch, BatchTimes, len(TrainDataLoader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #           % (LossRate / (BatchTimes + 1), 100. * CorrectTimes / Total, CorrectTimes, Total))
        print(epoch, '(Train) :Loss: %.3f | Acc: %.3f%% (%d/%d)' % (LossRate / (BatchTimes + 1), 100. * CorrectTimes / Total,
                                                           CorrectTimes, Total))

        test_loss = 0
        TP = 0
        test_total = 0
        with torch.no_grad():
            for BatchTimes, batch in enumerate(TestDataLoader):
                testwavs = batch['input'].to(device)
                testlabels = batch['label'].to(device)

                outputs = model(testwavs)
                loss = criterion(outputs, testlabels)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                # print(f"{predicted[0:2]}\n{testlabels.argmax(dim=1).view(-1, 1)[0]}")
                test_total += testlabels.size(0)
                TP += torch.all(torch.eq(predicted.view(-1, 1), testlabels.argmax(dim=1).view(-1, 1)), dim=1).sum().item()
            print(epoch, '(Test) :Loss:%.3f | Access:%.3f%%(%d/%d)'% (test_loss / (BatchTimes + 1), 100. * TP / test_total,
                                                              TP, test_total))
        if epoch % 20 == 0:
            torch.save(model.state_dict(), f'model/{model.name}-{epoch}-acc{100 * TP / test_total:0.4f}.pth')

    # 验证集的label应该是有问题，导致val_loss越来越高
