'''
使用案例，训练两个类型的语音，然后测试，对CPU和内存要求不高。内存使用在 1G 左右
'''

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
    TrainDataLoader, TestDataLoader = create_datasets(root_path='test/')

    # 构建模型
    model = FCNModel(input_size=16000, classes=CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-6, nesterov=True)

    # [编译模型] 配置模型，损失函数采用交叉熵，优化采用Adadelta，将识别准确率作为模型评估

    # callback_earlyStop = callbacks.EarlyStopping(monitor='loss', min_delta=0.001, patience=100, mode='auto',
    #                                              restore_best_weights=False)
    # save_dir = f'model/RNN_weights_-bs{BATCHSIZE}-eps{EPOCHS}-lr{lr}'
    # callback_chkpoint = callbacks.ModelCheckpoint(filepath=os.path.join(save_dir, 'model_{epoch:04d}.keras'),
    #                                               monitor='val_loss',
    #                                               save_freq='epoch',
    #                                               verbose=0,
    #                                               save_best_only=True,
    #                                               save_weights_only=False, mode='auto')
    # history = model.fit(wavs, labels, batch_size=BATCHSIZE, epochs=EPOCHS,
    #                     callbacks=[callback_chkpoint, callback_earlyStop], verbose=1,
    #                     validation_data=(testwavs, testlabels), )
    LossRate = 0
    CorrectTimes = 0
    Total = 0
    for epoch in range(EPOCHS):
        LossRate = 0
        Total = 0
        CorrectTimes = 0
        for BatchTimes, batch in enumerate(TrainDataLoader):
            wavs = batch['input']
            # print(wavs.shape)
            labels = batch['label']
            wavs = wavs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(),
            outputs = model(wavs)
            # print(outputs, labels)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()  # targets为真实结果

            LossRate += loss.item()
            a, predicted = outputs.max(1)
            # print(predicted, labels)
            Total += labels.size(0)
            CorrectTimes += torch.all(torch.eq(predicted.view(-1, 1), labels.argmax(dim=1).view(-1, 1)), dim=1).sum().item()

            # if BatchTimes % 100 == 0 or BatchTimes == (len(TrainDataLoader) - 1):
            #     print(epoch, BatchTimes, len(TrainDataLoader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #           % (LossRate / (BatchTimes + 1), 100. * CorrectTimes / Total, CorrectTimes, Total))
        print(epoch, 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (LossRate / (BatchTimes + 1), 100. * CorrectTimes / Total,
                                                           CorrectTimes, Total))

        test_loss = 0
        TP = 0
        test_total = 0
        with torch.no_grad():
            for BatchTimes, batch in enumerate(TestDataLoader):
                testwavs = batch['input']
                testlabels = batch['label']
                testwavs = testwavs.to(device)
                testlabels = testlabels.to(device)
                outputs = model(testwavs)
                loss = criterion(outputs, testlabels)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                test_total += testlabels.size(0)
                TP += torch.all(torch.eq(predicted.view(-1, 1), testlabels.argmax(dim=1).view(-1, 1)), dim=1).sum().item()
            print(epoch, 'Loss:%.3f | Access:%.3f%%(%d/%d)'% (test_loss / (BatchTimes + 1), 100. * TP / test_total,
                                                              TP, test_total))
        if epoch % 20 == 0:
            torch.save(model.state_dict(), f'model/{model.name}-{epoch}-acc{100 * TP / test_total:0.4f}.pth')

    # 验证集的label应该是有问题，导致val_loss越来越高
