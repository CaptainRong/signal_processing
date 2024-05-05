from Hype import *
from model import FCNModel, RNN_model, RNN
from dataset import create_datasets
import os
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


# from keras import callbacks

def draw_confusion_matrix(label_true, label_pred, label_name, title="Confusion Matrix", pdf_save_path=None, dpi=300):
    cm = confusion_matrix(y_true=label_true, y_pred=label_pred, normalize='true')

    plt.imshow(cm, cmap='Blues')
    plt.title(title)
    plt.xlabel("Predict label")
    plt.ylabel("Truth label")
    plt.yticks(range(label_name.__len__()), label_name)
    plt.xticks(range(label_name.__len__()), label_name, rotation=45)

    plt.tight_layout()

    plt.colorbar()

    for i in range(label_name.__len__()):
        for j in range(label_name.__len__()):
            color = (1, 1, 1) if i == j else (0, 0, 0)  # 对角线字体白色，其他黑色
            value = float(format('%.2f' % cm[j, i]))
            plt.text(i, j, value, verticalalignment='center', horizontalalignment='center', color=color, fontsize=4)

    # plt.show()
    if not pdf_save_path is None:
        plt.savefig(pdf_save_path, bbox_inches='tight', dpi=dpi)
    plt.clf()


if __name__ == '__main__':
    TrainDataLoader, TestDataLoader = create_datasets(root_path='wav/')

    # 构建模型
    # FCN: input_size = time_length*features else = features
    model = FCNModel(input_size=32 * 44, classes=CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))

    LossRate = 0
    CorrectTimes = 0
    Total = 0
    train_losses = []
    train_accuracies = []

    test_losses = []
    test_accuracies = []

    for epoch in range(EPOCHS):
        if epoch % 10 == 0:
            all_labels = []
            all_predicted = []
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
            CorrectTimes += torch.all(torch.eq(predicted.view(-1, 1), labels.argmax(dim=1).view(-1, 1)),
                                      dim=1).sum().item()

            # if BatchTimes % 100 == 0 or BatchTimes == (len(TrainDataLoader) - 1):
            #     print(epoch, BatchTimes, len(TrainDataLoader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #           % (LossRate / (BatchTimes + 1), 100. * CorrectTimes / Total, CorrectTimes, Total))
        print(epoch,
              '(Train) :Loss: %.3f | Acc: %.3f%% (%d/%d)' % (LossRate / (BatchTimes + 1), 100. * CorrectTimes / Total,
                                                             CorrectTimes, Total))
        train_accuracy = 100. * CorrectTimes / Total
        train_losses.append(LossRate / (BatchTimes + 1))
        train_accuracies.append(train_accuracy)

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
                TP += torch.all(torch.eq(predicted.view(-1, 1), testlabels.argmax(dim=1).view(-1, 1)),
                                dim=1).sum().item()
                if epoch % 10 == 0:
                    for i in testlabels.argmax(dim=1).tolist():
                        all_labels.append(i)
                    for i in predicted.tolist():
                        all_predicted.append(i)
            print(epoch,
                  '(Test) :Loss:%.3f | Access:%.3f%%(%d/%d)' % (test_loss / (BatchTimes + 1), 100. * TP / test_total,
                                                                TP, test_total))
        test_accuracy = 100. * TP / test_total
        test_losses.append(test_loss / (BatchTimes + 1))
        test_accuracies.append(test_accuracy)

        if epoch % 20 == 0:
            torch.save(model.state_dict(), f'model/complete_{model.name}-{epoch}-acc{100 * TP / test_total:0.4f}.pth')
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.plot(train_losses, label='Train')
            plt.plot(test_losses, label='Test')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Loss Curve')
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(train_accuracies, label='Train')
            plt.plot(test_accuracies, label='Test')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy (%)')
            plt.title('Accuracy Curve')
            plt.legend()

            plt.savefig(f'curves_{epoch}.png')
            plt.close()
        if epoch % 10 == 0:
            draw_confusion_matrix(all_labels, all_predicted, list(os.listdir('wav/train/')),
                                  pdf_save_path=f'Confusion_matrix{epoch}')

    # 验证集的label应该是有问题，导致val_loss越来越高
