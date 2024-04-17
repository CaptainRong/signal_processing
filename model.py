from Hype import *
import torch
import torch.nn as nn
import torch.optim as optim

# def create_model(classes):
#     model = RNN_model(classes)
#     # [编译模型] 配置模型，损失函数采用交叉熵，优化采用Adadelta，将识别准确率作为模型评估
#     model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.SGD(learning_rate=lr, decay=1e-6, momentum=0.9, nesterov=True),
#                   metrics=['accuracy'])
#     #  validation_data为验证集
#     model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.SGD(learning_rate=lr, decay=1e-6, momentum=0.9, nesterov=True),
#                   metrics=['accuracy'])
#
#     return model


class FCNModel(nn.Module):
    def __init__(self, input_size, classes=CLASSES):
        super(FCNModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.name = 'FCNModel'

    def forward(self, x):
        input_size = x.shape[1]*x.shape[2]
        x = x.resize(-1, input_size)
        print(x.shape)
        x = self.relu(self.fc1(x))
        # print(x.shape)
        x = self.relu(self.fc2(x))
        # print(x.shape)
        x = self.relu(self.fc3(x))
        # print(x.shape)
        x = self.relu(self.fc4(x))
        # print(x.shape)
        return x


# def conv_1d_model(classes):
#     model_m = Sequential()
#     # model_m.add(Reshape((1600, 10), input_shape=(20, 44)))
#     model_m.add(Conv1D(512, 5, activation='relu', input_shape=(20, 44)))
#     model_m.add(Conv1D(256, 3, activation='relu', input_shape=(16, 512)))
#     model_m.add(MaxPooling1D(2))
#     model_m.add(Conv1D(256, 4, activation='relu', input_shape=(7, 256)))
#     model_m.add(Conv1D(128, 3, activation='relu', input_shape=(4, 256)))
#     model_m.add(GlobalAveragePooling1D())
#     model_m.add(Dense(64, activation='relu'))
#     model_m.add(Dense(32, activation='relu'))
#     model_m.add(Dropout(0.5))
#     model_m.add(Dense(classes, activation='softmax'))
#     return model_m


class RNN_model(nn.Module):
    def __init__(self, input_size, classes=CLASSES):
        super(RNN_model, self).__init__()
        self.hidden_size = 128
        self.LSTM = nn.LSTM(input_size=8, hidden_size=self.hidden_size, num_layers=2, batch_first=True)
        self.linear1 = nn.Linear(16000, 10240)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(10240, 4096)
        self.conv1 = nn.Conv1d(128, 64, 5)  #508, 128
        self.maxpool1 = nn.MaxPool1d(2, 2) # 254
        self.conv2 = nn.Conv1d(64, 32, 3) # 252
        self.maxpool2 = nn.MaxPool1d(2, 5)  # 50
        self.linear3 = nn.Linear(32*51, classes)
        self.name = 'RNN_model'

    def forward(self, x):
        bth = x.shape[0]
        x = x.view(bth, -1)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = x.view(bth, 512, 8)
        # print(x.shape)
        x, (ht,ct) = self.LSTM(x) # length*hidden 512, 128
        # print(x.shape)
        x = x.permute(0, 2, 1)

        x = self.conv1(x)
        # print(x.shape)
        x = self.relu(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        # print(x.shape)
        x = self.relu(x)
        x = self.maxpool2(x)
        # print(x.shape)

        x = x.view(bth, -1)
        # print(x.shape)
        x = self.linear3(x)
        # print(x.shape)
        return x


class RNN(nn.Module):
    def __init__(self, input_size, classes):
        super(RNN, self).__init__()
        hidden_size=128
        num_layers = 2
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, classes)
        self.name = 'RNN'

    def forward(self, x):
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # 前向传播RNN
        out, _ = self.rnn(x, h0)

        # 取最后一个时间步的输出
        out = self.fc(out[:, -1, :])
        return out

if __name__ == '__main__':
    model = FCNModel(input_size=16000, classes=11)
    x = torch.randn(128, 16000)
    model.forward(x)

    inputs = torch.randn(128, 48, 40)
    model = RNN_model(classes=11)
    output, (_, _) = model.forward(inputs)
    print(output.shape)