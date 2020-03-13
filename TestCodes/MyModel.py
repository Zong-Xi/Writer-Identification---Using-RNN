import torch as t
import torch.autograd.variable as V
import torch.utils.data as Data
import torch.nn as nn
from collections import Counter
import numpy as np
from pre import preprocessing


def MyFunction(filedata,num_class):

    # 初始值
    if num_class==10:
        seq = 50
        num_RHS = 1000
        layer = 1
        input_size = 3
        hidden_size = 500
        batch_size = 200
        classes = [2016310874, 2017310350, 2018211051, 2018211054, 2018211702,
       2018310874, 2018310892, 2018310897, 2018310898, 2018310939]  # 学号


    elif num_class==107:
        seq = 100
        num_RHS = 1000
        layer = 1
        input_size = 3
        hidden_size = 500
        batch_size = 200
        classes = [2015011414, 2015011431, 2015011455, 2015011548, 2016310874,
       2017210966, 2017211061, 2017213725, 2017213726, 2017310350,
       2017310472, 2017310851, 2017310881, 2017312279, 2017312287,
       2018210461, 2018210809, 2018210817, 2018210850, 2018211038,
       2018211039, 2018211047, 2018211048, 2018211051, 2018211053,
       2018211054, 2018211057, 2018211058, 2018211059, 2018211060,
       2018211061, 2018211062, 2018211063, 2018211064, 2018211067,
       2018211068, 2018211069, 2018211073, 2018211074, 2018211077,
       2018211079, 2018211080, 2018211081, 2018211114, 2018211167,
       2018211208, 2018211270, 2018211277, 2018211702, 2018214042,
       2018214043, 2018214052, 2018214113, 2018270031, 2018270032,
       2018280076, 2018280357, 2018310692, 2018310755, 2018310769,
       2018310874, 2018310875, 2018310876, 2018310881, 2018310882,
       2018310883, 2018310884, 2018310885, 2018310887, 2018310888,
       2018310892, 2018310894, 2018310895, 2018310897, 2018310898,
       2018310900, 2018310904, 2018310906, 2018310907, 2018310908,
       2018310909, 2018310910, 2018310911, 2018310915, 2018310916,
       2018310919, 2018310921, 2018310922, 2018310926, 2018310927,
       2018310929, 2018310932, 2018310933, 2018310934, 2018310936,
       2018310939, 2018310942, 2018310943, 2018310946, 2018310948,
       2018311127, 2018311146, 2018312459, 2018312470, 2018312476,
       2018312481, 2018312484]  # 学号

    else:
        print('num_class is wrong')
        return 0


    # 数据预处理
    testset = preprocessing(filedata, num_RHS, seq)

    testloader = Data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2)


    # 定义网络
    class Net10(nn.Module):
        def __init__(self):
            super(Net10, self).__init__()
            self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=layer,
                                batch_first=True, bidirectional=True)
            self.linear = nn.Linear(hidden_size, num_class)

        def forward(self, x):
            h0 = V(t.randn(2 * layer, batch_size, hidden_size)).cuda()
            c0 = V(t.randn(2 * layer, batch_size, hidden_size)).cuda()
            output, (hn, cn) = self.lstm(x, (h0, c0))
            outputs = output.view(batch_size, seq, 2, hidden_size)
            out_mean = t.zeros(batch_size, hidden_size).cuda()
            for i in range(len(outputs)):
                out_mean[i] = (outputs[i][seq-1][0] + outputs[i][0][1]) / 2
            out = self.linear(out_mean)
            return out

    class Net107(nn.Module):
        def __init__(self):
            super(Net107, self).__init__()
            self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=layer,
                                batch_first=True, bidirectional=True)
            self.linear = nn.Linear(hidden_size, num_class)

        def forward(self, x):
            h0 = V(t.randn(2 * layer, batch_size, hidden_size)).cuda()
            c0 = V(t.randn(2 * layer, batch_size, hidden_size)).cuda()
            output, (hn, cn) = self.lstm(x, (h0, c0))
            outputs = output.view(batch_size, seq, 2, hidden_size)
            out_mean = t.zeros(batch_size, hidden_size).cuda()
            for i in range(len(outputs)):
                out_mean[i] = (outputs[i][seq-1][0] + outputs[i][0][1]) / 2
            out = self.linear(out_mean)
            return out

    # 加载网络
    if num_class==10:
        PATH = './model/model_10_20.pkl'
        model = Net10()
        model.load_state_dict(t.load(PATH, map_location="cuda:0"))
        model.cuda()

    else:
        PATH = './model/model_107_7.pkl'
        model = Net107()
        model.load_state_dict(t.load(PATH, map_location="cuda:0"))
        model.cuda()

    # 预测
    predict = []
    with t.no_grad():
        for data in testloader:
            inputs = data
            inputs = V(inputs.float())
            inputs = inputs.cuda()
            outputs = model(inputs)
            _, predicted = t.max(outputs, 1)
            predict.append(predicted)
    predict = t.stack(predict)
    predict = t.reshape(predict, (-1,))
    predict.tolist()
    result = Counter(predict).most_common(1)[0][0]

    return classes[result]
