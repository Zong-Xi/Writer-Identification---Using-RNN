import torch as t
from torchvision import transforms
import torch.utils.data as Data
import torch.autograd.variable as V
from torch import optim
import torch.nn as nn
import traindataset
from collections import Counter
import time
import numpy as np


# 数据预处理参数设置
path_train = './data/Data10/Validation'
path_test = './data/Data10/Train'
train_sample = 3000  #训练集中每个同学所有字合并的RHS个数
test_sample = 1000  #测试集中每个同学所有字合并的RHS个数
seq = 50    #单个RHS长度
# 网络参数设置
num_epoch = 20   #训练epoch次数
layer = 2
input_size = 3
hidden_size = 300
batch_size = 200
num_class=10
num_RHS = test_sample
PATH='./model/model_10.pkl'  #保存训练模型位置

# 定义数据的预处理及网络
transform = transforms.Compose([
    transforms.ToTensor(),  # 转为Tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 归一化
])
t.set_num_threads(8)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
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
            out_mean[i] = (outputs[i][seq - 1][0] + outputs[i][0][1]) / 2
        out = self.linear(out_mean)
        return out

net = Net()
net.cuda()  #在GPU训练

# 定义损失函数及优化方法
criterion = nn.CrossEntropyLoss()  # 损失函数
optimizer = optim.Adam(net.parameters(), lr=0.001)

if __name__ == '__main__':

    # 加载数据
    train_data, train_label, test_data, test_label = traindataset.dataset(path_train,path_test,seq,train_sample,seq,test_sample)

    # 训练集
    tensor_data = t.from_numpy(train_data)
    tensor_label = t.from_numpy(train_label)
    trainset = Data.TensorDataset(tensor_data, tensor_label)

    trainloader = t.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2)

    # 测试集
    tensor_data = t.from_numpy(test_data)
    tensor_label = t.from_numpy(test_label)
    testset = Data.TensorDataset(tensor_data, tensor_label)

    testloader = t.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2)

    # 开始训练
    t.set_num_threads(8)
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    start_time = time.time()
    loss_arr = []
    for epoch in range(num_epoch):

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs = V(inputs.float())  #tensor转为Variable类型(需为float)
            labels = labels.long().squeeze()    #label转为long类型(展开成一维)
            inputs = inputs.cuda()  #在GPU训练
            labels = labels.cuda()  #在GPU训练

            outputs = net(inputs)

            # 梯度清零
            optimizer.zero_grad()

            # forward + backward
            loss = criterion(outputs, labels)
            loss_arr.append(loss.item())
            loss.backward()

            # 更新参数
            optimizer.step()

            # 打印log信息
            # loss 是一个scalar,需要使用loss.item()来获取数值，不能使用loss[0]
            running_loss += loss.item()
            if i % 10 == 9:  # 每10个batch打印一下训练状态
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0
    np.save('loss_10.npy', loss_arr)
    end_time = time.time()
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print("training time: " + str(end_time - start_time))
    print('Finished Training')


    # 存储模型
    t.save(net.state_dict(), PATH)
    print('Model Saved')
    # 加载模型
    model = Net()
    model.load_state_dict(t.load(PATH, map_location="cuda:0"))
    model.cuda()
    print('Model Loaded')


    # 开始预测
    correct = 0  # 预测正确的图片数
    total = 0  # 总共的图片数
    predict=[]

    with t.no_grad():
        for data in testloader:


            inputs, labels = data
            inputs = V(inputs.float())  #tensor转为Variable类型(需为float)
            labels = labels.long().squeeze()  #label转为long类型(展开成一维)
            inputs = inputs.cuda()  #在GPU训练
            labels = labels.cuda()  #在GPU训练
            outputs = model(inputs)

            _, predicted = t.max(outputs, 1)    #取softmax后概率最大的作为预测结果
            total += labels.size(0)
            correct += (predicted == labels).sum()
            predict.append(predicted)   #投票机制

    print('测试集中单个RHS的准确率为: %d %%' % (100 * correct / total))

    # 投票预测所有RHS是哪位同学
    predict = t.stack(predict)  #list转为tensor
    predict = t.reshape(predict, (-1,))     #将tensor展开成一维
    predict.tolist()
    Step=test_sample
    accracy=0
    preData = [predict[i:i+Step] for i in range(0,len(predict),Step)]   #将预测结果划分为10类(每test_sample个为一个同学)
    # print(type(data))
    for i in range(len(preData)):
        X=preData[i].tolist()
        # print(type(X))
        result = Counter(X).most_common(1)[0][0]    #投票数最高者
        accracy += (result==i)
    print('测试集投票后的准确率为: %d %%' % (100 * accracy/num_class))