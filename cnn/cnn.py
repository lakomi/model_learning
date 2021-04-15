import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from torchvision import datasets, transforms
# lenet5手写识别数字demo
PATH = "model415.pt"


# 构建网络
class cnnNet(nn.Module):
    def __init__(self):
        # Conv1d即一维卷积，常用于对文本数据。宽度卷积高度不卷积。
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        # LogSoftmax对softmax结果取log
        self.logsoftmax = nn.LogSoftmax()

    # 重写。前向传播
    def forward(self, x):
        in_size = x.size(0)
        # 第一部分，卷积池化激活
        out = self.relu(self.pool(self.conv1(x)))
        # 第二部分，卷积池化激活
        out = self.relu(self.pool(self.conv2(out)))
        out = out.view(in_size, -1)
        # 全连接、激活
        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out))
        # 全连接后通过softmax输出结果。a
        out = self.fc3(out)
        return self.logsoftmax(out)


# 获取训练集测试集
train_dataset = datasets.MNIST('data/', download=False, train=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,)),
                               ]))
test_dataset = datasets.MNIST('data/', download=False, transform=transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.1307,), (0.3081,)),
     ]))

# 数据集加载器。可自动将数据分割成batch，打乱顺序
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=64, shuffle=True)


model = cnnNet()
# 损失函数
lossfun = nn.NLLLoss()
# 优化。选用梯度下降法
optimsgd = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.5)


# 开始训练
for epoch in range(20):
    # t序号，数据-标签
    for t, (data, label) in enumerate(train_loader):
        data, label = Variable(data), Variable(label)
        # 模型输出的结果
        predict = model(data)
        # 计算loss
        loss = lossfun(predict, label)

        # 先将梯度归零、反向传播计算每个梯度值、通过参数下降执行参数更新
        optimsgd.zero_grad()
        loss.backward()
        optimsgd.step()

# 保存模型参数字典
torch.save(model.state_dict(), PATH)
# 读取模型，先实例化一个模型的对象，在加载之前保存的参数。
themodel = cnnNet()
themodel.load_state_dict(torch.load(PATH))

# 看看效果
correct = 0
for data, target in test_loader:
    data, target = Variable(data), Variable(target)
    output = themodel(data)
    # 1表示找第二维，max找最大值，即找第二维的最大值。keepdim为True保持维度、False则输出比输入少一个维度。
    predict = output.data.max(1, keepdim=True)[1]
    correct += predict.eq(target.data.view_as(predict)).cpu().sum()
print('{:.3f}%\n'.format(100. * correct / len(test_loader.dataset)))
