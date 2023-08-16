import torch 
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms


# ================================================================== #
#                         目录                                        #
# ================================================================== #

# 1. 基础自动求导示例 1                     (第25行到第39行)
# 2. 基础自动求导示例 2                     (第46行到第83行)
# 3. 从numpy加载数据                        (第90行到第97行)
# 4. 输入管道                              (第104行到第129行)
# 5. 自定义数据集的输入管道                  (第136行到第156行)
# 6. 预训练模型                            (第163行到第176行)
# 7. 保存和加载模型                         (第183行到第189行) 


# ================================================================== #
#                     1. 基础自动求导示例 1                           #
# ================================================================== #

# 创建张量。
x = torch.tensor(1., requires_grad=True)
w = torch.tensor(2., requires_grad=True)
b = torch.tensor(3., requires_grad=True)

# 构建计算图。
y = w * x + b    # y = 2 * x + 3

# 计算梯度。
y.backward()

# 打印梯度。
print(x.grad)    # x.grad = 2 
print(w.grad)    # w.grad = 1 
print(b.grad)    # b.grad = 1 


# ================================================================== #
#                    2. 基础自动求导示例 2                            #
# ================================================================== #

# 创建形状为(10, 3)和(10, 2)的张量。
x = torch.randn(10, 3)
y = torch.randn(10, 2)

# 构建全连接层。
linear = nn.Linear(3, 2)
print ('w: ', linear.weight)
print ('b: ', linear.bias)

# 构建损失函数和优化器。
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(linear.parameters(), lr=0.01)

# 前向传播。
pred = linear(x)

# 计算损失。
loss = criterion(pred, y)
print('loss: ', loss.item())

# 反向传播。
loss.backward()

# 打印梯度。
print ('dL/dw: ', linear.weight.grad) 
print ('dL/db: ', linear.bias.grad)

# 1步梯度下降。
optimizer.step()

# 也可以在低级别执行梯度下降。
# linear.weight.data.sub_(0.01 * linear.weight.grad.data)
# linear.bias.data.sub_(0.01 * linear.bias.grad.data)

# 打印1步梯度下降后的损失。
pred = linear(x)
loss = criterion(pred, y)
print('loss after 1 step optimization: ', loss.item())


# ================================================================== #
#                     3. 从numpy加载数据                              #
# ================================================================== #

# 创建一个numpy数组。
x = np.array([[1, 2], [3, 4]])

# 将numpy数组转换为torch张量。
y = torch.from_numpy(x)

# 将torch张量转换为numpy数组。
z = y.numpy()


# ================================================================== #
#                         4. 输入管道                                 #
# ================================================================== #

# 下载并构造CIFAR-10数据集。
train_dataset = torchvision.datasets.CIFAR10(root='../../data/',
                                             train=True, 
                                             transform=transforms.ToTensor(),
                                             download=True)

# 获取一个数据对（从磁盘读取数据）。
image, label = train_dataset[0]
print (image.size())
print (label)

# 数据加载器（这提供了非常简单的队列和线程）。
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=64, 
                                           shuffle=True)

# 当迭代开始时，队列和线程开始从文件加载数据。
data_iter = iter(train_loader)

# 小批量图像和标签。
images, labels = data_iter.next()

# 数据加载器的实际使用如下。
for images, labels in train_loader:
    # 这里应写训练代码。
    pass


# ================================================================== #
#                5. 自定义数据集的输入管道                             #
# ================================================================== #

# 应该如下构建您的自定义数据集。
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self):
        # TODO
        # 1. 初始化文件路径或文件名列表。 
        pass
    def __getitem__(self, index):
        # TODO
        # 1. 从文件中读取一个数据（例如，使用numpy.fromfile，PIL.Image.open）。
        # 2. 预处理数据（例如，torchvision.Transform）。
        # 3. 返回一个数据对（例如，图像和标签）。
        pass
    def __len__(self):
        # 您应该将0更改为数据集的总大小。
        return 0 

# 然后可以使用预构建的数据加载器。
custom_dataset = CustomDataset()
train_loader = torch.utils.data.DataLoader(dataset=custom_dataset,
                                           batch_size=64, 
                                           shuffle=True)


# ================================================================== #
#                        6. 预训练模型                               #
# ================================================================== #

# 下载并加载预训练的ResNet-18。
resnet = torchvision.models.resnet18(pretrained=True)

# 如果你只想微调模型的顶层，可以如下设置。
for param in resnet.parameters():
    param.requires_grad = False

# 替换顶层以进行微调。
resnet.fc = nn.Linear(resnet.fc.in_features, 100)  # 100是一个例子。

# 前向传播。
images = torch.randn(64, 3, 224, 224)
outputs = resnet(images)
print (outputs.size())     # (64, 100)


# ================================================================== #
#                      7. 保存和加载模型                              #
# ================================================================== #

# 保存和加载整个模型。
torch.save(resnet, 'model.ckpt')
model = torch.load('model.ckpt')

# 保存和加载模型参数（推荐）。
torch.save(resnet.state_dict(), 'params.ckpt')
resnet.load_state_dict(torch.load('params.ckpt'))