---
name: mnist-handwriting-recognition
description: 使用 PyTorch 构建并训练一个简单的全连接神经网络，对 MNIST 手写数字数据集进行分类识别。
---

# MNIST 手写数字识别

## 概述
本技能提供了一个端到端的 PyTorch 流程，涵盖从环境配置、数据集加载、模型构建，到训练、评估和模型持久化的全部步骤。目标是训练一个可以对 28×28 灰度手写数字图像进行分类的神经网络。

## 何时使用此技能
- 需要快速搭建一个手写数字识别原型时
- 学习 PyTorch 基础流程（数据集、模型、训练循环、保存/加载）
- 作为基准模型，用于测试不同的优化器、损失函数或网络结构

## 工作流程

### 第一步：环境准备
确保已安装 PyTorch 和 torchvision：
```bash
pip install torch torchvision
```

在代码中导入必要的模块：

python

```
import torch
import torchvision.datasets
from torch.utils import data
from torchvision import transforms
from torch import nn
```



### 第二步：构建数据集

利用 `torchvision.datasets.MNIST` 下载并加载数据，将其转换为张量格式，再通过 `DataLoader` 封装成可迭代的批次数据。

python

```
trans = transforms.ToTensor()

mnist_train = torchvision.datasets.MNIST(
    root='../data', train=True, transform=trans, download=True
)
mnist_test = torchvision.datasets.MNIST(
    root='../data', train=False, transform=trans, download=True
)

batch_size = 64
train_loader = data.DataLoader(mnist_train, batch_size=batch_size)
test_loader  = data.DataLoader(mnist_test, batch_size=batch_size)
```



### 第三步：定义模型

构建一个简单的全连接网络：先将 28×28 的图像展平为 784 维向量，然后通过三个线性层（含 ReLU 激活）映射到 10 个类别。

python

```
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device=0)   # 使用 GPU 0，若用 CPU 改为 'cpu'
print(model)
```



### 第四步：配置损失函数与优化器

多分类任务使用交叉熵损失，优化器采用随机梯度下降（SGD）。

python

```
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
```



### 第五步：编写训练与测试循环

训练函数执行前向传播、损失计算、反向传播和参数更新；测试函数在 `torch.no_grad()` 下计算平均损失与准确率。

python

```
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(0), y.to(0)   # 设备需与模型一致

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            print(f"loss: {loss.item():>7f}  [{batch * len(X):>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(0), y.to(0)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    print(f"Test Error:\n Accuracy: {100*correct:>0.1f}%, Avg loss: {test_loss:>8f}\n")
```



主训练循环：

python

```
epochs = 50
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------")
    train(train_loader, model, loss_fn, optimizer)
    test(test_loader, model, loss_fn)
print("Done!")
```



### 第六步：保存与加载模型

python

```
torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")

# 加载模型
model = NeuralNetwork()
model.load_state_dict(torch.load("model.pth"))
model.to(0)
```



## 预期结果

经过 50 个 epoch 训练，简单全连接网络在测试集上的准确率大约为 65%；如果使用保存的权重继续训练，准确率可提升至约 92%。最终指标取决于初始化、超参数和硬件。

## 约定与最佳实践

- **设备一致性**：张量和模型必须位于同一设备，示例中固定为 `0`（GPU），请根据实际情况修改（CPU 时使用 `'cpu'`）。
- **损失函数细节**：`nn.CrossEntropyLoss` 内部已包含 `log_softmax`，因此模型最后一层无需再加 softmax。
- **评估模式**：测试前务必调用 `model.eval()` 以固定 dropout 或 batch norm 等层的行为。
- **监控与调试**：每 100 个 batch 打印一次损失，便于观察训练是否收敛。
