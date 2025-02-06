"""
 @Author: TCcjx
 @Email: tcc2025@163.com
 @FileName: pytorch.py
 @DateTime: 2025-02-05 20:49
 @SoftWare: PyCharm
"""
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

n_points = 500
data = torch.rand(n_points,2) * 2 - 1
labels = (data.norm(dim=1) > 0.7).float().unsqueeze(1)
data = data.to(device)
labels = labels.to(device)

# 创建模型类
class CircleClassifier(nn.Module):
    def __init__(self):
        super(CircleClassifier, self).__init__()
        self.layer1 = nn.Linear(2,20)
        self.layer2 = nn.Linear(20,1)

    def forward(self,x): # 使对象成为可调用对象
        x = torch.sigmoid(self.layer2(torch.relu(self.layer1(x))))
        return x


lr = 0.05
n_epochs = 1000

# 实例化
model = CircleClassifier()
model = model.to(device)
loss_fn = nn.BCELoss()
optimizer = optim.AdamW(model.parameters(),lr = lr) # 优化对象
prediction = model(data) # 可调用对象
# print(prediction)

for epoch in range(n_epochs):
    optimizer.zero_grad() # 梯度清零
    prediction = model(data) # 可调用对象，做前馈运算
    loss = loss_fn(prediction,labels)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f'Epoch: {epoch},Loss: {loss.item():.4f},device : {device}')


data = data.cpu()
labels = labels.cpu()
plt.scatter(data[:,0],data[:,1],c=(labels.squeeze() > 0.5),cmap='coolwarm')
plt.title('view')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# print(labels)
# print(data)