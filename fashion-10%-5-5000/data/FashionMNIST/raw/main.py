import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import gc
# 1. Sample batch index
def sample_batch_index(total, batch_size):
    '''Sample index of the mini-batch.'''
    total_idx = np.random.permutation(total)
    batch_idx = total_idx[:batch_size]
    return batch_idx

# 2. Initialize basic settings
batch_size = 32  # Adjusted batch size to 32
image_size = 32  # Adjusted image size to 32x32
img_transform = transforms.Compose([
    transforms.Resize(image_size),  # Resize images to 32x32
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 3. Download MNIST dataset and prepare DataLoader
MNIST = dset.MNIST('./data/', transform=img_transform, download=True)
dataloader = torch.utils.data.DataLoader(MNIST, batch_size=batch_size, shuffle=True, num_workers=0)

# 4. Data collection
images = []
labels = []
for i, data in enumerate(dataloader):
    image, label = data
    images.append(image)
    labels.append(label)

images = torch.cat(images)
print(images.shape)

# 5. Set device and other parameters
batch_size = 32  # Adjusted batch size
image_size = 32  # Image size set to 32x32
num_worker = 5  # Reduced to 5 nodes
use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")

# 6. Define loss print function
def print_list(model_list, G_D, err_list):
    result = ''
    if G_D == "D":
        for i in range(len(model_list)):
            result = result + ' Loss_D_' + str(model_list[i].get_id()) + ': ' + str(format(err_list[i], '.4f'))
    if G_D == "G":
        for i in range(len(model_list)):
            result = result + ' Loss_G_' + str(model_list[i].get_id()) + ': ' + str(format(err_list[i], '.4f'))
    return result

# 7. Initialize weights for the models
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# 8. Generator model (Reduced layers and adjusted for 32x32 images)
class G(nn.Module):
    def __init__(self, id):
        super(G, self).__init__()
        self.id = id
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 128, 4, 1, 0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def get_id(self):
        return self.id

    def forward(self, input):
        output = self.main(input)
        return output

# 9. Discriminator model (Reduced layers and adjusted for 32x32 images)
class D(nn.Module):
    def __init__(self, id):
        super(D, self).__init__()
        self.id = id
        self.main = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(256, 1, 4, 1, 0, bias=False),
        )

    def get_id(self):
        return self.id

    def forward(self, input):
        output = self.main(input)
        return output.view(-1)

# 10. Initialize networks
netD_list = [D(i).to(device).apply(weights_init) for i in range(num_worker)]
netG_list = [G(i).to(device).apply(weights_init) for i in range(num_worker)]

optimizerD_list = [optim.RMSprop(netD.parameters(), lr=0.0002, alpha=0.9) for netD in netD_list]
optimizerG_list = [optim.RMSprop(netG.parameters(), lr=0.0002, alpha=0.9) for netG in netG_list]

# 11. Create directories
if not os.path.exists('ODDFedGAN'):
    os.makedirs('ODDFedGAN')

if not os.path.exists('ODDFedGAN/models'):
    os.makedirs('ODDFedGAN/models')

if not os.path.exists('ODDFedGAN/gen_images'):
    os.makedirs('ODDFedGAN/gen_images')





# 12. Simulating OOD data using Fashion-MNIST
def simulate_ood_data(images, labels, ood_percentage=0.1):
    """Replace a portion of MNIST data with Fashion-MNIST images to simulate OOD."""
    num_samples = len(images)
    ood_size = int(ood_percentage * num_samples)

    # Load Fashion-MNIST dataset
    fashion_mnist = dset.FashionMNIST('./data/', transform=img_transform, download=True)
    fashion_loader = torch.utils.data.DataLoader(fashion_mnist, batch_size=ood_size, shuffle=True, num_workers=0)
    fashion_images, _ = next(iter(fashion_loader))

    # Randomly select indices to replace
    ood_indices = np.random.choice(num_samples, ood_size, replace=False)
    images[ood_indices] = fashion_images[:ood_size]

    return images

images = simulate_ood_data(images, labels)


# 13. Training loop
num_epochs = 5001
#losses = []
# 存储损失数据
loss_D_list = []
loss_G_list = []

torch.save(netG_list[0].state_dict(), "ODDFedGAN/models/netG_00000.pt")
sum_data = images.shape[0]
node_data = sum_data / num_worker
p = node_data / sum_data

for epoch in range(num_epochs):
    loss_fake_list = []
    errD_list = []
    fake_list = []

    for t in range(3):
        batch_idxes = []
        for (i, netD), optimizerD in zip(enumerate(netD_list), optimizerD_list):
            batch_idx = sample_batch_index(images.shape[0], batch_size)
            data = images[batch_idx]
            real = Variable(data).to(device)
            noise = Variable(torch.randn(real.size()[0], 100, 1, 1)).to(device)
            fake = netG_list[i](noise)
            pred_real = netD(real)
            pred_fake = netD(fake)
            netD.zero_grad()
            loss_real = torch.mean(pred_real)
            loss_fake = torch.mean(pred_fake)
            errD = - loss_real + loss_fake
            errD.backward()     # 移除 retain_graph=True
            optimizerD.step()

            for parm in netD.parameters():
                parm.data.clamp_(-0.01, 0.01)

            if t == 0:
                loss_fake_list.append(loss_fake)
                errD_list.append(errD)
                fake_list.append(fake)
            else:
                loss_fake_list[i] = loss_fake
                errD_list[i] = errD
                fake_list[i] = fake

    # Update generator weights
    errG_list = []
    for (i, netG), optimizerG in zip(enumerate(netG_list), optimizerG_list):
        # 生成器重新 forward 一次，避免使用旧图
        noise = torch.randn(batch_size, 100, 1, 1, device=device)
        fake = netG(noise)  # 👈 新图，干净

        loss_fake = torch.mean(netD_list[i](fake))
        errG = - loss_fake

        netG.zero_grad()
        errG.backward()
        optimizerG.step()


        errG_list.append(errG)

    # Model aggregation
    if epoch % 1000 == 0:
        # 自动平均聚合判别器参数
        with torch.no_grad():
            for params in zip(*[netD.parameters() for netD in netD_list]):
                avg_data = sum(p.data for p in params) / num_worker
                for p in params:
                    p.data.copy_(avg_data)

        # 自动平均聚合生成器参数
        with torch.no_grad():
            for params in zip(*[netG.parameters() for netG in netG_list]):
                avg_data = sum(p.data for p in params) / num_worker
                for p in params:
                    p.data.copy_(avg_data)

    if epoch % 50 == 0:
        loss_D_list.append(np.mean([abs(errD.cpu().detach().numpy()) for errD in errD_list]))  # 保存每个epoch的D损失（绝对值）
        loss_G_list.append(np.mean([abs(errG.cpu().detach().numpy()) for errG in errG_list]))  # 保存每个epoch的G损失（绝对值）
        print(f'Epoch [{epoch}/{num_epochs}], Loss_D: {loss_fake_list}, Loss_G: {errG_list}')

    # Saving the model and images
    if epoch % 1000 == 0:
        torch.save(netG_list[0].state_dict(), f"ODDFedGAN/models/netG_{epoch}.pt")
        vutils.save_image(fake_list[0].detach(), f"ODDFedGAN/gen_images/fake_samples_epoch_{epoch}.png", normalize=True)

    # 清理显存（放在每轮结束）释放未使用的显存，避免累积导致CUDA 内存溢出（out of memory，OOM）
    gc.collect()
    torch.cuda.empty_cache()

# 使用真实数据绘制损失曲线
epochs = np.arange(0, num_epochs, 50)
# 去除第一个点（epoch=0）
epochs = epochs[1:]
loss_D_trimmed = loss_D_list[1:]
loss_G_trimmed = loss_G_list[1:]

plt.figure(figsize=(8, 5))
plt.plot(epochs, loss_D_trimmed, label="Loss_D (Discriminator)", color='red', marker='o')
plt.plot(epochs, loss_G_trimmed, label="Loss_G (Generator)", color='blue', marker='s')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Curve of FedGAN Training")
plt.ylim(0, 0.6)  # 固定纵坐标范围
plt.legend()
plt.grid()
plt.show()


# 假设每个节点有一个随机的x和y坐标，模拟节点位置
nodes_x = np.random.uniform(0, 10, num_worker)
nodes_y = np.random.uniform(0, 10, num_worker)

# 假设我们用欧氏距离来表示节点之间的关系
distance_matrix = np.zeros((num_worker, num_worker))

# 计算每两个节点之间的欧氏距离
for i in range(num_worker):
    for j in range(num_worker):
        distance_matrix[i, j] = np.sqrt((nodes_x[i] - nodes_x[j])**2 + (nodes_y[i] - nodes_y[j])**2)

# 使用节点损失数据（这里使用损失作为权重）来修改热力图
node_losses = np.array([np.mean(abs(errD.cpu().detach().numpy())) for errD in errD_list])  # 获取每个节点的D损失的绝对值平均值

# 将损失数据添加到距离矩阵中，反映每个节点的负载
for i in range(num_worker):
    for j in range(num_worker):
        distance_matrix[i, j] += node_losses[i] * 0.1  # 通过损失调整距离，影响热力图的颜色强度

# 绘制节点关系热力图，统一颜色映射范围（例如数值范围设为0~10）
plt.figure(figsize=(8, 6))
sns.heatmap(
    distance_matrix,
    annot=True,
    cmap='coolwarm',
    xticklabels=range(num_worker),
    yticklabels=range(num_worker),
    cbar_kws={'label': 'Relationship (Distance + Loss)'},
    vmin=0,   # 最小值
    vmax=10   # 最大值，根据你预期的关系值范围来设定
)
plt.title("Node Heatmap")
plt.xlabel("Node Index")
plt.ylabel("Node Index")
plt.show()


# 假设每个节点有一个随机的x和y坐标
nodes_x = np.random.uniform(0, 10, num_worker)
nodes_y = np.random.uniform(0, 10, num_worker)

# 绘制节点分布图
plt.figure(figsize=(6, 5))
plt.scatter(nodes_x, nodes_y, c=node_losses, cmap='coolwarm', marker='o', label='Nodes')
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.title("Node Distribution ")
plt.colorbar(label='Loss')
plt.legend()
plt.grid()
plt.show()


