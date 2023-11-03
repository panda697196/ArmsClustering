import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import matplotlib as mpl
#mpl.use('Agg')
from collections import Counter
import re
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os
import shutil

# 定义一个简单的VAE模型
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z):
        return self.decoder(z)

# 准备BVH数据并转换为PyTorch张量
# 替换下面的代码为您的数据加载和预处理步骤
# bvhs = load_and_preprocess_bvh_data()
# 定义读取bvh文件中手臂部分数据的函数
def read_arm_data(filename):
    # 打开文件
    with open(filename, 'r') as f:
        # 读取所有行
        lines = f.readlines()
        # 找到MOTION行的索引
        motion_index = lines.index('MOTION\n')
        # 找到Frames:行的索引
        frames_index = motion_index + 1
        # 获取帧数
        frames = int(lines[frames_index].split()[1])
        # 找到Frame Time:行的索引
        frame_time_index = motion_index + 2
        # 获取帧时间
        frame_time = float(lines[frame_time_index].split()[2])
        # 获取数据部分的所有行
        data_lines = lines[frame_time_index + 1:]
        # 将数据转换为numpy数组
        data = np.array([list(map(float, line.split())) for line in data_lines])
        # 选择手臂部分的数据
        arm_data = data[:, 27:27 + 24]
        # 返回手臂部分数据作为这个文件的特征向量
        return arm_data

# 定义文件夹路径
folder_path = 'D:/Dev/Emilya/AllNArm_CUT/'
#Emillyacut CutParts MotionBVH_rotated_cut D:/Dev/Emilya/Netural_CUT_Parts

# 获取文件夹中所有的bvh文件名
file_names = glob.glob(folder_path + '*.bvh')

# 创建一个空的列表，用来存放所有文件的特征向量
feature_vectors = []

# 遍历所有文件名
for file_name in file_names:
    # 调用函数，读取手臂部分数据，并返回特征向量
    feature_vector = read_arm_data(file_name)
    # 将特征向量添加到列表中
    feature_vectors.append(feature_vector)

# 将列表转换为NumPy数组
feature_vectors_array = np.array(feature_vectors)

# 将NumPy数组转换为PyTorch张量
bvhs_tensor = torch.Tensor(feature_vectors_array)


# 定义VAE模型
input_dim = bvhs_tensor.size(1)  # 根据数据维度定义输入维度
latent_dim = 32  # 潜在空间维度
vae = VAE(input_dim, latent_dim)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(vae.parameters(), lr=0.001)

# 训练VAE模型
num_epochs = 100
for epoch in range(num_epochs):
    for data in bvhs_tensor:
        # 前向传播
        mu, logvar = vae.encode(data)
        z = vae.reparameterize(mu, logvar)
        reconstruction = vae.decode(z)

        # 计算损失
        reconstruction_loss = criterion(reconstruction, data)
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = reconstruction_loss + kld_loss

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# 提取潜在特征
latent_features = []
for data in bvhs_tensor:
    mu, logvar = vae.encode(data)
    z = vae.reparameterize(mu, logvar)
    latent_features.append(z)

# latent_features 现在包含了BVH数据的潜在特征
from sklearn.cluster import KMeans

# 聚类的簇数
n_clusters = 2  # 适应您的需求

# 初始化K均值聚类器
kmeans = KMeans(n_clusters=n_clusters)

# 对潜在特征进行聚类
cluster_labels = kmeans.fit_predict(latent_features)

# 打印聚类标签
print('Cluster labels:', cluster_labels)