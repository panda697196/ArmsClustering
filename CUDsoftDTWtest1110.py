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
from fastdtw import fastdtw
from tqdm import tqdm
from scipy.spatial.distance import euclidean
from sklearn.cluster import KMeans
import numpy as np
import torch
from soft_dtw_cuda import SoftDTW


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
folder_path = 'D:/Data/AllNarm_cut/'
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

# 将 feature_vectors 转换为 PyTorch 张量
data = torch.tensor(feature_vectors)

# # 计算Soft DTW距离矩阵
# soft_dtw = SoftDTW(use_cuda=True, gamma=1.0)  # 假设已经导入了正确的 Soft DTW 实现
# soft_dtw_distances = []  # 用于存储 Soft DTW 距离矩阵
#
# for i in range(len(data)):
#     for j in range(i+1, len(data)):
#         # 计算 Soft DTW 距离
#         distance = soft_dtw(data[i:i+1], data[j:j+1]).item()
#         soft_dtw_distances.append(distance)
#
# # 将距离转换为对称矩阵形式
# n = len(data)
# distance_matrix = np.zeros((n, n))
# indices = np.triu_indices(n, k=1)
# distance_matrix[indices] = soft_dtw_distances
# distance_matrix[(indices[1], indices[0])] = soft_dtw_distances

if os.path.exists('soft_dtw_distances.npy') and os.path.exists('soft_dtw_distances_scaled.npy'):
    dtw_distances = np.load('soft_dtw_distances.npy')
    dtw_distances_scaled = np.load('soft_dtw_distances_scaled.npy')
else:
    # # 计算所有特征向量之间的DTW距离矩阵
    # dtw_distances = np.zeros((len(feature_vectors), len(feature_vectors)))
    # for i in tqdm(range(len(feature_vectors))):
    #     for j in range(len(feature_vectors)):
    #         dtw_distances[i, j], _ = fastdtw(feature_vectors[i], feature_vectors[j])
    #
    # # 使用StandardScaler对DTW距离矩阵进行标准化
    # scaler = StandardScaler()
    # dtw_distances_scaled = scaler.fit_transform(dtw_distances)

    # 计算Soft DTW距离矩阵
    soft_dtw = SoftDTW(use_cuda=True, gamma=1.0)  # 假设已经导入了正确的 Soft DTW 实现
    soft_dtw_distances = []  # 用于存储 Soft DTW 距离矩阵

    for i in range(len(data)):
        for j in range(i+1, len(data)):
            # 计算 Soft DTW 距离
            distance = soft_dtw(data[i:i+1], data[j:j+1]).item()
            soft_dtw_distances.append(distance)

    # 将距离转换为对称矩阵形式
    n = len(data)
    distance_matrix = np.zeros((n, n))
    indices = np.triu_indices(n, k=1)
    distance_matrix[indices] = soft_dtw_distances
    distance_matrix[(indices[1], indices[0])] = soft_dtw_distances
    scaler = StandardScaler()
    soft_dtw_distances_scaled = scaler.fit_transform(soft_dtw_distances)

    # 将计算得到的DTW距离保存到文件中
    np.save('soft_dtw_distances.npy', soft_dtw_distances)
    np.save('soft_dtw_distances_scaled.npy', soft_dtw_distances_scaled)


# 创建一个DBSCAN聚类器对象
dbscan = DBSCAN(eps=10, min_samples=1)  # 可根据实际情况调整eps和min_samples参数
# 对标准化后的DTW距离矩阵进行聚类，并获取聚类标签
cluster_label = dbscan.fit_predict(soft_dtw_distances_scaled)

# 打印聚类标签
print('Cluster labels:', cluster_label)


# # 创建一个K-Means聚类器对象，指定要分成的簇数（k）
# k = 2  # 你可以根据实际情况调整簇数
# kmeans = KMeans(n_clusters=k)
#
# # 对标准化后的DTW距离矩阵进行聚类，并获取聚类标签
# cluster_label = kmeans.fit_predict(dtw_distances_scaled)
#
# # 打印聚类标签
# print('Cluster labels:', cluster_label)



# 将文件名与对应的聚类标签保存到字典中，并统计每个聚类中的文件数量
cluster_files = {}
for i, label in enumerate(cluster_label):
    if label not in cluster_files:
        cluster_files[label] = []
    cluster_files[label].append(file_names[i])

# 打印每个聚类中的文件及文件数量
for cluster, files in cluster_files.items():
    print(f"Cluster {cluster} contains {len(files)} files:")
    for file in files:
        print(file)

# 创建目录并移动文件到相应的聚类文件夹
output_folder = 'D:/Data/AllNarm_cut/clustered_files'
os.makedirs(output_folder, exist_ok=True)

for cluster, files in cluster_files.items():
    cluster_folder = os.path.join(output_folder, f'Cluster_{cluster}')
    os.makedirs(cluster_folder, exist_ok=True)  # 创建聚类文件夹

    # 移动文件到相应的聚类文件夹
    for file in files:
        file_name = os.path.basename(file)
        destination_file = os.path.join(cluster_folder, file_name)
        shutil.copy(file, destination_file)

print("Files have been moved to the respective cluster folders.")

feature_vectors = np.array(feature_vectors)  # 转换为NumPy数组
feature_vectors = feature_vectors.reshape(feature_vectors.shape[0], -1)

# 创建PCA对象，指定降维到的目标维度（通常为2或3）
pca_2d = PCA(n_components=2)
pca_3d = PCA(n_components=3)

# 对数据进行PCA降维
X_pca_2d = pca_2d.fit_transform(feature_vectors)
X_pca_3d = pca_3d.fit_transform(feature_vectors)

# 创建一个新的图形窗口
plt.figure(figsize=(16, 6))  # 增加图形窗口的宽度，以便同时显示两个子图

# 2D PCA可视化
plt.subplot(1, 2, 1)  # 创建第一个子图
plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=cluster_label, cmap='viridis', alpha=0.5)
plt.title('2D PCA Visualization')
plt.xlabel('PCA Dimension 1')
plt.ylabel('PCA Dimension 2')

# 添加颜色条
cbar = plt.colorbar()
cbar.set_label('Cluster Label')

# 创建图例
legend_labels = list(set(cluster_label))
for label in legend_labels:
    plt.scatter([], [], label=f'Cluster {label}', c='k')

# 显示图例
plt.legend()

# 3D PCA可视化
ax = plt.subplot(1, 2, 2, projection='3d')  # 创建第二个子图，使用3D投影
scatter = ax.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2], c=cluster_label, cmap='viridis', alpha=0.5)
ax.set_title('3D PCA Visualization')
ax.set_xlabel('PCA Dimension 1')
ax.set_ylabel('PCA Dimension 2')
ax.set_zlabel('PCA Dimension 3')

# 添加颜色条
cbar = plt.colorbar(scatter)
cbar.set_label('Cluster Label')

# 显示图形
plt.show()

# 计算距离矩阵（欧氏距离）
distances = pairwise_distances(feature_vectors, metric='euclidean')

# 绘制距离直方图
plt.hist(distances, bins=50, alpha=0.5)
plt.title("Distance Histogram")
plt.xlabel("Distance")
plt.ylabel("Frequency")
plt.show()

plt.imshow(distances, cmap='viridis', aspect='auto')
plt.title("Distance Matrix")
plt.colorbar()
plt.show()

# 绘制距离直方图
plt.hist(soft_dtw_distances.flatten(), bins=50, alpha=0.5)
plt.title("DTW Distance Histogram")
plt.xlabel("Distance")
plt.ylabel("Frequency")
plt.show()

# 可视化距离矩阵
plt.imshow(soft_dtw_distances, cmap='viridis', aspect='auto')
plt.title("DTW Distance Matrix")
plt.colorbar()
plt.show()