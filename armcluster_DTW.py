import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import glob
from sklearn.preprocessing import StandardScaler
from fastdtw import fastdtw
from tqdm import tqdm
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

 #定义文件夹路径，这里假设所有的bvh文件都在同一个文件夹中，您可以根据您的实际情况进行修改
folder_path = 'D:\Dev\AbeTomoaki/'
#D:\Dev\AbeTomoaki
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

# 计算所有特征向量之间的DTW距离矩阵
dtw_distances = np.zeros((len(feature_vectors), len(feature_vectors)))
for i in range(len(feature_vectors)):
    for j in range(len(feature_vectors)):
        dtw_distances[i, j], _ = fastdtw(feature_vectors[i], feature_vectors[j])

# 使用StandardScaler对DTW距离矩阵进行标准化
scaler = StandardScaler()
dtw_distances_scaled = scaler.fit_transform(dtw_distances)

# 创建一个k-means聚类器对象
kmeans = KMeans(n_clusters=4)
# 对标准化后的DTW距离矩阵进行聚类，并获取聚类结果和标签
cluster_result = kmeans.fit_predict(dtw_distances_scaled)
cluster_label = kmeans.labels_

# 打印聚类结果和标签
print('Cluster result:', cluster_result)
print('Cluster labels:', kmeans.labels_)
cluster_files = {}
for i, label in enumerate(cluster_label):
    if label not in cluster_files:
        cluster_files[label] = []
    cluster_files[label].append(file_names[i])

# Print the files in each cluster
for cluster, files in cluster_files.items():
    print(f"Files in Cluster {cluster}:")
    for file in files:
        print(file)

# 导入matplotlib库
import matplotlib.pyplot as plt
# 绘制散点图，横轴为样本索引，纵轴为第一个DTW距离值，颜色为聚类标签
plt.scatter(range(len(dtw_distances)), dtw_distances[:, 0], c=cluster_result[:len(dtw_distances)]) # 修改：让颜色参数c和横轴或纵轴长度

# 添加标题和坐标轴标签
plt.title('Clustering result')
plt.xlabel('Sample index')
plt.ylabel('DTW Distance')
# 显示图形
plt.show()

# 绘制聚类图
plt.figure(figsize=(8, 6))
for i in range(4):
    cluster_points = dtw_distances[cluster_label == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i + 1}')

plt.title('Clustering Visualization')
plt.xlabel('DTW Distance to Cluster Center 1')
plt.ylabel('DTW Distance to Cluster Center 2')
plt.legend()
plt.show()

# 绘制热图
plt.figure(figsize=(8, 6))
plt.imshow(dtw_distances, cmap='viridis', origin='lower')
plt.colorbar()
plt.title('DTW Distance Heatmap')
plt.xlabel('Sample Index')
plt.ylabel('Sample Index')
plt.show()