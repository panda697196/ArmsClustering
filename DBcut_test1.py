import numpy as np
import pandas as pd
import glob
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')
from collections import Counter
import re
import seaborn as sns

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
folder_path = 'C:/Users/YUXUAN TENG/Downloads/CutParts/'

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

# 将列表转换为numpy数组
feature_vectors = np.array(feature_vectors)

# 将三维数组转换为二维数组
feature_vectors = feature_vectors.reshape(feature_vectors.shape[0], -1)

# 使用原始数据进行DBSCAN聚类
dbscan = DBSCAN(eps=0.5, min_samples=1)
cluster_label = dbscan.fit_predict(feature_vectors)

# 打印聚类标签
print('Cluster labels:', cluster_label)

# 将文件名与对应的聚类标签保存到字典中
cluster_files = {}
for i, label in enumerate(cluster_label):
    if label not in cluster_files:
        cluster_files[label] = []
    cluster_files[label].append(file_names[i])

# 打印每个聚类中的文件
for cluster, files in cluster_files.items():
    print(f"Files in Cluster {cluster}:")
    for file in files:
        print(file)

# 绘制散点图
plt.figure(figsize=(8, 6))
unique_labels = np.unique(cluster_label)
for label in unique_labels:
    if label == -1:
        cluster_points = feature_vectors[cluster_label == label]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label='Noise', alpha=0.5)
    else:
        cluster_points = feature_vectors[cluster_label == label]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {label}', alpha=0.5)

plt.title('DBSCAN Clustering Visualization')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.savefig('dbscan_cluster_visualization.png')
