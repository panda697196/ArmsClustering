import numpy as np
import pandas as pd
import glob
from sklearn.preprocessing import StandardScaler
from fastdtw import fastdtw
from tqdm import tqdm
from scipy.cluster.hierarchy import linkage, to_tree, dendrogram
from sklearn.cluster import AgglomerativeClustering  # 修改导入语句
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import re
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
        # 选择手臂部分的数据，这里假设手臂部分有6个特征，您可以根据您的数据格式进行修改
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

# 使用层次聚类方法进行聚类
# linkage函数计算距离矩阵的层次聚类，'ward'代表使用ward方法进行聚类
Z = linkage(dtw_distances_scaled, method='ward')

# 使用AgglomerativeClustering进行层次聚类，不需要指定聚类簇的数量
agg_clustering = AgglomerativeClustering(distance_threshold=0, n_clusters=None, linkage='ward')
agg_clustering.fit(dtw_distances_scaled)

# 获取聚类结果
cluster_label = agg_clustering.labels_

# 将聚类结果按照不同的簇进行存储
cluster_files = {}
for i, label in enumerate(cluster_label):
    if label not in cluster_files:
        cluster_files[label] = []
    cluster_files[label].append(file_names[i])

# 统计每个聚类簇中不同感情的数量
emotion_counts = {}
for cluster, files in cluster_files.items():
    emotions = [re.search(r'_(\w+)_\d', file).group(1) for file in files]  # Extract emotion using regex
    emotion_counts[cluster] = dict(pd.Series(emotions).value_counts())

# 打印聚类结果和感情数量统计
for cluster, emotion_count in emotion_counts.items():
    print(f"Cluster {cluster}:")
    for emotion, count in emotion_count.items():
        print(f"    {emotion}: {count}")

# 降维到二维空间
pca = PCA(n_components=2)
dtw_distances_pca = pca.fit_transform(dtw_distances_scaled)

# 绘制聚类图
plt.figure(figsize=(10, 8))
for i, label in enumerate(cluster_label):
    plt.scatter(dtw_distances_pca[i, 0], dtw_distances_pca[i, 1], color=plt.cm.Set1(label))
plt.title('Hierarchical Clustering in 2D')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()

# 绘制层次聚类树状图（长条形）
plt.figure(figsize=(10, 6))
dendrogram(Z, labels=file_names, orientation='top')
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample index')
plt.ylabel('Distance')
plt.show()

# 绘制不同聚类簇的感情数量统计图
plt.figure(figsize=(12, 6))
plt.bar(emotion_counts.keys(), [sum(count.values()) for count in emotion_counts.values()])
plt.xlabel('Cluster')
plt.ylabel('Total Count')
plt.title('Emotion Counts in Each Cluster')
plt.show()