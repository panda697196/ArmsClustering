import numpy as np
import pandas as pd
import glob
from sklearn.preprocessing import StandardScaler
from fastdtw import fastdtw
from tqdm import tqdm
from scipy.cluster.hierarchy import linkage, fcluster, to_tree, dendrogram
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


# 定义文件夹路径，这里假设所有的bvh文件都在同一个文件夹中，您可以根据您的实际情况进行修改
folder_path = 'D:\Dev\AbeTomoaki/'
# D:\Dev\AbeTomoaki
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

# 获取不同阈值下的聚类结果，并统计每个聚类簇中不同感情的数量
emotion_counts = {}
thresholds = np.arange(0.1, 3.0, 0.1)  # 调整阈值范围，根据实际情况进行调整
for t in thresholds:
    cluster_label = fcluster(Z, t=t, criterion='distance')

    # 将聚类结果按照不同的簇进行存储
    cluster_files = {}
    for i, label in enumerate(cluster_label):
        if label not in cluster_files:
            cluster_files[label] = []
        cluster_files[label].append(file_names[i])

    # 统计每个聚类簇中不同感情的数量
    emotion_counts[t] = {}
    for cluster, files in cluster_files.items():
        emotions = [re.search(r'_(\w+)_\d', file).group(1) for file in files]  # Extract emotion using regex
        emotion_counts[t][cluster] = dict(pd.Series(emotions).value_counts())

# 打印不同阈值下的聚类结果和感情数量统计
for t, counts in emotion_counts.items():
    print(f"\nClustering with threshold = {t:.1f}")
    for cluster, emotion_count in counts.items():
        print(f"Cluster {cluster}:")
        for emotion, count in emotion_count.items():
            print(f"    {emotion}: {count}")


# 获取树的根节点
root_node = to_tree(Z)

# 绘制层次聚类树状图（长条形）
plt.figure(figsize=(10, 6))
dendrogram(Z, labels=file_names, orientation='top')
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample index')
plt.ylabel('Distance')
plt.show()

# 绘制热图（Heatmap）
plt.figure(figsize=(10, 6))
plt.imshow(dtw_distances_scaled, cmap='hot', aspect='auto', origin='lower')
plt.colorbar()
plt.title('Hierarchical Clustering Heatmap')
plt.xlabel('Sample index')
plt.ylabel('Sample index')
plt.show()

# 绘制圆形树状图（Circular Dendrogram）
def plot_circle_dendrogram(node, xs, ys):
    if node.is_leaf():
        xs.append(node.dist)
        ys.append(0)
    else:
        xs.append(node.dist)
        ys.append(1)
        for child in [node.left, node.right]:  # 使用left和right属性来获取子节点
            plot_circle_dendrogram(child, xs, ys)

plt.figure(figsize=(10, 6))
xs, ys = [], []
plot_circle_dendrogram(root_node, xs, ys)
plt.scatter(xs, ys, c='black')
plt.title('Circular Hierarchical Clustering Dendrogram')
plt.xlabel('Distance')
plt.ylabel('Depth')
plt.show()

# 绘制层次聚类网格图（Dendrogram Grid）
plt.figure(figsize=(12, 8))
dendrogram(Z, labels=file_names, orientation='left', leaf_font_size=8, above_threshold_color='gray')
plt.title('Hierarchical Clustering Dendrogram Grid')
plt.xlabel('Sample index')
plt.ylabel('Distance')
plt.show()

# 获取不重复的聚类标签
unique_labels = np.unique(cluster_label)

# 统计聚类的类数
num_clusters = len(unique_labels)
print("当前样本被聚类成", num_clusters, "类")