import numpy as np
import pandas as pd
import glob
from sklearn.preprocessing import StandardScaler
from fastdtw import fastdtw
from tqdm import tqdm
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
folder_path = '/home/uken/workspace/EParts/'
#C:/Users/YUXUAN TENG/Downloads/MotinBVH_Part/
#/home/uken/workspace/MotionBVH_rotated
#D:\Dev\AbeTomoaki/
#C:\Users\YUXUAN TENG\Downloads\cmuconvert-daz-nohipcorrect-01-09\part

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
for i in tqdm(range(len(feature_vectors))):
    for j in range(len(feature_vectors)):
        dtw_distances[i, j], _ = fastdtw(feature_vectors[i], feature_vectors[j])

# 使用StandardScaler对DTW距离矩阵进行标准化
scaler = StandardScaler()
dtw_distances_scaled = scaler.fit_transform(dtw_distances)

# 创建一个DBSCAN聚类器对象
dbscan = DBSCAN(eps=3, min_samples=3)
# 对标准化后的DTW距离矩阵进行聚类，并获取聚类标签
cluster_label = dbscan.fit_predict(dtw_distances_scaled)

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

# 匹配文件名中的人名和感情信息
person_emotion_pattern = re.compile(r'(\d{4}-\d{2}-\d{2})_(\w+)_(\w+)_\d+_\w+\.bvh')
# 将文件名与对应的聚类标签保存到字典中
cluster_files = {}
for i, label in enumerate(cluster_label):
    if label not in cluster_files:
        cluster_files[label] = []
    cluster_files[label].append(file_names[i])

# 遍历每个聚类中的文件，并提取人名和感情信息
for cluster, files in cluster_files.items():
    print(f"Person and emotions in Cluster {cluster}:")
    person_emotions = []
    for file in files:
        # 使用正则表达式匹配人名和感情信息，并添加到列表中
        match = person_emotion_pattern.search(file)
        if match:
            person_emotions.append((match.group(2), match.group(3)))
    # 使用Counter对象统计每种组合出现的次数，并打印结果
    person_emotion_counts = Counter(person_emotions)
    for (person, emotion), count in person_emotion_counts.items():
        print(f"{person} - {emotion}: {count} times")



# 绘制聚类图
plt.figure(figsize=(8, 6))
unique_labels = np.unique(cluster_label)
for label in unique_labels:
    if label == -1:
        cluster_points = dtw_distances[cluster_label == label]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label='Noise', alpha=0.5)
    else:
        cluster_points = dtw_distances[cluster_label == label]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {label}', alpha=0.5)

plt.title('DBSCAN Clustering Visualization')
plt.xlabel('DTW Distance to Cluster Center 1')
plt.ylabel('DTW Distance to Cluster Center 2')
plt.legend()
# 添加标题和坐标轴标签
plt.title('DBSCAN Clustering Visualization')
plt.xlabel('MDS Dimension 1')
plt.ylabel('MDS Dimension 2')

# 显示图形
plt.show()
# 保存图形
plt.savefig('cluster1.png')