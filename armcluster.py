import numpy as np
import pandas as pd
from sklearn.cluster import KMeans # 选择k-means算法，你也可以换成其他的
import glob # 用来读取文件夹中的所有文件

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
        # 选择手臂部分的数据，这里假设手臂部分有6个特征，你可以根据你的数据格式进行修改
        arm_data = data[:, :6]
        # 返回手臂部分数据，作为这个文件的特征向量
        return arm_data # 修改：不再返回平均值，而是返回所有特征

# 定义文件夹路径，这里假设所有的bvh文件都在同一个文件夹中，你可以根据你的实际情况进行修改
folder_path = 'D:/Dev/AbeTomoaki/'
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
# 将列表中的所有特征向量拼接成一个大矩阵
feature_matrix = np.concatenate(feature_vectors, axis=0) # 新增：将列表转换为大矩阵
# 获取每个文件对应的样本数（假设每个文件有相同数量的帧）
sample_per_file = feature_matrix.shape[0] // len(file_names) # 新增：计算每个文件的样本数
# 将大矩阵按文件划分为多个小矩阵，并存入列表中
feature_matrices = [feature_matrix[i*sample_per_file:(i+1)*sample_per_file, :] for i in range(len(file_names))] # 新增：将大矩阵划分为小矩阵列表

print(feature_matrix.shape)
print(feature_matrix)

# 创建一个k-means聚类器对象，这里假设你想将动作分为4类，你可以根据你的实际情况进行修改
kmeans = KMeans(n_clusters=4)
# 创建一个空列表，用来存放每个文件的聚类结果和标签
cluster_results = [] # 新增：创建空列表用来存放聚类结果
cluster_labels = [] # 新增：创建空列表用来存放聚类标签
# 遍历列表中的每个小矩阵
for feature_matrix in feature_matrices: # 新增：遍历小矩阵列表
    # 对小矩阵进行聚类，并获取聚类结果和标签
    cluster_result = kmeans.fit_predict(feature_matrix) # 修改：对小矩阵进行聚类
    cluster_label = kmeans.labels_
    # 将聚类结果和标签添加到列表中
    cluster_results.append(cluster_result) # 新增：将聚类结果添加到列表中
    cluster_labels.append(cluster_label) # 新增：将聚类标签添加到列表中
# 将列表中的所有聚类结果和标签拼接成一个大数组
cluster_result = np.concatenate(cluster_results, axis=0) # 新增：将列表转换为大数组
cluster_label = np.concatenate(cluster_labels, axis=0) # 新增：将列表转换为大数组

# 打印聚类结果和标签
print('Cluster result:', cluster_result)
print('Cluster labels:', kmeans.labels_)
# 导入matplotlib库
import matplotlib.pyplot as plt
# 绘制散点图，横轴为样本索引，纵轴为特征值，颜色为聚类标签
# 绘制散点图，横轴为样本索引，纵轴为第一个特征值，颜色为聚类标签
# 绘制散点图，横轴为样本索引，纵轴为第一个特征值，颜色为聚类标签
plt.scatter(range(len(feature_matrix[:, 0])), feature_matrix[:, 0],
            c=cluster_result[:len(feature_matrix[:, 0])])

# 添加标题和坐标轴标签
plt.title('Clustering result')
plt.xlabel('Sample index')
plt.ylabel('Feature value')
# 显示图形
plt.show()
