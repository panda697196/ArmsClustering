import numpy as np
import pandas as pd
import tensorflow as tf
import glob
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from fastdtw import fastdtw
from tqdm import tqdm
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns

# 设置TensorFlow在GPU上运行
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

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
folder_path = 'C:/Users/YUXUAN TENG/Downloads/MotionBVH_rotated/'
#C:/Users/YUXUAN TENG/Downloads/MotinBVH_Part/
#D:\Dev\AbeTomoaki/
#C:\Users\YUXUAN TENG\Downloads\cmuconvert-daz-nohipcorrect-01-09\part
#C:\Users\YUXUAN TENG\Downloads\MotionBVH_rotated

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

# 将特征向量列表转换为一个大的二维数组
all_feature_vectors = np.vstack(feature_vectors)

# 标准化特征向量（可选，但在某些情况下可能有帮助）
scaler = StandardScaler()
scaled_feature_vectors = scaler.fit_transform(all_feature_vectors)

# 使用PCA进行降维
n_components = 2  # 降至2维，你可以根据需要进行调整
pca = PCA(n_components=n_components)
reduced_feature_vectors = pca.fit_transform(scaled_feature_vectors)

# 在2D空间中绘制降维后的数据
plt.figure(figsize=(10, 8))
sns.scatterplot(x=reduced_feature_vectors[:, 0], y=reduced_feature_vectors[:, 1], alpha=0.5)
plt.title('PCA Reduced Feature Vectors')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# 使用DBSCAN进行聚类
eps = 3  # 邻域半径，需要根据数据进行调整
min_samples = 3  # 最小样本数，需要根据数据进行调整
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
cluster_labels = dbscan.fit_predict(reduced_feature_vectors)

# 绘制聚类结果
plt.figure(figsize=(10, 8))
sns.scatterplot(x=reduced_feature_vectors[:, 0], y=reduced_feature_vectors[:, 1], hue=cluster_labels, palette='Set1', alpha=0.7)
plt.title('DBSCAN Clustering Result')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()

