import os
import shutil
from tqdm import tqdm

# 源文件夹的路径
source_folder = 'D:/Dev/Emilya/All_BvH_Files/'

# 目标文件夹的路径
destination_folder = 'D:/Dev/Emilya/AllBVH/'

# 获取源文件夹中所有BVH文件的路径
bvh_files = [os.path.join(root, filename) for root, _, filenames in os.walk(source_folder) for filename in filenames if filename.endswith('.bvh')]

# 创建目标文件夹（如果不存在）
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# 使用tqdm创建进度条
with tqdm(total=len(bvh_files), desc="复制文件") as pbar:
    for source_file in bvh_files:
        filename = os.path.basename(source_file)
        destination_file = os.path.join(destination_folder, filename)

        # 如果目标文件夹中已存在同名文件，可以选择重命名或跳过
        if os.path.exists(destination_file):
            print(f"文件 {filename} 已存在于目标文件夹中，跳过。")
        else:
            # 复制文件到目标文件夹
            shutil.copy2(source_file, destination_file)
            pbar.update(1)  # 更新进度条
            pbar.set_postfix(file=filename)  # 显示当前正在处理的文件名

print("任务完成。")
