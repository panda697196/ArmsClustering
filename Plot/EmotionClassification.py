import os
import shutil
import re

# 指定包含文件的文件夹路径
source_folder = 'C:/Users/YUXUAN TENG/Downloads/MotionBVH_rotated_cut/2022/'

# 感情类别列表
emotions = ['joy', 'neutral', 'pride', 'sadness', 'shame', 'surprise', 'anger', 'contempt', 'disgust', 'fear', 'gratitude', 'guilt', 'jealousy']

# 创建对应的文件夹
for emotion in emotions:
    emotion_folder = os.path.join(source_folder, emotion)
    if not os.path.exists(emotion_folder):
        os.makedirs(emotion_folder)

# 遍历源文件夹中的文件并根据文件名将它们移动到相应的文件夹
for filename in os.listdir(source_folder):
    if os.path.isfile(os.path.join(source_folder, filename)):
        for emotion in emotions:
            if emotion in filename:
                source_file_path = os.path.join(source_folder, filename)
                destination_folder = os.path.join(source_folder, emotion)
                destination_file_path = os.path.join(destination_folder, filename)

                shutil.move(source_file_path, destination_file_path)
                break

print("文件分类完成。")