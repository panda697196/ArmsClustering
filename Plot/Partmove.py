import os
import shutil

# 源文件夹和目标文件夹的路径
source_folder = "D:/Dev/Emilya/Netural_CUT"
target_folder = "D:/Dev/Emilya/Netural_CUT_Parts"

# 创建目标文件夹（如果不存在）
if not os.path.exists(target_folder):
    os.makedirs(target_folder)

# 定义要移动的文件列表
files_to_move = [
    "Nt1BS1_Actor10_720_960.bvh",
    "Nt1SDBS1_Actor10_960_1200.bvh",
    "Nt1SD_Actor10_1200_1440.bvh",
    "Nt1SD_Actor10_960_1200.bvh",
    "Nt1BS1_Actor2_240_480.bvh",
    "Nt1SDBS1_Actor2_480_720.bvh",
    "Nt1SD_Actor2_480_720.bvh",
    "Nt1SD_Actor2_720_960.bvh",
    "Nt1BS1_Actor41_240_480.bvh",
    "Nt1SDBS1_Actor41_480_720.bvh",
    "Nt1SD_Actor41_480_720.bvh",
    "Nt1BS2_Actor10_720_960.bvh",
    "Nt1BS2_Actor10_960_1200.bvh",
    "Nt1SDBS2_Actor10_1200_1440.bvh",
    "Nt1SDBS2_Actor10_1440_1680.bvh",
    "Nt1SD_Actor10_3120_3360.bvh",
    "Nt1SD_Actor10_3360_3600.bvh",
    "Nt1BS3_Actor10_960_1200.bvh",
    "Nt1SDBS3_Actor10_1440_1680.bvh",
    "Nt1SD_Actor10_5520_5760.bvh",
    "Nt1BS4_Actor10_240_480.bvh",
    "Nt1SDBS4_Actor10_960_1200.bvh",
    "Nt1SD_Actor10_7200_7440.bvh",
    "Nt1BS4_Actor10_480_720.bvh",
    "Nt1SDBS4_Actor10_1200_1440.bvh",
    "Nt1SD_Actor10_7440_7680.bvh",
    "Nt1BS4_Actor10_960_1200.bvh",
    "Nt1SDBS4_Actor10_1680_1920.bvh",
    "Nt1SD_Actor10_7920_8160.bvh",
    "Nt1SD1_Actor2_0_240.bvh",
    "Nt1SDBS1_Actor2_0_240.bvh",
    "Nt1SD_Actor2_0_240.bvh",
    "Nt1SD1_Actor9_0_240.bvh",
    "Nt1SDBS1_Actor9_0_240.bvh",
    "Nt1SD_Actor9_0_240.bvh",
    "Nt1SD3_Actor31_0_240.bvh",
    "Nt1SDBS3_Actor31_0_240.bvh",
    "Nt1SD_Actor31_1200_1440.bvh",
    "Nt1SD4_Actor10_0_240.bvh",
    "Nt1SDBS4_Actor10_0_240.bvh",
    "Nt1SD_Actor10_6240_6480.bvh",
    "Nt1SD4_Actor10_240_480.bvh",
    "Nt1SDBS4_Actor10_240_480.bvh",
    "Nt1SD_Actor10_6480_6720.bvh",
    "Nt1SW1_Actor8_0_240.bvh",
    "Nt1SW_Actor8_0_240.bvh",
    "Nt2SW1_Actor8_0_240.bvh",
    "Nt2SW_Actor8_0_240.bvh",
    "Nt1SW1_Actor8_240_480.bvh",
    "Nt1SW_Actor8_240_480.bvh",
    "Nt2SW1_Actor8_240_480.bvh",
    "Nt2SW_Actor8_240_480.bvh",
    "Nt2BS2_Actor4_0_240.bvh",
    "Nt2SDBS2_Actor4_480_720.bvh",
    "Nt2SD_Actor4_1200_1440.bvh",
    "Nt2BS3_Actor2_240_480.bvh",
    "Nt2SDBS3_Actor2_480_720.bvh",
    "Nt2SD_Actor2_2640_2880.bvh",
    "Nt2BS3_Actor4_0_240.bvh",
    "Nt2SDBS3_Actor4_480_720.bvh",
    "Nt2SD_Actor4_2400_2640.bvh",
    "Nt2BS4_Actor10_240_480.bvh",
    "Nt2SDBS4_Actor10_480_720.bvh",
    "Nt2SD_Actor10_3600_3840.bvh",
    "Nt2BS4_Actor7_240_480.bvh",
    "Nt2SDBS4_Actor7_480_720.bvh",
    "Nt2SD_Actor7_3600_3840.bvh",
    "Nt2SD1_Actor10_0_240.bvh",
    "Nt2SDBS1_Actor10_0_240.bvh",
    "Nt2SD_Actor10_0_240.bvh",
    "Nt2SD1_Actor1_0_240.bvh",
    "Nt2SDBS1_Actor1_0_240.bvh",
    "Nt2SD_Actor1_0_240.bvh",
    "Nt2SD1_Actor2_0_240.bvh",
    "Nt2SDBS1_Actor2_0_240.bvh",
    "Nt2SD_Actor2_0_240.bvh",
    "Nt2SD1_Actor3_0_240.bvh",
    "Nt2SDBS1_Actor3_0_240.bvh",
    "Nt2SD_Actor3_0_240.bvh",
    "Nt2SD1_Actor9_0_240.bvh",
    "Nt2SDBS1_Actor9_0_240.bvh",
    "Nt2SD_Actor9_0_240.bvh",
    "Nt2SD2_Actor10_0_240.bvh",
    "Nt2SDBS2_Actor10_0_240.bvh",
    "Nt2SD_Actor10_960_1200.bvh",
    "Nt3SD4_Actor4_0_240.bvh",
    "Nt3SDBS4_Actor4_0_240.bvh",
    "Nt3SD_Actor4_2640_2880.bvh"
]

# 移动文件
for file_to_move in files_to_move:
    source_path = os.path.join(source_folder, file_to_move)
    target_path = os.path.join(target_folder, file_to_move)

    if os.path.exists(source_path):
        shutil.move(source_path, target_path)
        print(f"Moved {file_to_move} to {target_folder}")
    else:
        print(f"File {file_to_move} not found in the source folder.")

print("File move completed.")
