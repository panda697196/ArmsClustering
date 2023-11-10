# 导入所需的库
import os

# 指定输入和输出文件夹以及帧数间隔
input_folder = 'D:/Data/Allarm/'
#C:\Users\YUXUAN TENG\Downloads\MotinBVH_Parts MotionBVH_rotated
output_folder = 'D:/Data/Allarm_cut/'
#Emillyacut MotionBVH_rotated_cut
frame_interval = 240  # 240帧，代表两秒的数据

# 获取输入文件夹中的所有BVH文件
bvh_files = [file for file in os.listdir(input_folder) if file.endswith('.bvh')]

# 遍历每个BVH文件
for bvh_file in bvh_files:
    with open(os.path.join(input_folder, bvh_file), 'r') as file:
        lines = file.readlines()

    # 找到"MOTION"部分的起始行
    motion_start_index = lines.index("MOTION\n")

    # 提取帧数和帧时间
    frames_line = lines[motion_start_index + 1]
    frame_count = int(frames_line.split(":")[1].strip())

    # 计算切割后的帧范围
    current_frame = 0
    while current_frame + frame_interval <= frame_count:
        start_frame = current_frame
        end_frame = current_frame + frame_interval

        # 创建新的BVH文件以保存切割后的数据
        output_file_name = f"{bvh_file.split('.')[0]}_{start_frame}_{end_frame}.bvh"
        output_file_path = os.path.join(output_folder, output_file_name)

        # 修改MOTION部分的Frames行为240
        lines[motion_start_index + 1] = f"Frames: {frame_interval}\n"

        # 写入BVH文件头部信息
        with open(output_file_path, 'w') as output_file:
            output_file.writelines(lines[:motion_start_index + 3])

            # 写入切割后的帧数据
            for i in range(start_frame, end_frame):
                output_file.write(lines[motion_start_index + 3 + i])

        current_frame = end_frame

print("切割完成！")


