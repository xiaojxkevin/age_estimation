import os
import cv2
import numpy as np

def map_age_stage(age):
    if age <= 3:
        return 0
    elif age <= 6:
        return 1
    elif age <= 12:
        return 2
    elif age <= 21:
        return 3
    elif age <= 37:
        return 4
    elif age <= 65:
        return 5
    elif age <= 84:
        return 6
    else:
        return 7

# data_base = r'./data'
# origin_data = r'./data/utkface_aligned_cropped/UTKFace'

# stage = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0,6:0,7:0}

# #随机种子
# np.random.seed(2024)

# for filename in os.listdir(origin_data):
#     age = filename[:-4].split('_')[0]
#     age_stage = map_age_stage(int(age))

#     stage[age_stage] += 1
#     img_path = os.path.join(origin_data, filename)
#     #把文件名中的age替换为age_stage
#     new_filename = filename.replace(age, str(age_stage))
#     #保存到名为stage{age_stage}的文件夹中
#     save_path = os.path.join(data_base, 'stage{}'.format(age_stage))
#     if not os.path.exists(save_path):
#         os.mkdir(save_path)
#     #随机分配到训练集（85）、验证集（10）、测试集中（5）
#     rand = np.random.randint(100)
#     if rand < 85:
#         save_path = os.path.join(save_path, 'train.txt')
#     elif rand < 95:
#         save_path = os.path.join(save_path, 'val.txt')
#     else:
#         save_path = os.path.join(save_path, 'test.txt')
#     with open(save_path, 'a') as f:
#         f.write(filename + '\n')
#     #保存图片
#     img = cv2.imread(img_path)
#     cv2.imwrite(os.path.join(data_base,'utkface_aligned_cropped_stage' , new_filename), img)

#访问split文件夹中的文件,对于每个文件，读取文件中的每一行，然后将每一行的文件名中的age替换为age_stage，然后保存到split-stage文件夹对应的文件中
for filename in os.listdir("./data/split"):
    with open(os.path.join("./data/split", filename), 'r') as f:
        lines = f.readlines()
        for line in lines:
            age = line[:-4].split('_')[0]
            age_stage = map_age_stage(int(age))
            new_line = line.replace(age, str(age_stage))
            with open(os.path.join("./data/split-stage", filename), 'a') as f:
                f.write(new_line)
