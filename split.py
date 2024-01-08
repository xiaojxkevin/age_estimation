import numpy as np
import os

p = [0.8, 0.1, 0.1] # train, val, test

data_dir = "./data/utkface_aligned_cropped/UTKFace"

files = os.listdir(data_dir)
np.random.shuffle(files)

total_length = len(files)
num_train, num_val = int(p[0] * total_length), int(p[1] * total_length)
num_test = total_length - num_train - num_val

print("The split for train, val, test are", num_train, num_val, num_test)

names = {"train": files[:num_train], 
         "val": files[num_train:num_train+num_val], 
         "test": files[num_train+num_val:]}


for t in ["train", "val", "test"]:
    with open("./data/split/{}.txt".format(t), "w+") as f:
        for line in names[t]:
            f.write(line + "\n")

        f.close()