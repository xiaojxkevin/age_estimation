import os
import numpy as np 

# split_path = "data/split"

val_split = "./data/split/val.txt"
test_split = "./data/split/test.txt"

young_list = []

val_file = np.genfromtxt(val_split, dtype=str)
test_file = np.genfromtxt(test_split, dtype=str)
files = np.concatenate([val_file, test_file], axis=0)
# print(files.shape)
# assert False

for f in files:
    w = f.split("_")
    age = int(w[0])
    if age <= 18:
        young_list.append(f)

with open("./data/young/files.txt", "w+") as f:
    for line in young_list:
        f.write(line + "\n")

    f.close()