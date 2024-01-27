import os
import numpy as np 

# split_path = "data/split"

val_split = "./data/split/val.txt"
test_split = "./data/split/test.txt"

ages = {"c1": [],
        "c2": [],
        "c3": [],
        "c4": [],
        "c5": [],
        "c6": [],
        "c7": [],
        "c8": []}

val_file = np.genfromtxt(val_split, dtype=str)
test_file = np.genfromtxt(test_split, dtype=str)
files = np.concatenate([val_file, test_file], axis=0)
# print(files.shape)
# assert False

for f in files:
    w = f.split("_")
    age = int(w[0])
    if age <= 3:
        ages["c1"].append(f)
    elif age <= 6 and age >= 4:
        ages["c2"].append(f)
    elif age <= 12 and age >= 7:
        ages["c3"].append(f)
    elif age <= 21 and age >= 13:
        ages["c4"].append(f)
    elif age <= 37 and age >= 22:
        ages["c5"].append(f)
    elif age <= 65 and age >= 38:
        ages["c6"].append(f)
    elif age <= 84 and age >= 66:
        ages["c7"].append(f)
    else:
        ages["c8"].append(f)
    

for c in ages.keys():
    with open("./data/c/{}.txt".format(c), "w+") as f:
        for line in ages[c]:
            f.write(line + "\n")

        f.close()