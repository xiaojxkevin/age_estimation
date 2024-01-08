import os
import numpy as np
import PIL.Image as pil
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class Utkface(Dataset):
    def __init__(self, data_dir:str, split:str) -> None:
        super().__init__()
        self.split = split
        assert split in ("train", "val", "test")
        print("----------Creating {} set----------".format(self.split))

        self.data_path = os.path.join(data_dir, "utkface_aligned_cropped/UTKFace")
        self.split_path = os.path.join(data_dir, "split", split+".txt")
        self.image_names = np.genfromtxt(self.split_path, dtype=str).tolist()
        self.trans = transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self) -> int:
        return len(self.image_names)
    
    def __getitem__(self, index):
        image_name = self.image_names[index]
        image = self.trans(pil.open(os.path.join(self.data_path, image_name)))
        image_name = image_name.split("_")
        age = int(image_name[0])
        gender = int(image_name[1])
        race = int(image_name[2])
        
        return {
            "age": age,
            "gender": gender,
            "race": race,
            "image": image
        }
    