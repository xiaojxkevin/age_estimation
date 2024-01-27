import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision.models import resnet18
from dl.options import *
from dl.dataset import Utkface

opts = options()
device = torch.device(opts.device)
torch.manual_seed(3407)
torch.backends.cudnn.benchmark = True

test_data = Utkface(opts.data_dir, split="test")
test_loader = DataLoader(test_data, 2, shuffle=False)
print("test dataset length :",len(test_data))

with torch.no_grad():

    overall_model = torch.load(r"./ckpts\overall\2024-01-27T16-59\021.pth", map_location=device)
    model0 = torch.load(r"./ckpts\stage0\2024-01-27T22-20-re\010.pth", map_location=device)
    model1 = torch.load(r"./ckpts\stage1\2024-01-27T22-26-re\020.pth", map_location=device)
    model2 = torch.load(r"./ckpts\stage2\2024-01-27T22-30-re\019.pth", map_location=device)
    model3 = torch.load(r"./ckpts\stage3\2024-01-27T22-36-re\008.pth", map_location=device)
    model4 = torch.load(r"./ckpts\stage4\2024-01-27T22-43-re\006.pth", map_location=device)
    model5 = torch.load(r"./ckpts\stage5\2024-01-27T23-08-re\015.pth", map_location=device)
    model6 = torch.load(r"./ckpts\stage6\2024-01-27T23-21-re\022.pth", map_location=device)
    model7 = torch.load(r"./ckpts\stage7\2024-01-27T23-27-re\014.pth", map_location=device)
    
    overall_model.eval()
    model0.eval()
    model1.eval()
    model2.eval()
    model3.eval()
    model4.eval()
    model5.eval()
    model6.eval()
    model7.eval()

    model_map = {0:model0, 1:model1, 2:model2, 3:model3, 4:model4, 5:model5, 6:model6, 7:model7}

    test_loss = 0
    for i, inputs in enumerate(test_loader):
        imgs = inputs["image"].to(device)
        ages = inputs["age"].reshape([-1, 1]).to(device).float()
        age_stage = overall_model(imgs)
        age_stage = age_stage.argmax(dim=1)
        batch_loss = 0
        for j in range(len(age_stage)):
            pred_ages = model_map[int(age_stage[j])](imgs[j].reshape([1, 3, 200, 200]))
            batch_loss += torch.abs(pred_ages - ages[j])
        batch_loss /= len(age_stage)
        test_loss += batch_loss.item()

    test_loss /= (i + 1)
    print("Test MAE: {} with model".format(test_loss))
