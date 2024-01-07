import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision.models import resnet18
from tensorboardX import SummaryWriter

from dl.options import options
from dl.dataset import Utkface

opts = options()
device = torch.device(opts.device)
torch.backends.cudnn.benchmark = True

model = resnet18(pretrained=True)
num_ftrs = model.fc.in_features  # obtain the number of input features in fc
model.fc = nn.Linear(num_ftrs, 111)
model = model.to(device)

optimizer = optim.AdamW(model.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=opts.gamma)
loss = nn.CrossEntropyLoss()

train_data = Utkface(opts.data_dir, split="train")
val_data = Utkface(opts.data_dir, split="val")
test_data = Utkface(opts.data_dir, split="test")
train_loader = DataLoader(train_data, opts.batch_size, shuffle=True, drop_last=True, num_workers=8, pin_memory=True)
val_loader = DataLoader(val_data, opts.batch_size, shuffle=False, drop_last=True, num_workers=8)
test_loader = DataLoader(test_data, 2, shuffle=False, num_workers=8)

train_data_len = len(train_data)
val_data_len = len(val_data)
test_data_len = len(test_data)
print("train dataset length : ", train_data_len)
print("eval dataset length : ", val_data_len)
print("test dataset length :", test_data_len)

current_time = time.strftime("%Y-%m-%dT%H:%M", time.localtime())
writer = SummaryWriter(log_dir=os.path.join(opts.log_path, current_time))

best_eval_loss = float("inf")
for epoch in range(1, opts.epochs + 1):
    model.train()
    train_loss = 0
    for i, inputs in enumerate(train_loader):
        imgs = inputs["image"].to(device)
        ages = inputs["age"].to(device).type(torch.long)
        optimizer.zero_grad()
        pred_ages = model(imgs)
        batch_loss = loss(pred_ages, ages)
        batch_loss.backward()
        optimizer.step()
        train_loss += batch_loss.item()

    lr_scheduler.step()
    train_loss /= (i + 1)
    writer.add_scalar("training loss", train_loss, epoch)

    with torch.no_grad():
        model.eval()
        eval_loss = 0
        for i, inputs in enumerate(train_loader):
            imgs = inputs["image"].to(device)
            ages = inputs["age"].to(device).type(torch.float)
            pred_ages = model(imgs)
            batch_loss = torch.mean(torch.abs(pred_ages.argmax(1) - ages))
            eval_loss += batch_loss

        eval_loss /= (i + 1)
        writer.add_scalar("eval loss", eval_loss.item(), epoch)
        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            save_dir = os.path.join(opts.ckpt_path, current_time)
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            torch.save(model, os.path.join(save_dir, "{:03d}.pth".format(epoch)))
