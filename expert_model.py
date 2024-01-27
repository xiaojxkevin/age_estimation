import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision.models import resnet18
from tensorboardX import SummaryWriter

from dl.options import options_for_stage
from dl.dataset import Utkface_for_stage

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

opts = options_for_stage(stage = 7)
print(opts)

device = torch.device(opts.device)
torch.manual_seed(3407)
torch.backends.cudnn.benchmark = True

model = resnet18(pretrained=True)
num_ftrs = model.fc.in_features  # obtain the number of input features in fc
model.fc = nn.Linear(num_ftrs, 1) # only return a scalar
model = model.to(device)

optimizer = optim.AdamW(model.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=opts.gamma)
loss = nn.L1Loss()

train_data = Utkface_for_stage(opts.data_dir, split="train", stage=7)
val_data = Utkface_for_stage(opts.data_dir, split="val", stage=7)
test_data = Utkface_for_stage(opts.data_dir, split="test", stage=7)
train_loader = DataLoader(train_data, opts.batch_size, shuffle=True, drop_last=True, pin_memory=True)
val_loader = DataLoader(val_data, opts.batch_size, shuffle=False, drop_last=True)
test_loader = DataLoader(test_data, 2, shuffle=False)

train_data_len = len(train_data)
val_data_len = len(val_data)
test_data_len = len(test_data)
print("train dataset length : ", train_data_len)
print("eval dataset length : ", val_data_len)
print("test dataset length :", test_data_len)

current_time = time.strftime("%Y-%m-%dT%H-%M", time.localtime())
writer = SummaryWriter(log_dir=os.path.join(opts.log_path, current_time+"-re"),
                       comment="origin")

best_eval_loss = float("inf")
best_model = ""
for epoch in range(1, opts.epochs + 1):
    model.train()
    train_loss = 0
    for i, inputs in enumerate(train_loader):
        imgs = inputs["image"].to(device)
        ages = inputs["age"].reshape([-1, 1]).to(device).float()
        # print(ages.shape)
        # assert False
        optimizer.zero_grad()
        pred_ages = model(imgs)
        # print(pred_ages.shape)
        # assert False
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
        for i, inputs in enumerate(val_loader):
            imgs = inputs["image"].to(device)
            ages = inputs["age"].reshape([-1, 1]).to(device).float()
            pred_ages = model(imgs)
            batch_loss = torch.mean(torch.abs(pred_ages - ages))
            eval_loss += batch_loss

        eval_loss /= (i + 1)
        #writer.add_scalar("eval loss", eval_loss.item(), epoch)
        writer.add_scalar("eval loss", eval_loss, epoch)
        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            save_dir = os.path.join(opts.ckpt_path, current_time + "-re")
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            best_model = os.path.join(save_dir, "{:03d}.pth".format(epoch))
            torch.save(model, best_model)


with torch.no_grad():
    test_model = torch.load(best_model, map_location=device)
    test_model.eval()
    test_loss = 0
    for i, inputs in enumerate(test_loader):
        imgs = inputs["image"].to(device)
        ages = inputs["age"].reshape([-1, 1]).to(device).float()
        pred_ages = test_model(imgs)
        batch_loss = torch.mean(torch.abs(pred_ages - ages))
        test_loss += batch_loss

    test_loss /= (i + 1)

    print("Test MAE: {} with model {}".format(test_loss, best_model))
    