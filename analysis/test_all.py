import torch
from torch.utils.data import DataLoader

from dl.dataset import Age

@torch.no_grad()
def test(path:str, split_path:str, device:str):
    model = torch.load(path, map_location=device)
    ds = Age("./data", split_path)
    ds_loader = DataLoader(ds, batch_size=1, num_workers=4)
    model.eval()
    total_loss = 0
    for i, inputs in enumerate(ds_loader):
        imgs = inputs["image"].to(device)
        ages = inputs["age"].to(device).float()
        pred_ages = model(imgs)
        loss = torch.mean(torch.abs(pred_ages.argmax(1) - ages))
        total_loss += loss.item()

    total_loss /= (i + 1)
    # return test_loss
    print("Test MAE: {} with model {} in {}".format(total_loss, path, split_path))


if __name__ == "__main__":
    device = "cuda:0"
    path = "ckpts/2024-01-27T16:06/007.pth"
    for c in ["c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8"]:
        test(path, "./data/c/{}.txt".format(c), device)
    