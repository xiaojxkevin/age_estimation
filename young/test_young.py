import torch
from torch.utils.data import DataLoader

from dl.dataset import Young

@torch.no_grad()
def test_young(path:str, device:str):
    model = torch.load(path, map_location=device)
    ds = Young("./data")
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
    print("Test MAE: {} with model {}".format(total_loss, path))


if __name__ == "__main__":
    device = "cuda:0"
    test_young("ckpts/2024-01-27T11:11/018.pth", device)
    test_young("ckpts/2024-01-27T11:11-weight2.5/009.pth", device)
    test_young("ckpts/2024-01-27T11:47-weight2/011.pth", device)
    test_young("ckpts/2024-01-27T13:28-weight0.75/022.pth", device)

