import argparse

def options():
    parser = argparse.ArgumentParser(description="MWR For UTKFace")
    parser.add_argument("--device", type=str, default='cuda:0', help='compute device')

    parser.add_argument("--data_dir", type=str, default="./data", help="Dataset directory")
    parser.add_argument("--log_path", type=str, default="./log", help="logs output directory")
    parser.add_argument("--ckpt_path", type=str, default="./ckpts", help="checkpoints directory")

    parser.add_argument("--epochs",type=int, default=10, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=10, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay")
    parser.add_argument("--step_size", type=int, default=2, help="learning rate decay after how much step")
    parser.add_argument("--gamma", type=float, default=0.9, help="learning rate decay factor")

    parser.add_argument("--num_class", type=int, default=117, help="# of class to separate")


    args = parser.parse_args()

    return args

