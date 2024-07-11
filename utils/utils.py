import matplotlib.pyplot as plt
import argparse
import time
import os
from data.dataloader import MyDataModule

def parse_argument(parser):
    parser.add_argument("--name", type=str, default=str(time.time()))
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    if not os.path.exists("./checkpoints/" + args.name):
        os.makedirs("./checkpoints/" + args.name)
    if not os.path.exists("./results/" + args.name):
        os.makedirs("./results/" + args.name)
    dataloader = MyDataModule(
        dataprovider={
            "class": "data.dataloader.MyDataProvider",
            "data_root": "./dataset/bcg2psg",
            "split": [0.90, 0.08, 0.02],
            "preprocess": False,
        },
        dataset={"class": "data.dataloader.MyDataset"},
        dataloader={
            "batch_size": 64,
            "num_workers": 4,
            "train": {"shuffle": True},
            "eval": {"shuffle": True},
        },
    )
    return args, dataloader

def visualize_loss(loss, name, args):
    plt.figure(figsize=(10, 6))
    plt.plot(loss)
    plt.title(name + ' Loss')
    plt.xlabel('Step')
    plt.ylabel(name + ' Loss')
    plt.tight_layout()
    plt.savefig(f"./checkpoints/{args.name}/{name}_loss.png")
    plt.close()

def visualize(epoch, batch, output, args, dir = "checkpoints"):
    bcg = batch["BCG"][0].numpy()
    ecg = batch["ECG"][0].numpy()
    rec = output
    if len(rec.shape) == 2:
        rec = rec[0]
    #print(rec.shape)
    plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 1) 
    plt.plot(bcg, label='BCG')
    plt.legend()
    plt.title('BCG Data')

    plt.subplot(3, 1, 2)  
    plt.plot(ecg, label='ECG')
    plt.legend()
    plt.title('ECG Data')

    plt.subplot(3, 1, 3)
    plt.plot(rec, label='REC')
    plt.legend()
    plt.title('REC Data')

    plt.tight_layout()

    plt.savefig(f"./{dir}/{args.name}/{epoch}.png")
    plt.close()