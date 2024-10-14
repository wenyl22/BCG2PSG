import matplotlib.pyplot as plt
import torch
import time
import os
from data.dataloader import MyDataModule
import getpass
import json
def parse_argument(parser):
    parser.add_argument("--name", type=str, default=str(time.time()))
    args, unknown = parser.parse_known_args()
    config_file = f"./configs/{args.name}.json"
    assert os.path.exists(config_file), f"Config file {config_file} does not exist"
    with open(config_file, "r") as f:
        config_args = json.load(f)
        for key, value in config_args.items():
            parser.add_argument(f"--{key}", type=type(value), default=value)
    args = parser.parse_args()

    checkpoint_dir = f"./checkpoints/{args.name}"
    result_dir = f"./results/{args.name}"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        current_user = getpass.getuser()
        os.system(f"chown -R {current_user}:{current_user} {checkpoint_dir}")
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        current_user = getpass.getuser()
        os.system(f"chown -R {current_user}:{current_user} {result_dir}")
    dataloader = MyDataModule(
        dataprovider={
            "class": "data.dataloader.MyDataProvider",
            "data_root": "/local_data/datasets/exported_parallel_data_v2/",
            "save_root": args.save_root,
            "split": [0.90, 0.08, 0.02],
            "preprocess": False,
            # "amount": args.amount,
        },
        dataset={"class": "data.dataloader.MyDataset"},
        dataloader={
            "batch_size": args.batch_size,
            "num_workers": 4,
            "train": {"shuffle": True},
            "eval": {"shuffle": False},
        },
    )
    return args, dataloader, args.__dict__

def visualize_loss(loss, name, args):
    plt.figure(figsize=(10, 6))
    plt.plot(loss)
    plt.title(name + ' Loss')
    plt.xlabel('Step')
    plt.ylabel(name + ' Loss')
    plt.tight_layout()
    plt.savefig(f"./checkpoints/{args.name}/{name}_loss.png")
    plt.close()

def visualize(epoch, batch, rec, args, dir = "checkpoints"):
    input = batch["input"][0].numpy()
    # if input has several channels, only visualize the first channel
    if len(input.shape) > 1:
        input = input[0]
    tar = batch["tar"][0].numpy()

    plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 1) 
    plt.plot(input, label='input')
    plt.legend()
    plt.title('input Data')

    plt.subplot(3, 1, 2)  
    plt.plot(tar, label='tar')
    plt.plot(rec, label='rec')
    plt.legend()
    plt.title('tar Data')

    plt.subplot(3, 1, 3)
    plt.plot(rec, label='rec')
    plt.legend()
    plt.title('rec Data')

    plt.tight_layout()

    plt.savefig(f"./{dir}/{args.name}/{epoch}.png")
    plt.close()

def visualize_coeffs(epoch, batch, rec, rec_real, rec_imag, args, dir = "checkpoints"):
    input = batch["input"][0].numpy()
    if len(input.shape) > 1:
        input = input[0]
    tar = batch["tar"][0].numpy()
    tar_real = batch["tar_dec"][0].numpy().real
    tar_imag = batch["tar_dec"][0].numpy().imag

    plt.figure(figsize=(10, 10))
    plt.subplot(5, 1, 1) 
    plt.plot(input, label='input')
    plt.legend()
    plt.title('input Data')

    plt.subplot(5, 1, 2)
    plt.plot(tar_real[0:180], label='tar_real')
    plt.plot(tar_imag[0:180], label='tar_imag')
    plt.legend()
    plt.title('tar Data')

    plt.subplot(5, 1, 3)
    plt.plot(rec_real[0:180], label='rec_real')
    plt.plot(rec_imag[0:180], label='rec_imag')
    plt.legend()
    plt.title('rec Data')

    plt.subplot(5, 1, 4)  
    plt.plot(tar, label='tar')
    plt.plot(rec, label='rec')
    plt.legend()
    plt.title('tar Data')

    plt.subplot(5, 1, 5)
    plt.plot(rec, label='rec')
    plt.legend()
    plt.title('rec Data')

    plt.tight_layout()
    plt.savefig(f"./{dir}/{args.name}/{epoch}.png")
    plt.close()
