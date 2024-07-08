from data.dataloader import MyDataModule
from model.resnet import ResNet
from model.unet import UNet1D
from model.UNET import UNet
from model.linear import Linear
import torch
import matplotlib.pyplot as plt
from utils.get_rri import get_rri
from scipy.fftpack import dct
import numpy as np

def visualize(epoch, batch, output):
    print("Visualizing")
    bcg = batch["BCG"][0].numpy()
    rsp = batch["RSP"][0].numpy()
    rec = output[0].detach().cpu().numpy()
    if len(rec.shape) == 2:
        rec = rec[0]
    print(rec.shape)
    plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 1) 
    plt.plot(bcg, label='BCG')
    plt.legend()
    plt.title('BCG Data')

    plt.subplot(3, 1, 2)  
    plt.plot(rsp, label='RSP')
    plt.legend()
    plt.title('ECG Data')

    plt.subplot(3, 1, 3)  
    plt.plot(rec, label='REC')
    plt.legend()
    plt.title('REC Data')

    plt.tight_layout()

    plt.savefig(f"./checkpoints/{epoch}.png")
    print("Visualized")
    
	
def main():
    dataloader = MyDataModule(
        dataprovider={
            "class": "data.dataloader.MyDataProvider", # "class": MyDataProvider,
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
    dataloader.setup("fit")
    model = UNet().to("cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    loss = []
    cur_loss = []
    tot_step = 0
    for epoch in range(50):
        model.train()
        for batch in dataloader.train_dataloader():
            tot_step += 1
            _ = model.get_loss(batch)
            _.backward()
            optimizer.step()
            cur_loss.append(_.item())
            if tot_step % 100 == 0:
                loss.append(sum(cur_loss) / len(cur_loss))
                print("Train Step: ", tot_step, "Loss: ", loss[-1])
                cur_loss = []
                model.eval()
        val_loss = []
        with torch.no_grad():
            for i, batch in enumerate(dataloader.val_dataloader()):
                _ = model.get_loss(batch)
                val_loss.append(_.item())
                if i == 0:
                    visualize(epoch, batch, model(batch["BCG"].to("cuda").unsqueeze(1), classification=False))
        print("Validation Loss: ", sum(val_loss) / len(val_loss))

if __name__ == "__main__":
    main()