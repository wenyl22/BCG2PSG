from model.unet import UNet
import torch
from utils.utils import visualize, parse_argument, dct_batch
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import zscore
def sort_key(x):
    tmp = x.split("/")[-1].split(".")[0]
    if tmp == "best":
        return float("inf")
    return int(tmp)

def main():
    parser = argparse.ArgumentParser()
    args, dataloader = parse_argument(parser)
    if not os.path.exists(f"./analyze/{args.name}"):
        os.makedirs(f"./analyze/{args.name}")
    dataloader.setup("fit")
    model = UNet(norm = "in", dropout_rate=0.5).to("cuda")
    model.load_state_dict(torch.load(f"./checkpoints/{args.name}/best.pth"))
    for p in model.encoder.module_list[0].parameters():
        # add the magnitude of weights in the first convolutional layer
        if(len(p.shape) != 3 or p.shape[1] != 12):
            continue
        for i in range(p.shape[1]):
            print(torch.norm(p[:, i, :], p = 1).item())
    # model.eval()
    # with torch.no_grad():
    #     for i, batch in enumerate(dataloader.val_dataloader()):
    #         result = model(batch["BCG"].to("cuda").unsqueeze(1), classification=False)
    #         for j in range(result.shape[0]):
    #             print(j)
    #             bcg = batch["BCG"][j].numpy()
    #             rsp = batch["RSP"][j].numpy()
    #             rec = result[j].detach().cpu().numpy()
    #             rec = zscore(rec)
    #             dct_bcg = np.abs(np.fft.fft(bcg))[0:1500] / 6000
    #             dct_rsp = np.abs(np.fft.fft(rsp))[0:3000] / 6000
    #             dct_rec = np.abs(np.fft.fft(rec))[0:3000] / 6000
    #             frequency = np.arange(3000) / 60

    #             plt.figure(figsize=(10, 12))
    #             plt.subplot(5, 1, 1) 
    #             plt.plot(bcg, label='BCG')
    #             plt.legend()
    #             plt.title('BCG Data')

    #             plt.subplot(5, 1, 2)  
    #             plt.plot(rsp, label='GTH')
    #             plt.plot(rec, label='REC')
    #             plt.legend()
    #             plt.title('RSP Data')

    #             plt.subplot(5, 1, 3)  
    #             plt.plot(rec, label='REC')
    #             plt.legend()
    #             plt.title('RSP Data')

    #             plt.subplot(5, 1, 4)
    #             plt.plot(frequency[0:50], dct_rsp[0:50], label='LOW DCT GTH')
    #             plt.plot(frequency[0:50], dct_rec[0:50], label='LOW DCT REC')
    #             #plt.plot(frequency[0:50], dct_bcg[0:50], label='LOW DCT BCG')
    #             plt.legend()
    #             plt.title('LOW DCT RSP Data')

    #             plt.subplot(5, 1, 5)
    #             plt.plot(frequency, dct_rsp, label='DCT GTH')
    #             plt.plot(frequency, dct_rec, label='DCT REC')
    #             #plt.plot(frequency[0:50], dct_bcg[0:50], label='LOW DCT BCG')
    #             plt.legend()
    #             plt.title('Full DCT RSP Data')

    #             plt.tight_layout()
    #             print(f"./analyze/{args.name}/{j}.png")
    #             plt.savefig(f"./analyze/{args.name}/{j}.png")
    #             plt.close()
    #         break
if __name__ == "__main__":
    main()