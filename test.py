from model.UNET import UNet
import torch
from utils.utils import visualize, parse_argument
import argparse
import glob
import matplotlib.pyplot as plt
def sort_key(x):
    tmp = x.split("/")[-1].split(".")[0]
    if tmp == "best":
        return float("inf")
    return int(tmp)

def main():
    parser = argparse.ArgumentParser()
    args, dataloader = parse_argument(parser)
    dataloader.setup("fit")
    model = UNet(norm = "in", dropout_rate=0.5).to("cuda")
    model.load_state_dict(torch.load(f"./checkpoints/{args.name}/best.pth"))
    loss = 0
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader.val_dataloader()):
            loss += model.get_loss(batch).item()
        print("Validation Loss: ", loss / len(dataloader.val_dataloader()))    
        state_dicts = glob.glob(f"./checkpoints/{args.name}/*.pth")
        state_dicts = sorted(state_dicts, key = sort_key)
        cnt = len(state_dicts)
        for i, batch in enumerate(dataloader.val_dataloader()):
            if i > 50:
                break
            bcg = batch["BCG"][0].numpy()
            rsp = batch["RSP"][0].numpy()
            plt.figure(figsize=(10, 2 * (cnt + 2)))
            plt.subplot(cnt + 2, 1, 1) 
            plt.plot(bcg, label='BCG')
            plt.legend()
            plt.title('BCG Data')

            plt.subplot(cnt + 2, 1, 2)  
            plt.plot(rsp, label='RSP')
            plt.legend()
            plt.title('RSP Data')
            for j, state_dict in enumerate(state_dicts):
                model.load_state_dict(torch.load(state_dict))
                model.eval()
                result = model(batch["BCG"].to("cuda").unsqueeze(1), classification=False)
               # print(result.std(), result.mean())
                result = result[0].detach().cpu().numpy()
                plt.subplot(cnt + 2, 1, j + 3)  
                plt.plot(result, label = state_dict.split("/")[-1])
                plt.legend()
                plt.title('RSP Data')
            plt.tight_layout()
            plt.savefig(f"./results/{args.name}/{i}.png")
            plt.close()    
if __name__ == "__main__":
    main()