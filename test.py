from model.roformer import Roformer4SignalFiltering
from model.unet import UNet
from model.fredformer import Fredformer4SignalFiltering
import multiprocessing
import torch
from utils.utils import visualize, parse_argument, visualize_coeffs
import argparse
import glob
import matplotlib.pyplot as plt

def write_stats(args, flow_loss, mse_loss):
    with open(f"./results/{args.name}/stats.txt", "w") as f:
        f.write(f"Flow loss: {sum(flow_loss) / len(flow_loss)}\n")
        f.write(f"MSE loss: {sum(mse_loss) / len(mse_loss)}\n")
    # plot the histogram of the loss
    plt.hist(flow_loss, bins = 100)
    plt.xlabel("Flow loss")
    plt.ylabel("Frequency")
    plt.title("Histogram of flow loss")
    plt.savefig(f"./results/{args.name}/flow_loss.png")
    plt.close()
    plt.hist(mse_loss, bins = 100)
    plt.xlabel("MSE loss")
    plt.ylabel("Frequency")
    plt.title("Histogram of mse loss")
    plt.savefig(f"./results/{args.name}/mse_loss.png")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    args, dataloader, kwargs = parse_argument(parser)
    dataloader.setup("fit")
    if args.model == "roformer" or args.model == "dit":
        model = Roformer4SignalFiltering(**kwargs).to(args.device)
    elif args.model == "unet":
        model = UNet(**kwargs).to(args.device)
    elif args.model == "fredformer":
        model = Fredformer4SignalFiltering(**kwargs).to(args.device)

    model.load_state_dict(torch.load(f"./checkpoints/{args.name}/best.pth", weights_only=True))
    model.eval()
    with torch.no_grad():
        mse_loss = []
        flow_loss = []
        for i, batch in enumerate(dataloader.val_dataloader()):
            if i < 50:
                if args.model == "fredformer":
                    result_real, result_imag, result = model.get_real_and_imag(**batch)
                    result = result[0].detach().cpu().numpy()
                    result_real = result_real[0].detach().cpu().numpy()
                    result_imag = result_imag[0].detach().cpu().numpy()
                    visualize_coeffs(i, batch, result, result_real, result_imag, args, dir = "results")
                else:
                    result = model.get_rec(**batch)[0].detach().cpu().numpy()
                    visualize(i, batch, result, args, dir = "results")
            flow_loss.append(model.get_loss(**batch).item())
            mse_loss.append(model.get_mse_loss(**batch).item())
            print(i, sum(flow_loss) / len(flow_loss), sum(mse_loss) / len(mse_loss))
    print("Flow loss:", sum(flow_loss) / len(flow_loss))
    print("MSE loss:", sum(mse_loss) / len(mse_loss))
    write_stats(args, flow_loss, mse_loss)

if __name__ == "__main__":
    main()