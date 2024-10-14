from model.unet import UNet
from model.roformer import Roformer4SignalFiltering
from model.fredformer import Fredformer4SignalFiltering
import torch
from utils.utils import visualize, visualize_loss, parse_argument, visualize_coeffs
import argparse
from tqdm import tqdm
import numpy as np
from torch.optim import lr_scheduler

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

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=3e-3)
    train_loss = []
    val_losses = []
    cur_loss = []
    tot_step = 0

    for epoch in range(args.epochs):
        model.train()
        with tqdm(dataloader.train_dataloader(), desc=f"Train Epoch {epoch}", unit="batch") as pbar:
            for i, batch in enumerate(pbar):
                tot_step += 1
                optimizer.zero_grad()
                _ = model.get_loss(**batch)
                _.backward()
                optimizer.step()
                cur_loss.append(_.item())
                if tot_step % 100 == 0:
                    train_loss.append(sum(cur_loss) / len(cur_loss))
                    visualize_loss(train_loss, "train", args)
                    cur_loss = []
                pbar.set_description(f"Train Loss: {train_loss[-1] if len(train_loss) > 0 else 0}")
        model.eval()
        val_loss = []
        with torch.no_grad():
            batch_id = 0
            with tqdm(dataloader.val_dataloader(), desc=f"Val Epoch {epoch}", unit="batch") as pbar:
                for i, batch in enumerate(pbar):
                    _ = model.get_loss(**batch)
                    val_loss.append(_.item())
                    if i == batch_id:
                        if args.model == "fredformer":
                            result_real, result_imag, result = model.get_real_and_imag(**batch)
                            result = result[0].cpu().detach().numpy()
                            result_real = result_real[0].cpu().detach().numpy()
                            result_imag = result_imag[0].cpu().detach().numpy()
                            visualize_coeffs(epoch, batch, result, result_real, result_imag, args)
                        else:
                            result = model.get_rec(**batch)[0].cpu().detach().numpy()
                            visualize(epoch, batch, result, args)
                    pbar.set_description(f"Val Loss: {sum(val_loss) / len(val_loss)}")
            val_losses.append(sum(val_loss) / len(val_loss))
        # if epoch % 4 == 3:
        #     torch.save(model.state_dict(), f"./checkpoints/{args.name}/{epoch}.pth")
        if val_losses[-1] == min(val_losses):
            torch.save(model.state_dict(), f"./checkpoints/{args.name}/best.pth")
        visualize_loss(val_losses, "val", args)
        print("Validation Loss: ", sum(val_loss) / len(val_loss))

if __name__ == "__main__":
    main()