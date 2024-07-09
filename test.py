from model.UNET import UNet
import torch
from utils.utils import visualize, parse_argument
import argparse
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
            if i < 50:
                result = model(batch["BCG"].to("cuda").unsqueeze(1), classification=False)
                result = result[0].detach().cpu().numpy()
                visualize(i, batch, result, args, dir = "results")
    print("Validation Loss: ", loss / len(dataloader.val_dataloader()))
if __name__ == "__main__":
    main()