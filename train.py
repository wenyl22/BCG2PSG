from model.UNET import UNet
import torch
from utils.utils import visualize, visualize_loss, parse_argument
import argparse
def main():
    parser = argparse.ArgumentParser()
    args, dataloader = parse_argument(parser)
    dataloader.setup("fit")
    model = UNet(norm = "in", dropout_rate=0.5).to("cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    train_loss = []
    val_losses = []
    cur_loss = []
    tot_step = 0
    for epoch in range(args.epochs):
        model.train()
        for batch in dataloader.train_dataloader():
            tot_step += 1
            _ = model.get_loss(batch)
            _.backward()
            optimizer.step()
            cur_loss.append(_.item())
            if tot_step % 100 == 0:
                train_loss.append(sum(cur_loss) / len(cur_loss))
                visualize_loss(train_loss, "train", args)
                print("Train Step: ", tot_step, "Loss: ", train_loss[-1])
                cur_loss = []
        model.eval()
        val_loss = []
        with torch.no_grad():
            for i, batch in enumerate(dataloader.val_dataloader()):
                _ = model.get_loss(batch)
                val_loss.append(_.item())
                if i == 0:
                    result = model(batch["BCG"].to("cuda").unsqueeze(1), classification=False)
                    result = result[0].detach().cpu().numpy()
                    visualize(epoch, batch, result, args)
        val_losses.append(sum(val_loss) / len(val_loss))
        if epoch % 10 == 9:
            torch.save(model.state_dict(), f"./checkpoints/{args.name}/{epoch}.pth")
        if val_losses[-1] == min(val_losses):
            torch.save(model.state_dict(), f"./checkpoints/{args.name}/best.pth")
        visualize_loss(val_losses, "val", args)
        print("Validation Loss: ", sum(val_loss) / len(val_loss))

if __name__ == "__main__":
    main()