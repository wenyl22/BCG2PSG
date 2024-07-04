from data.dataloader import MyDataModule
from model.resnet import ResNet
import torch
import matplotlib.pyplot as plt
from wuji.Preprocessor.signal import interp_heartbeat
from utils.get_rri import get_rri

def visualize(epoch, batch, output):
    print("Visualizing")
    bcg = batch["input"][0].numpy()
    ecg = batch["target"][0].numpy()
    rec = output.squeeze(1)[0].detach().cpu().numpy()
    plt.figure(figsize=(10, 12))
    plt.subplot(5, 1, 1) 
    plt.plot(bcg, label='BCG')
    plt.legend()
    plt.title('BCG Data')

    plt.subplot(5, 1, 2)  
    plt.plot(ecg, label='ECG')
    ecg_rri = get_rri(ecg)
    plt.legend()
    plt.title('ECG Data')

    plt.subplot(5, 1, 3)  
    plt.plot(ecg_rri, label='ECG_RRI')
    plt.legend()
    plt.title('ECG_RRI Data')

    plt.subplot(5, 1, 4)  
    plt.plot(rec, label='REC')
    rec_rri = get_rri(rec)
    plt.legend()
    plt.title('REC Data')

    plt.subplot(5, 1, 5)  
    plt.plot(rec_rri, label='REC_RRI')
    plt.legend()
    plt.title('REC Data')

    plt.tight_layout()

    plt.savefig(f"./checkpoints_2/{epoch}.png")
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
            "batch_size": 16,
            "num_workers": 4,
            "train": {"shuffle": True},
            "eval": {"shuffle": True},
        },
    )
    dataloader.setup("fit")
    model = ResNet().to("cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss = []
    cur_loss = []
    tot_step = 0
    #wandb.init(project="bcg2psg")
    for epoch in range(50):
        model.eval()
        val_loss = []
        with torch.no_grad():
            for i, batch in enumerate(dataloader.val_dataloader()):
                _ = model.get_loss(batch)
                val_loss.append(_.item())
                if i == 0:
                    visualize(epoch, batch, model(batch["input"].to("cuda").unsqueeze(1)))
        print("Validation Loss: ", sum(val_loss) / len(val_loss))
        model.train()
        for batch in dataloader.train_dataloader():
            #print(batch["input"].shape)
            tot_step += 1
            _ = model.get_loss(batch)
            _.backward()
            optimizer.step()
            cur_loss.append(_.item())
            if tot_step % 100 == 0:
                loss.append(sum(cur_loss) / len(cur_loss))
                print("Train Step: ", tot_step, "Loss: ", loss[-1])
                cur_loss = []

if __name__ == "__main__":
    main()