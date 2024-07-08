import torch
import torch.nn as nn
from model.loss import SpecAmpLoss

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x

class UNet1D(nn.Module):
    def __init__(self, in_channels = 1, out_channels = 1):
        super(UNet1D, self).__init__()
        
        self.encoder1 = ConvBlock(in_channels, 32)
        self.encoder2 = ConvBlock(32, 64)
        self.encoder3 = ConvBlock(64, 128)
        self.pool = nn.MaxPool1d(2, 2)
                
        self.upconv2 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)
        self.decoder2 = ConvBlock(128, 64)
        
        self.upconv1 = nn.ConvTranspose1d(64, 32, kernel_size=2, stride=2)
        self.decoder1 = ConvBlock(64, 32)
        
        self.final_conv = nn.Conv1d(32, out_channels, kernel_size=1)

    def forward(self, x, classification = False):
        # Encoder path
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        
        d2 = self.upconv2(e3)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.decoder2(d2)
        
        d1 = self.upconv1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.decoder1(d1)
        
        return self.final_conv(d1)
    def get_loss(self, batch):
        input = batch["BCG"].to("cuda").unsqueeze(1)
        target = batch["RSP"].to("cuda").unsqueeze(1)
        prediction = self(input)
        return nn.MSELoss()(prediction, target).mean()
if __name__ == "__main__":
    model = UNet1D(in_channels=1, out_channels=1)
    x = torch.randn(16, 1, 20000)  
    out = model(x)
    print(out.shape)
