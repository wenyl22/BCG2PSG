import torch.nn as nn
import torch.nn.init as init
class Linear(nn.Module):
    def __init__(self, inc = 200):
        super(Linear, self).__init__()
        self.inc = inc
        self.linear1 = nn.Linear(inc, inc)
        self.bn = nn.BatchNorm1d(inc)
    
    def forward(self, x):
        out = self.bn(self.linear1(x))
        return out
    def get_loss(self, batch):
        input = batch["input_dct"][..., 0:self.inc].to("cuda")
        target = batch["target_dct"][..., 0:self.inc].to("cuda")
      #  print(input.shape)
        prediction = self(input)
        return nn.MSELoss()(prediction, target).mean()
