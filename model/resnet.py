import torch.nn as nn
import torch.nn.init as init

class BasicBlock(nn.Module):
    def __init__(self, in_c, out_c, stride):
        super(BasicBlock, self).__init__()
        # nn.Conv2d(in_c, out_c, kernel, stride, padding, bias)
        ker, pad = 3, 1
        self.conv1 = nn.Conv1d(in_c, out_c, ker, stride, pad, bias=False)
        
        self.bn1 = nn.BatchNorm1d(out_c)
        self.conv2 = nn.Conv1d(out_c, out_c, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_c)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_c, out_c, 1, stride, bias=False),
                nn.BatchNorm1d(out_c)
            )
        init.kaiming_normal_(self.conv1.weight)
        init.kaiming_normal_(self.conv2.weight)
    def forward(self, x):
        out = nn.GELU()(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = nn.GELU()(out)
        return out


class ResNet(nn.Module):
    def __init__(self, nf = 32):
        super(ResNet, self).__init__()
        self.layers = [nn.Conv1d(1, nf, 7, 1, 3, bias=False), nn.BatchNorm1d(nf), nn.ReLU(True)]
        n_downsample = 2
        for i in range(n_downsample):
            mult = nf * (2 ** i)
            self.layers += [nn.Conv1d(mult, mult * 2, 3, 2, 1, bias=False)]
            self.layers += [nn.BatchNorm1d(mult * 2)]
            self.layers += [nn.GELU()]
        n_blocks = 6
        nf_ = nf * (2 ** n_downsample)
        for i in range(n_blocks):
            self.layers.append(BasicBlock(nf_, nf_, stride=1))
        for i in range(n_downsample):
            mult = nf * (2 ** (n_downsample - i - 1))
            self.layers += [nn.ConvTranspose1d(mult * 2, mult, 3, 2, 1, 1, bias=False)]
            self.layers += [nn.BatchNorm1d(mult)]
            self.layers += [nn.GELU()]
        self.layers += [nn.Conv1d(nf, 1, 7, 1, 3, bias=False)]
        self.model = nn.Sequential(*self.layers)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
        self.metrics = SpecAmpLoss(20000, 20000, power=2)
    def forward(self, x):
        return self.model(x)
    def get_loss(self, batch):
        input = batch["input"].to("cuda").unsqueeze(1)
        target = batch["target"].to("cuda").unsqueeze(1)
        prediction = self(input)
        return self.metrics(prediction, target)
