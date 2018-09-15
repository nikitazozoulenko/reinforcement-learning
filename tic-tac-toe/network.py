import torch
import torch.nn as nn

class FCC(nn.Module):
    def __init__(self, size=3):
        super(FCC, self).__init__()

        layers = [nn.Linear(6*size*size*6, 512),
                  nn.ReLU(inplace = True),
                  nn.BatchNorm1d(512),
                  nn.Dropout(0.1)]
        
        for _ in range(2):
            layers += [nn.Linear(512, 512),
                       nn.ReLU(inplace = True),
                       nn.BatchNorm1d(512),
                       nn.Dropout(0.1)]

        layers += [nn.Linear(512, 12)]
        
        self.fcc = nn.Sequential(*layers)


    def forward(self, x):
        return self.fcc(x)


class FCC2x2(nn.Module):
    def __init__(self):
        super(FCC2x2, self).__init__()
        self.fcc = FCC(size=2)


    def forward(self, x):
        return self.fcc(x)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        conv = [nn.BatchNorm2d(36),
                  nn.Conv2d(36, 64, kernel_size=3, stride=1, padding=1),
                  nn.ReLU(inplace = True)]
        for _ in range(3):
            conv += [ResidualBlock(32, expansion=2, cardinality=1)]
        self.conv = nn.Sequential(*conv)

        self.linear = nn.Linear(32*2*3*3, 12)

    def forward(self, x):
        x = self.conv(x)
        x = self.linear(x.view(-1, 32*2*3*3))
        return x


class ResidualBlock(nn.Module):
    def __init__(self, channels, expansion = 4, cardinality = 1):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(nn.Conv2d(channels*expansion, channels, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(channels),
                                   nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups = cardinality, bias=False),
                                   nn.BatchNorm2d(channels),
                                   nn.Conv2d(channels, channels*expansion, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(channels*expansion))
        
        self.relu = nn.ReLU(inplace = True)
        
        
    def forward(self, x):
        res = x

        out = self.block(x)
        out = self.relu(out+res)
        
        return out