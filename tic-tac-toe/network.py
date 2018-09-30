import torch
import torch.nn as nn

class FCC(nn.Module):
    def __init__(self, size=3, channels=128, n_layers=5):
        super(FCC, self).__init__()
        layers = [nn.Linear(size*size, channels),
                  nn.ReLU(inplace = True),
                  nn.BatchNorm1d(channels),
                  nn.Dropout(0.05)]
        for _ in range(n_layers):
            layers += [nn.Linear(channels, channels),
                       nn.ReLU(inplace = True),
                       nn.BatchNorm1d(channels),
                       nn.Dropout(0.05)]
        layers += [nn.Linear(channels, size*size)]
        layers += [nn.Tanh()]
        self.fcc = nn.Sequential(*layers)


    def forward(self, x):
        return self.fcc(x)


class CNN(nn.Module):
    def __init__(self, size=5, channels=64, n_layers=5):
        super(CNN, self).__init__()
        self.size = size
        self.channels = channels
        conv = [nn.Conv2d(1, channels, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace = True)]
        for _ in range(n_layers):
            conv += [ResidualBlock(channels//2, expansion=2, cardinality=1)]
        self.conv = nn.Sequential(*conv)
        self.linear = nn.Sequential(nn.Linear(size*size*channels, size*size),
                                    nn.Tanh())


    def forward(self, x):
        x = self.conv(x.view(-1, self.size, self.size, 1))
        x = self.linear(x.view(-1, self.size*self.size*self.channels))
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