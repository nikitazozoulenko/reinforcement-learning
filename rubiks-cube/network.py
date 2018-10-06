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


class FCC3x3(nn.Module):
    def __init__(self):
        super(FCC3x3, self).__init__()
        self.fcc = FCC(size=3)


    def forward(self, x):
        return self.fcc(x)
