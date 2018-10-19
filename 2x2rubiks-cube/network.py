import torch
import torch.nn as nn

class FCC(nn.Module):
    def __init__(self, size=3):
        super(FCC, self).__init__()
        layers1 = [nn.Linear(6*size*size*6, 2048),
                  nn.ReLU(inplace = True),
                  nn.BatchNorm1d(2048),
                  nn.Dropout(0.1)]
        layers2 = [nn.Linear(2048, 4096),
                   nn.ReLU(inplace = True),
                   nn.BatchNorm1d(4096),
                   nn.Dropout(0.1)]
        layers3 = [nn.Linear(4096, 4096),
                   nn.ReLU(inplace = True),
                   nn.BatchNorm1d(4096),
                   nn.Dropout(0.1)]
        layers4 = [nn.Linear(4096, 1024),
                   nn.ReLU(inplace = True),
                   nn.BatchNorm1d(1024),
                   nn.Dropout(0.1),
                   nn.Linear(1024, 12)]
        
        self.fcc = nn.Sequential(*layers1, *layers2, *layers3, *layers4)


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
