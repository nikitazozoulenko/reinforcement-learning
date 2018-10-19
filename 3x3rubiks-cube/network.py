import torch
import torch.nn as nn

class FCC(nn.Module):
    def __init__(self, size=3):
        super(FCC, self).__init__()
        layers1 = [nn.Linear(6*3*3*6, 2048),
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