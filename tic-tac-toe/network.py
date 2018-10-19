import torch
import torch.nn as nn

# class FCC(nn.Module):
#     def __init__(self, size=3, channels=128, n_layers=4):
#         super(FCC, self).__init__()
#         layers = [nn.Linear(size*size, channels),
#                   nn.ReLU(inplace = True),
#                   nn.BatchNorm1d(channels),
#                   nn.Dropout(0.05)]
#         for _ in range(n_layers):
#             layers += [nn.Linear(channels, channels),
#                        nn.ReLU(inplace = True),
#                        nn.BatchNorm1d(channels),
#                        nn.Dropout(0.05)]
#         self.fcc = nn.Sequential(*layers)

#         probs = [nn.Linear(channels, size*size), 
#                  nn.Softmax(dim=1)]
#         self.probs = nn.Sequential(*probs)

#         v = [nn.Linear(channels, 1), 
#              nn.Tanh()]
#         self.v = nn.Sequential(*v)


#     def forward(self, x):
#         layers = self.fcc(x)
#         probs = self.probs(layers)
#         v = self.v(layers)
#         return probs, v



class CNN(nn.Module):
    def __init__(self, size, channels=128, n_layers=4):
        super(CNN, self).__init__()
        self.size = size
        self.channels = channels
        conv = [nn.Conv2d(1, channels, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace = True)]
        for _ in range(n_layers):
            conv += [ResidualBlock(channels//2, expansion=2, cardinality=1)]
        self.conv = nn.Sequential(*conv)
        self.probs = nn.Sequential(ResidualBlock(channels//2, expansion=2, cardinality=1),
                                   nn.Conv2d(channels, 1, kernel_size=1))
        self.softmax = nn.Softmax(dim=1)
        self.v = ResidualBlock(channels//2, expansion=2, cardinality=1)
        self.linear_and_tanh = nn.Sequential(nn.Linear(size*size*channels, 1), nn.Tanh())


    def forward(self, x):
        conv = self.conv(x.view(-1, 1, self.size, self.size))

        probs = self.probs(conv)
        probs = probs.view(-1, self.size*self.size)
        probs = self.softmax(probs)

        v = self.v(conv)
        v = v.view(-1, self.size*self.size*self.channels)
        v = self.linear_and_tanh(v)
        return probs, v


class ResidualBlock(nn.Module):
    def __init__(self, channels, expansion = 2, cardinality = 1):
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