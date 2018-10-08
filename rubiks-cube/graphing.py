import matplotlib.pyplot as plt
import os

class Grapher:
    def __init__(self, filename):
        self.filename = filename


    def write(self, value):
        with open(self.filename, "a") as f:
            f.write(value+",")

    
    def read(self, alpha):
        '''Reads self.filename and returns EWMA:d list of losses'''
        with open(self.filename, "r") as f:
            lines = f.read().split(",")[:-1]
            lines = [float(x) for x in lines]
        return lines

    
    def values2ewma(self, losses, alpha = 0.999):
        losses_ewma = []
        if losses:
            ewma = losses[0]
        for loss in losses:
            ewma = alpha*ewma + (1-alpha)*loss
            losses_ewma += [ewma]
        return losses_ewma


def graph_all(path="save_dir/loss_folder/", alpha=0):
    list_dir = os.listdir(path)
    list_losses = [Grapher(path+filename).read(alpha) for filename in list_dir]
    for i, losses in enumerate(list_losses):
        plt.figure(i+1)
        plt.plot(losses, "b", label = list_dir[i])
        plt.legend(loc=1)
    plt.show()


if __name__ == "__main__":
    graph_all()