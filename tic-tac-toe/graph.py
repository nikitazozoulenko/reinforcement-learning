import matplotlib.pyplot as plt

def ewma(losses, alpha = 0.999):
    losses_ewma = []
    ewma = losses[0]
    for loss in losses:
        ewma = alpha*ewma + (1-alpha)*loss
        losses_ewma += [ewma]
    return losses_ewma

def graph(losses):
    plt.figure(1)
    plt.plot(losses, "b", label = "Loss")
    plt.legend(loc=1)
    plt.show()