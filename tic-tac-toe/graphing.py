import matplotlib.pyplot as plt

def values2ewma(losses, alpha = 0.999):
    losses_ewma = []
    ewma = -1
    for loss in losses:
        ewma = alpha*ewma + (1-alpha)*loss
        losses_ewma += [ewma]
    return losses_ewma

def graph(returns):
    plt.figure(1)
    plt.plot(returns, "b", label = "Total Episode Return")
    plt.legend(loc=1)
    plt.show()