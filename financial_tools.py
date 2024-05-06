import numpy as np

def MaxDrawdown(balance):
    return_list = []
    for i in range(1,len(balance)):
        return_list.append((balance[i] - balance[i-1])/balance[i-1])
    return ((np.maximum.accumulate(return_list) - return_list) / np.maximum.accumulate(return_list)).max()

def SharpeRatio(balance):
    return_list = []
    for i in range(1, len(balance)):
        return_list.append((balance[i] - balance[i - 1]) / balance[i - 1])
    average_return = np.mean(return_list)
    return_stdev = np.std(return_list)
    sharpe_ratio = (average_return-0.02) * np.sqrt(252)  / return_stdev
    return sharpe_ratio
