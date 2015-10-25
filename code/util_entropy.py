import numpy as np
import scipy.stats as stats

def entropy(X, axis=0):
    if axis==1:
        Y = [stats.entropy(x) for x in X]
    elif axis==0:
        Y = [stats.entropy(x) for x in X.T]
    else:
        raise ValueError('Error: Choose axis=0 or axis=1')
    return np.array(Y)