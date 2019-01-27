import numpy as np
import matplotlib.pyplot as plt

import random



def gen_data(num=30 , v=4,  a=0):
    """
    Generate a dataset of independent (X) an dependent (Y)
    
    Parameters
    -----------
    num: Integer.  The number of observations
    
    Returns
    --------
    (X,y): a tuple consisting of X and y.  Both are ndarrays
    """
    rng = np.random.RandomState(42)


    # X = num * rng.uniform(size=num)
    X = num * rng.normal(size=num)
    # X = X - X.min()

    X = X.reshape(-1,1)

    e = (v + a*X)
    y = v * X #  +  e * rng.uniform(-1,1, size=(num,1))

    a_term =  0.5 * a * (X**2)
    y = y + a_term

    return X,y

def gen_plot(X,y, xlabel, ylabel):
    fig = plt.figure()
    ax  = fig.add_subplot(1,1,1)


    _ = ax.scatter(X, y, color="red")


    _ = ax.set_xlabel(xlabel)
    _ = ax.set_ylabel(ylabel)
