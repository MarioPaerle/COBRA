import numpy as np
from timeit import default_timer as timer
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


class Chronometer:
    """an easy chronometer implementation to make it easier to tic toc in python"""
    def __init__(self):
        self.passed = 0
        self.start = 0

    def tic(self):
        self.start = timer()

    def toc(self):
        return timer() - self.start



def fplot(*args, color=None):
    """Makes it easier to plot faster"""

    import matplotlib.pyplot as plt

    for i, arg in enumerate(args):
        plt.plot(arg, color=color if i == 0 else None)

    plt.show()

def distance(x, y, axis=None):
    return np.linalg.norm(x - y, axis=axis)

def distance2(x, y):
    return np.sum(np.sqrt(np.pow(x - y, 2)))

def chronoscore(function, args=None, kwargs={}, tries=100):
    c = Chronometer()
    c.tic()
    times = []
    for i in range(tries):
        function(*args, **kwargs)
        times.append(c.toc())

    print("Elapsed time mean (seconds):", np.mean(times))
    return times

def diff(X):
    return pd.Series(X).diff().fillna(0).values

def rolling(X, n, axis=None):
    return sliding_window_view(X, window_shape=n, axis=axis)

def printinput(*args, sep=' ', end='\n', **kwargs):
    print(*args, sep=sep, end=end, **kwargs)
    input()

def get_market(stock='EURUSD=X', period='2y', interval='1d'):
    import yfinance as yf
    return yf.download(stock, period=period, interval=interval)



