'''

'''

import csv
import numpy as np
import scipy.stats as sps
from scipy.stats import norm
import pandas as pd


def cross(*args):  # Can put either 1 or 2 things into this function
    x = args[0]
    if len(args) == 1:
        y = x
    else:
        y = args[1]
    return (x.T.dot(y))


ls = lambda X, y: np.linalg.solve(cross(X), cross(X, y))


class ols:
    def __init__(self, X, y):
        self.n = X.shape[0]
        self.X = np.hstack((np.ones((self.n, 1)), X))
        self.y = y
        self.b = ls(self.X, self.y)
        self.pred = self.X.dot(self.b)
        self.res = self.y - self.pred
        RSS = cross(self.res)[0, 0]
        ybar = self.y.mean()
        e_0 = self.y - ybar
        TSS = cross(e_0)[0, 0]
        self.rsq = 1 - RSS / TSS
        sig = np.sqrt(RSS / self.n)
        self.vcov = np.linalg.inv(cross(self.X)) * (sig ** 2)

        # Question 2: Calculation of log likelihood
        self.logLikelihood = -self.n/2 * np.log(2*np.pi*sig**2) - self.n/(2*sig**2) * self.res.T.dot(self.res)/self.n


        print("log likelihood is", self.logLikelihood)

    def predict(self, newX):
        newn = newX.shape[0]
        newX = np.hstack((np.ones((newn, 1)), newX))
        return newX.dot(self.b)

    def tidy(self):
        se = np.sqrt(np.diag(self.vcov)).reshape(-1, 1)
        tstats = self.b / se
        pval = norm.cdf(-np.abs(tstats)) * 2
        outp = np.hstack((self.b, se, tstats, pval))
        outp = pd.DataFrame(outp)
        outp.columns = ['beta', 'stderr', 'tstat', 'pval']
        print(outp)
        print("RSq=" + str(self.rsq))
