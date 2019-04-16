# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 16:18:31 2018

@author: Jason Parker
"""

import numpy as np
import scipy.stats as sps

def cross(*args): ##defined the number of arguements going into the cross
    x = args[0]
    if len(args)==2: y = args[1] 
    else: y = x
    return(x.T.dot(y))
def lreg(y,x):
    return(np.linalg.solve(cross(x),cross(x,y)))

class lm: ##linear model
    def __init__(self,y,X):
        self.y = y
        self.X = X
        (self.n,self.r) = X.shape 

        x_0 = np.ones((self.n,1))
        self.X = np.concatenate((x_0,self.X),axis=1)
        self.r += 1

        b = lreg(self.y,self.X)
        b_0 = lreg(self.y,x_0)
        e = y - self.X.dot(b)
        e_0 = y - x_0.dot(b_0)
        self.coef = b
        self.residuals = e

        vb = self.__varfunc()
        Rsq = float(1-cross(e)/cross(e_0))
        self.vb = vb
        self.Rsq = Rsq

        sterr = np.sqrt(np.diag(vb))
        self.sterr = sterr
        (self.tstat,self.pval) = self.coeftest(np.zeros((self.r,1)))

    def __varfunc(self):        # OLS Variance Function
        return float(cross(self.residuals)/self.n)*np.linalg.inv(cross(self.X))
    def coeftest(self,hyp):
        tstat = (self.coef-hyp).T/self.sterr
        pval = sps.norm.cdf(-np.abs(tstat))*2
        return(tstat.T,pval.T)
    def __str__(self):
        return str(self.coef)


np.set_printoptions(suppress=True)
nsim = 10000 ##simulation replications
nvec = (10,25,50,100,250) ##sample size
heterosked = 0

outp1 = np.zeros((len(nvec),4))
outp2 = np.zeros((len(nvec),4))
outp3 = np.zeros((len(nvec),4))
row = 0
for n in nvec:
    bhat = np.zeros((nsim,3)) ##storage containers 3*10,000
    pval = np.zeros((nsim,3))
    np.random.seed(seed=75080)
    for isim in range(nsim):
        #data generating process
        x1 = np.random.normal(size=(n,1))
        x2 = np.random.normal(size=(n,1))
        e = np.random.normal(size=(n,1))
        if heterosked == 1: e = np.multiply(e,np.abs(x1))
        if heterosked == 2: e = np.multiply(e,np.abs(x2))
        y = 1 + 2*x1 + e
        bigX = np.concatenate((x1,x2),axis=1)
        #running model
        model1 = lm(y,bigX)
        bhat[isim,:] = model1.coef.T
        pval[isim,:] = model1.pval.T
    outp1[row,:] = np.hstack(([n],np.mean(bhat,axis=0)))
    outp2[row,:] = np.hstack(([n],np.var(bhat,axis=0)))
    outp3[row,:] = np.hstack(([n],np.mean(pval<=0.05,axis=0)))
    row += 1

print()
print("heterosked is: " + str(heterosked))
print()

print("Average Betas")
print(outp1)
print()

print("Variance of Betas")
print(outp2)
print()

print("Rejection rate from sig test")
print(outp3)
print()
print()
