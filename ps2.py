# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 18:25:55 2018

@author: Yudong Shi / yxs175130
"""

## Problem Set II
## Question 3:

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# When Yt = Yt-1 + Et, Xt = Xt-1 + Ut:
sig = 0.5
nsim = 1000
Tvec = [10,50,100,200,300]
nvec = [1,2,5,10]
holding_place_1=np.zeros((nsim,len(Tvec),len(nvec)))
for ndx,n in enumerate(nvec):
    for tdx, T in enumerate(Tvec):
        for isim in range(nsim):
            y=np.random.normal(0,sig,size=(T,1))
            x=np.random.normal(0,sig,size=(T,n))
            for t in range(1,T):
                y[t] +=y[t-1]
                x[t] +=x[t-1]
                model_1=smf.OLS(y,x).fit()
                holding_place_1[isim,tdx,ndx]=model_1.rsquared
print(holding_place_1.mean(0))

# When Yt = Et, Xt = Ut:
holding_place_2 = np.zeros((nsim,len(Tvec),len(nvec)))
for ndx,n in enumerate(nvec):
    for tdx,T in enumerate(Tvec):
        for isim in range(nsim):
            y = np.random.normal(0,sig,size=(T,1))
            x = np.random.normal(0,sig,size=(T,n))
            x = sm.add_constant(x)
            model_2 = smf.OLS(y,x).fit()
            holding_place_2[isim,tdx,ndx] = model_2.rsquared
print(holding_place_2.mean(0))

## Conclusion: 
# The R Square will increase when the independent variables increases.  

## Question 4

ts=[10,50,100,200,300,400]
sig=0.5
rho=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
nsim=1000

par1=np.zeros((nsim,len(ts),len(rho)))
par2=np.zeros((nsim,len(ts),len(rho)))
bias1=np.zeros((nsim,len(ts),len(rho)))
bias2=np.zeros((nsim,len(ts),len(rho)))
for rdx,r in enumerate(rho):
    for tdx, T in enumerate(ts):
        for isim in range(nsim):
            y = np.random.normal(0,sig,size=(T+100,1))
            for t in range(1,T+100):
                y[t] += rho[rdx]*y[t-1]
            y = y[100:]
            y_0 = y[2:T]
            y_1 = y[1:T-1]
            xdata = sm.add_constant(y_1)
            z=smf.OLS(y_0,y_1).fit().params
            par1[isim,tdx,rdx]=z
            bias1[isim,tdx,rdx]=par1[isim,tdx,rdx]-rho[rdx]
            c=smf.OLS(y_0,xdata).fit().params
            par2[isim,tdx,rdx]=c[1]
            bias2[isim,tdx,rdx]=par2[isim,tdx,rdx]-rho[rdx]
print(bias1.mean(0))
print(bias2.mean(0))

x_var=[]
for tdx, T in enumerate(ts):
    for rdx, r in enumerate(rho):
        x=r/T
        x_var.append(x)
print(x_var)
type(x_var)
y1_var=np.reshape(bias1.mean(0),len(ts)*len(rho))
y2_var=np.reshape(bias2.mean(0),len(ts)*len(rho))

model1=smf.OLS(y1_var,x_var).fit()
model2=smf.OLS(y2_var,x_var).fit()
model1.summary()
model2.summary()

