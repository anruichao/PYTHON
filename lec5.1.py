# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 16:27:20 2018

@author: jap090020
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

context1 = pd.read_csv('bwght.csv')
(n,r) = context1.shape
n
r
context1
context1['bwght']
context1['bwght'][0]
context1['cigs']

context1.mean()
context1.std()
context1.min()
context1.median()
context1.max()

def sstats(data):
    df = (data.mean(),
          data.std(),
          data.min(),
          data.median(),
          data.max(),
          data.isnull().mean())
    df = pd.concat(df,axis=1)
    df.columns = ['mean','st dev','min','median','max','na portion']
    return df

sstats(context1)

models = [smf.ols('bwght~cigs',data=context1).fit(),
          smf.OLS(context1['bwght'],context1['cigs']).fit(),
          smf.OLS(context1['bwght'],sm.add_constant(context1['cigs'])).fit(),
          smf.ols('bwght~cigs+cigs^2',data=context1).fit(),
          smf.ols('bwght~cigs+faminc',data=context1).fit(),
          smf.ols('bwght~cigs*faminc',data=context1).fit(),
          smf.ols('np.log(bwght)~cigs',data=context1).fit(),
          smf.ols('np.log(bwght)~cigs',data=context1).fit(cov_type='HC3')]
#print(models[0].summary())

for (ndx,item) in enumerate(models):
    print('Model '+str(ndx)+':')
    print(item.summary())
    print('\n')

models[0].predict()
Xnew = np.matrix([[0],[20],[40]])
Xnew = sm.add_constant(Xnew)
models[0].predict(Xnew,transform=False)
Xnew = np.matrix([[0],[20],[40]])
Xnew = pd.DataFrame(Xnew)
Xnew.columns = ['cigs']
models[0].predict(Xnew)


context1['smokes'] = (context1['cigs']>0).astype(int)


models_new = [smf.ols('bwght~smokes',data=context1).fit(),
              smf.ols('smokes~faminc+motheduc+fatheduc',data=context1).fit(),
              smf.logit('smokes~faminc+motheduc+fatheduc',data=context1).fit(),
              smf.glm('smokes~faminc+motheduc+fatheduc',data=context1,family=sm.families.Binomial()).fit(),
              smf.glm('cigs~faminc+motheduc+fatheduc',data=context1,family=sm.families.Poisson()).fit()]


for (ndx,item) in enumerate(models_new):
    print('New Model '+str(ndx)+':')
    print(item.summary())
    print('\n')


context1['smokes'] = (context1['cigs']>0).astype(int)
context1['cigsq'] = context1['cigs']**2
context1['cigprod'] = context1['cigs']*context1['cigtax']
pd.concat((context1['cigs'],context1['cigsq'],context1['cigprod']),axis=1)
context1['educ'] = context1['motheduc']+context1['fatheduc']
context1['educ_star'] = context1['motheduc']+np.where(context1['fatheduc'].isnull(),0,context1['fatheduc'])
pd.concat((context1['fatheduc'],context1['motheduc'],context1['educ'],context1['educ_star']),axis=1)

context1

print(context1.groupby(['cigs']).mean())
