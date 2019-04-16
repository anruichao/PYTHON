
import csv
import numpy as np
from scipy.stats import norm

file = open('bwght.csv')
file.close()

inmat = []
with open('bwght.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        inmat.append(line)

#print(inmat[0])
#for line in inmat[1:]:
#    for item in line:
#        print(item)
#        print(float(item))
inmat = [row[:4]+row[6:] for row in inmat]
for ndx,item in enumerate(inmat[0]):
    print(ndx,item)

data = np.matrix(inmat[1:],dtype=float)
#print(data)

x = np.array([1,2,3])
print(x)
y = np.array([4,5,6])
print(y)
print(x+y) # addition
print(x*y) # element-wise multiplication
print(4*x) # scalar multiplication

x = [[1,2],[3,4]]
y = [[5],[8]]
x = np.matrix(x)
y = np.matrix(y)
print(x+y)  # gives an answer because of broadcasting (I hate this)
print(x*y)
print(np.dot(x,y))
print(x.dot(y))
print(np.transpose(x))
print(x.transpose())
print(x.T)
print(x.T.dot(x))
print(x.T.dot(y))
print(np.linalg.inv(x.T.dot(x)))
print(np.linalg.inv(x.T.dot(x)).dot(x.T.dot(y)))
print(np.linalg.solve(x.T.dot(x),x.T.dot(y)))
print(np.linalg.det(x.T.dot(x)))
print(np.linalg.det(np.linalg.inv(x.T.dot(x))))
#np.linalg.solve(A,b) = inv(A)*b

y = data[:,3]
y
x = data[:,7]
x
n = len(y)
bigX = np.ones((n,1))
bigX = np.hstack((bigX,x))
bigX
b = np.linalg.solve(bigX.T.dot(bigX),bigX.T.dot(y))
b
def cross(*args):  # Can put either 1 or 2 things into this function
    x = args[0]
    if len(args)==1:
        y = x
    else:
        y = args[1]
    return(x.T.dot(y))
def least_squares(X,y):
    return(np.linalg.solve(cross(X),cross(X,y)))
def ols(X,y):
    # goals: bhats, se, tstats, pvals, Rsq
    b = least_squares(X,y)
    yhat = X.dot(b)
    e = y-yhat
    RSS = cross(e)[0,0]
    ybar = y.mean()
    e_0 = y-ybar
    TSS = cross(e_0)[0,0]
    Rsq = 1-RSS/TSS
    sig = np.sqrt(RSS/n)
    vb = np.linalg.inv(cross(X))*(sig**2)
    se = np.sqrt(np.diag(vb)).reshape(-1,1)
    tstats = b/se
    pval = norm.cdf(-np.abs(tstats))*2
    return(b,se,tstats,pval,Rsq)
def white(X,y):
    # goals: bhats, se, tstats, pvals, Rsq
    b = least_squares(X,y)
    yhat = X.dot(b)
    e = y-yhat
    RSS = cross(e)[0,0]
    ybar = y.mean()
    e_0 = y-ybar
    TSS = cross(e_0)[0,0]
    Rsq = 1-RSS/TSS
    sig = np.sqrt(RSS/n)
    vb = np.diagflat(np.power(e,2))
    vb = X.T.dot(vb).dot(X)
    vb = np.linalg.inv(cross(X))*vb*np.linalg.inv(cross(X))
    se = np.sqrt(np.diag(vb)).reshape(-1,1)
    tstats = b/se
    pval = norm.cdf(-np.abs(tstats))*2
    return(b,se,tstats,pval,Rsq)
def tidyols(X,y):
    (b,se,tstats,pval,Rsq) = ols(X,y)
    print(np.hstack((b,se,tstats,pval)))
    print("RSq="+str(Rsq))    
def tidywhitey(X,y):
    (b,se,tstats,pval,Rsq) = white(X,y)
    print(np.hstack((b,se,tstats,pval)))
    print("RSq="+str(Rsq))    
    
    
data
ols(bigX,y)
tidyols(bigX,y)
bigX = np.hstack((np.ones((n,1)),data[:,7],data[:,0],data[:,5],data[:,6]))
    
tidyols(bigX,y)
tidywhitey(bigX,y)
    
    
    
    
    
    
    
    
    
    
    
    


