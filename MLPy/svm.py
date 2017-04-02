from sklearn import svm
'''
#Simple Example
x=[[2,0],[1,1],[2,3]]
y=[0,0,1]

clf=svm.SVC(kernel='linear')
clf.fit(x,y)

print clf
#get support vectors
print clf.support_vectors_
#get indices of support vectors
print clf.support_
#get number of support vectors for each class
print clf.n_support_


print clf.predict([2,0])
'''

print(__doc__)
import numpy as np
import pylab as pl

np.random.seed(0)
X=np.r_[np.random.randn(20,2)-[2,2], np.random.randn(20,2)+[2,2]]
Y=[0]*20+[1]*20

print Y

clf=svm.SVC(X,Y)
clf.fit(X,Y)


w=clf.coef_[0]
a=-w[0]/w[1]
xx=np.linspace(-5,5)
yy=a*xx-(clf.intercept_[0])/w[1]

