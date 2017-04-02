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

#在此处就是带入求值
b=clf.support_vectors_[0]
yy_down=a*xx+(b[1]-a*b[0])
b=clf.support_vectors_[-1]
yy_up=a*xx+(b[1]-a*b[0])

print 'w:', w
print 'a:', a

print 'support_vector:', clf.support_vectors_
print 'clf.coef_:', clf.coef_

pl.plot(xx, yy, 'k-')
pl.plot(xx,yy_down,'k--')
pl.plot(xx,yy_up,'kk--')

pl.scatter(clf.support_vectors_[:,0], clf.support_vectors_[:,1],
            set=80,facecolors='none')
pl.scatter(X[:,0], X[:1], c=Y, cmap=pl.cm.Paired)
pl.axis('tight')
pl.show()