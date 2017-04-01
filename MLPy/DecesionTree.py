from sklearn.feature_extraction import DictVectorizer
import csv
from sklearn import preprocessing
from sklearn import tree
from sklearn.externals.six import StringIO

allData = open(r'D:my.csv','rb')
reader=csv.reader(allData)
headers=reader.next()

print(headers)

featureList=[]
labelList=[]

for row in reader:
    labelList.append(row[len(row)-1])
    rowDict={}
    for i in range(1, len(row)-1):
        rowDict[headers[i]]=row[i]
    featureList.append(rowDict)
print(featureList)

vec=DictVectorizer()
dummyX=vec.fit_transform(featureList).toarray()

print('dummyX:'+str(dummyX))
print(vec.get_feature_names())
print('labelList:'+str(labelList))

lb=preprocessing.LabelBinarizer()
dummyY=lb.fit_transform(labelList)
print('dummyY:'+str(dummyY))

clf=tree.DecisionTreeClassifier(criterion='entropy')
clf=clf.fit(dummyX, dummyY)
print('clf:'+str(clf))

with open('alldata.dot', 'w') as f:
    f=tree.export_graphviz(clf, feature_names=vec.get_feature_names())
onRowX=dummyX[0, :]
print('oneRowX:'+str(onRowX))

newRowX=onRowX
newRowX[0]=1
newRowX[2]=0
print('newRowX:'+str(newRowX))

predictedY=clf.predict(newRowX)
print('predictedY:'+str(predictedY))