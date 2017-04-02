from __future__ import print_function

from time import time
import logging
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split
from sklearn.datasets import  fetch_lfw_people
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import RandomizedPCA
from sklearn.svm import SVC

print(__doc__)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
lfw_people=fetch_lfw_people(min_faces_per_person=70, resize=0.4)
n_samples, h, w=lfw_people.images.shape

X=lfw_people.data