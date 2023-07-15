import pandas as pd
import numpy as np
from sklearn import model_selection, ensemble, metrics, preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
import joblib
import matplotlib.pyplot as plt

# 1.Load Data
dataset_url='http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
housing=pd.read_csv(dataset_url)
print(housing.head(3))


# plot the data attributes
plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
housing.hist(bins=100,figsize=(10,10))
plt.show()


# 2. Split Data
housing['incom_cat']=pd.cut(housing['median_incom'],bins=[0,5,6,7,10],labels=[1,2,3,4])
train_data, test_data = model_selection.train_test_split(housing, test_size=0.2, random_state=42)

# find the correlation between each pair of attributes
