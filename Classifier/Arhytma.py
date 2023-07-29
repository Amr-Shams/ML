# Load the data to https://archive.ics.uci.edu/ml/datasets/Arrhythmia.

import numpy as np
from pathlib import Path
import pandas as pd
import tarfile
import urllib.request
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

df=pd.read_csv("arrhythmia.csv",header=None)


# Handling missing values
# find the missing values
print(df.isnull().sum())

# replace the '?'s with NaN
df = df.replace('?', np.NaN)
print(df.isnull().sum())
