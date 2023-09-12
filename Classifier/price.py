import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncodercvcv
from sklearn.preprocessing import OrdinalEncoder

# load data
test = pd.read_csv('test.csv')
train = pd.read_csv('train.csv')

# see the data
# create a function to see the data
def see_data(data):
    print(data.head(5))
    print(data.shape)
    print(data.info())
    print(data.describe())

# see_data(train)
# see_data(test)

# data visualization
# Bar chart for categorical features
"""
1- Pclass
2- Sex
3- SibSp
4- Parch
5- Embarked
6- Cabin
7- Name
8- Ticket

"""
def bar_chart(feature):
    survived = train[train['Survived']==1][feature].value_counts()
    dead = train[train['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index = ['Survived','Dead']
    df.plot(kind= 'bar', title=feature, figsize=(15,10), stacked=True)
    plt.show()

# a function to get the ratio of survived and dead
def ratio(feature):
    survived = train[train['Survived']==1][feature].value_counts()
    dead = train[train['Survived']==0][feature].value_counts()
    ratio = survived/dead
    # print the ratio and the survived perecentage and dead percentage
    print(ratio)


# create the pipeline
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('std_scaler', StandardScaler()),
])

# create the pipeline for categorical features
cat_pipeline = Pipeline([
        ("ordinal_encoder", OrdinalEncoder()),
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("cat_encoder", OneHotEncoder(sparse=False)),
    ])

# create the full pipeline
num_attribs = ['Age', 'SibSp', 'Parch', 'Fare']
cat_attribs = ["Pclass", "Sex", "Embarked"]
preprocess_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs),
    ])

# prepare the data
X_train = train.drop('Survived', axis=1)
y_train = train['Survived']
train_prepared = preprocess_pipeline.fit_transform(X_train)
test_prepared = preprocess_pipeline.transform(test)

# choose the model
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# create a function to get the score of the model
def get_score(model):
    model.fit(train_prepared, y_train)
    return model.score(train_prepared, y_train)
# create a function to get the cross validation score
def get_cross_val_score(model):
    from sklearn.model_selection import cross_val_score
    return cross_val_score(model, train_prepared, y_train, cv=10, scoring="accuracy")
# create a function to get the confusion matrix
def get_confusion_matrix(model):
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import confusion_matrix
    y_train_pred = cross_val_predict(model, train_prepared, y_train, cv=3)
    return confusion_matrix(y_train, y_train_pred)
# create a function to get the f1 score
def get_f1_score(model):
    from sklearn.metrics import f1_score
    from sklearn.model_selection import cross_val_predict
    y_train_pred = cross_val_predict(model, train_prepared, y_train, cv=3)
    return f1_score(y_train, y_train_pred)

"""
# Logistic Regression
log_reg = LogisticRegression()
log_reg_score = get_score(log_reg)
log_reg_cross_val_score = get_cross_val_score(log_reg)
log_reg_confusion_matrix = get_confusion_matrix(log_reg)
print(log_reg_score, log_reg_cross_val_score, log_reg_confusion_matrix, get_f1_score(log_reg), sep='\n')


# SVC
svc = SVC()
svc_score = get_score(svc)
svc_cross_val_score = get_cross_val_score(svc)
svc_confusion_matrix = get_confusion_matrix(svc)
svc_f1_score = get_f1_score(svc)
print(svc_score, svc_cross_val_score, svc_confusion_matrix, get_f1_score(svc),svc_f1_score, sep='\n')

# Random Forest Classifier
forest_clf = RandomForestClassifier()
forest_clf_score = get_score(forest_clf)
forest_clf_cross_val_score = get_cross_val_score(forest_clf)
forest_clf_confusion_matrix = get_confusion_matrix(forest_clf)
forest_clf_f1_score = get_f1_score(forest_clf)
print(forest_clf_score, forest_clf_cross_val_score, forest_clf_confusion_matrix, get_f1_score(forest_clf),forest_clf_f1_score, sep='\n')

# KNeighborsClassifier
knn = KNeighborsClassifier()
knn_score = get_score(knn)
knn_cross_val_score = get_cross_val_score(knn)
knn_confusion_matrix = get_confusion_matrix(knn)
knn_f1_score = get_f1_score(knn)
print(knn_score, knn_cross_val_score, knn_confusion_matrix, get_f1_score(knn),knn_f1_score, sep='\n')
"""
# do some feature engineering
# tune the hyperparameters
from sklearn.model_selection import GridSearchCV
param_grid = [
    {'n_estimators': [3, 10, 30, 50, 100], 'max_features': [2, 4, 6, 8, 10, 12, 14, 16]},
    {'bootstrap': [False], 'n_estimators': [3, 10, 30, 50, 100], 'max_features': [2, 3, 4, 5, 6, 7, 8, 9, 10]},
    ]
forest_clf = RandomForestClassifier()
grid_search = GridSearchCV(forest_clf, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
grid_search.fit(train_prepared, y_train)
print(grid_search.best_params_)
print(grid_search.best_estimator_)
print(grid_search.best_score_)
# get the feature importance
feature_importances = grid_search.best_estimator_.feature_importances_
print(feature_importances)
# get the score
final_model = grid_search.best_estimator_
final_score = get_score(final_model)
final_cross_val_score = get_cross_val_score(final_model)
final_confusion_matrix = get_confusion_matrix(final_model)
final_f1_score = get_f1_score(final_model)
print(final_score, final_cross_val_score, final_confusion_matrix, get_f1_score(final_model),final_f1_score, sep='\n')

# predict the test data
y_pred = final_model.predict(test_prepared)
print(y_pred)


import numpy as np

np.random.seed(42)  # to make this code example reproducible
m = 100  # number of instances
X = 2 * np.random.rand(m, 1)  # column vector
y = 4 + 3 * X + np.random.randn(m, 1)  # column vector

# let's plot our data as points in 2D space
plt.plot(X, y, "b.")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.axis([0, 2, 0, 15])
plt.show()
