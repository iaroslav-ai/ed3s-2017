import numpy as np
import pandas as ps
from time import time

# Choice of models inspired by
# https://arxiv.org/pdf/1708.05070.pdf
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier

from sklearn.preprocessing import RobustScaler, FunctionTransformer, Imputer, OneHotEncoder
from sklearn.pipeline import Pipeline, make_pipeline, make_union
from sklearn.model_selection import GridSearchCV, train_test_split


def idx(X, pos=0):
    return X[[pos]].as_matrix()


# read the data
csv = ps.read_csv('titanic.csv')
y = csv['Survived']
X = csv

feats = make_union(
    make_pipeline(
        FunctionTransformer(idx, kw_args={'pos': 'Fare'}, validate=False),
        Imputer(),
    ),
    make_pipeline(
        FunctionTransformer(idx, kw_args={'pos': 'Pclass'}, validate=False),
        OneHotEncoder(sparse=False),
    ),
)


model = GridSearchCV(
    estimator=Pipeline([
        ('feats', feats),
        ('scale', RobustScaler()),
        ('model', GradientBoostingClassifier()),
    ]),
    param_grid=[{
        'model': [GradientBoostingClassifier()],
        'model__n_estimators': [2 ** i for i in range(1, 11)],
        #'model__learning_rate': [2 ** i for i in range(-10, 10)],
    }],
    n_jobs=-1,
)



# split data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75)


# search for best hyperparameters
model.fit(X_train, y_train)
dummy_model = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)

# evaluate model
test_score = model.score(X_test, y_test)
dummy_score = dummy_model.score(X_test, y_test)

print(dummy_score, test_score)