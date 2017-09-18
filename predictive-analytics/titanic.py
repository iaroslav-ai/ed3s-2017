import numpy as np
import pandas as ps
from time import time

# Choice of models inspired by
# https://arxiv.org/pdf/1708.05070.pdf
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier

from sklearn.preprocessing import RobustScaler, OneHotEncoder, Imputer
from sklearn.pipeline import Pipeline, make_pipeline, make_union
from sklearn.model_selection import GridSearchCV, train_test_split

from misc import ColumnSelector, IntEncoder

# read the data
csv = ps.read_csv('titanic.csv')

# get the output
y = csv['Survived']

# delete the output from csv
del csv['Survived']

X = csv.as_matrix()
col_idx = {v: i for i, v in enumerate(csv.columns)}

features = make_union(
    make_pipeline(
        ColumnSelector(col_idx['Fare']),
        Imputer(), # replaces missing values with mean of values
    ),
    make_pipeline(
        ColumnSelector(col_idx['Pclass']),
        IntEncoder(),
        OneHotEncoder(sparse=False),
    ),
    make_pipeline(
        ColumnSelector(col_idx['Embarked']),
        IntEncoder(),
        OneHotEncoder(sparse=False),
    ),
    make_pipeline(
        ColumnSelector(col_idx['Gender']),
        IntEncoder(),
        OneHotEncoder(sparse=False),
    ),
)

"""
features.fit(X)
X = features.transform(X)
"""

estimator = Pipeline([
    ('feats', features),
    ('scale', RobustScaler()),
    ('model', GradientBoostingClassifier()),
])

gbrt = {
    'model': [GradientBoostingClassifier()],
    'model__n_estimators': [2 ** i for i in range(1, 11)],
    'model__learning_rate': [2 ** i for i in range(-10, 10)],
}
svc = {
    'model': [SVC()],
    'model__C': [10 ** i for i in np.linspace(-6, 6, 20)],
    'model__gamma': [10 ** i for i in np.linspace(-6, 6, 20)],
}
linsvc = {
    'model': [LinearSVC(max_iter=10000)],
    'model__C': [10 ** i for i in np.linspace(-6, 6, 20)],
}
knn = {
    'model': [KNeighborsClassifier()],
    'model__n_neighbors': range(1, min(len(X)-1, 256)),
}
dectree = {
    'model': [DecisionTreeClassifier()],
    'model__max_depth': range(1, 20),
    'model__min_samples_split': [2 ** i for i in range(-20, -1)],
}

model = GridSearchCV(
    estimator=estimator,
    param_grid=[gbrt, linsvc, knn, dectree], # svc,
    n_jobs=-1,
    verbose=1
)

dummy_model = GridSearchCV(
    estimator=estimator,
    param_grid=[{
        'model': [DummyClassifier()],
        'model__strategy': ['most_frequent', 'uniform'],
    }],
    n_jobs=1,
)

# split data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75)

# search for best hyperparameters
model.fit(X_train, y_train)
dummy_model.fit(X_train, y_train)

# evaluate models
test_score = model.score(X_test, y_test)
dummy_score = dummy_model.score(X_test, y_test)

print("Trivial accuracy: %s" % dummy_score)
print("Model accuracy: %s" % test_score)
