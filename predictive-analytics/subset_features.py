"""
Find most informative subset of 2 features regarding the
output of a dataset.
A regression problem is considered.
"""

import numpy as np
import pandas as ps

# Choice of models inspired by
# https://arxiv.org/pdf/1708.05070.pdf
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.datasets import load_diabetes

from itertools import combinations

def render_model(model, X, y):
    # Make data.
    nump = 51
    X1 = np.linspace(min(X[:, 0]), max(X[:, 0]), nump)
    X2 = np.linspace(min(X[:, 1]), max(X[:, 1]), nump)

    X1, X2 = np.meshgrid(X1, X2)
    Z = X1 * 0.0

    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            Z[i,j] = model.predict([[X1[i,j], X2[i,j]]])[0]

    return X1, X2, Z

def sc(I=None):
    gbrt = {
        'model': [GradientBoostingRegressor()],
        'model__n_estimators': [2 ** i for i in range(1, 11)],
    }
    lin = {
        'model': [Lasso()],
        'model__alpha': [2.0 ** i for i in range(-20, 20)],
    }
    knn = {
        'model': [KNeighborsRegressor()],
        'model__n_neighbors': [1]
    }
    svr = {
        'model': [SVR()],
        'model__C': [10.0 ** i for i in range(-4, 4)],
        'model__gamma': [10.0 ** i for i in range(-4, 4)],
        'model__epsilon': [10.0 ** i for i in range(-4, 4)],
    }

    model = GridSearchCV(
        estimator=Pipeline([
            ('scale', RobustScaler()),
            ('model', GradientBoostingRegressor()),
        ]),
        param_grid=[
            #gbrt,
            #lin,
            #knn,
            svr,
        ],
        n_jobs=-1,
    )

    csv = ps.read_csv('concrete.csv')
    XY = csv.as_matrix()

    X = XY[:, :-1]
    y = XY[:, -1]

    if I is None:
        I = range(X.shape[-1])

    X = X[:, I]

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)

    render = True

    if render:
        print(X.shape[-1])
        import plotly.offline as py
        import plotly.graph_objs as go
        import numpy as np

        Xp, Yp, Zp = render_model(model, X, y)

        data = [
            go.Scatter3d(
                x=X[:, 0], y=X[:, 1], z=y,
                mode='markers', marker={'size': 3}),
            go.Surface(
                x=Xp, y=Yp, z=Zp
            )
        ]

        layout = go.Layout(
            title='Dataset',
            autosize=True,
            margin=dict(l=65, r=50, b=65, t=90),
            scene=go.Scene(
                xaxis=dict(title=csv.columns[0]),
                yaxis=dict(title=csv.columns[1]),
                zaxis=dict(title=csv.columns[-1]),
            )
        )
        fig = go.Figure(data=data, layout=layout)
        py.plot(fig)


    return score

from tqdm import tqdm

"""
Plant output:
I=None, 0.96
I=(0, 1), 0.968027979859
I=(0, 2), 0.921976342767
I=(0, 3), 0.931381517996
I=(1, 2), 0.903416215633
I=(1, 3), 0.916395421384

Diabetes:
I=(2, 3), 0.45
I=(3, 8), 0.44
I=(8, 9), 0.38
"""

# index set of columns
combs = list(combinations(range(8), 2))

cv = []
print("base:")
print(sc())

search = False

if search:
    for I in tqdm(combs):
        cv.append((I, sc(I)))

    cv.sort(key=lambda x: x[-1], reverse=True)
    print(cv)