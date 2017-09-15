import pandas as ps
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import numpy as np

data = ps.read_csv("winequality-white.csv", sep = ';')

XY = data.as_matrix()

X = XY[:, :-1]
Y = XY[:, -1]

# see a distribution of outputs

"""
plt.hist(Y, 10)
plt.show()
"""

# separate datasaet into training, testing, validation parts:
# [ training: 70%               | validtn.: 15% | testing: 15% ]
tr_ix = int(len(X) * 0.7)
vl_ix = tr_ix + int(len(X) * 0.15)

X, Xv, Xt = X[:tr_ix], X[tr_ix:vl_ix], X[vl_ix:]
Y, Yv, Yt = Y[:tr_ix], Y[tr_ix:vl_ix], Y[vl_ix:]

# perform normalization of the data
""""""
sc = StandardScaler()
sc.fit(X)
X = sc.transform(X)
Xv = sc.transform(Xv)
Xt = sc.transform(Xt)

# example fitting of the model
model = SVR()
model.fit(X,Y)
Yp = model.predict(Xv)
print model.score(Xv, Yv)
print np.mean(abs(Yp - Yv))

# example parameter selection
scores = []

for c in [0.001, 0.01,0.1,1.0,10.0,100.0]:#
    model = SVR(C = c)
    model.fit(X, Y)
    Yp = model.predict(Xv)
    score = np.mean(abs(Yp - Yv))
    print [c, score]
    scores.append([c, score])

scores = np.array(scores)
# get parameter of the best score
best = min(scores, key= lambda p: p[1])

# train on concatenaded arrays
model = SVR(C = best[0])
XXv = np.concatenate([X, Xv])
YYv = np.concatenate([Y, Yv])
model.fit( XXv , YYv )
Yp = model.predict(Xt)
score = np.mean(abs(Yp - Yt))
print "test error:", score

# print score distribution
plt.plot( np.log(scores[:,0]), scores[:,1] )
plt.show()