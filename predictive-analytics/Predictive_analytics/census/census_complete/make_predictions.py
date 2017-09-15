import cPickle as pc
import pandas as ps

data = ps.read_csv("predict.csv", sep = ',')
X = data.as_matrix()

obj = pc.load(open('model.bin'))
Y = obj.predict(X)

print Y


