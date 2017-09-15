import numpy as np

class OneHotEncoder():
    def fit(self, X):
        # here X is one column
        # determine all the unique values
        U = np.unique(X)
        I = range(len(U))

        # this dictionary is used to encode the values
        self.forward = {key: value for key, value in zip(U, I)}

    def onehot(self, idx, max):
        z = np.zeros(max)
        z[idx] = 1.0
        return z

    def transform(self,X):
        result = [ self.onehot( self.forward[x], len(self.forward)) for x in X]
        return np.array(result)

if __name__ == "__main__":
    # example usage
    X = np.array(['a','b','ab','a','b'])
    enc = OneHotEncoder()
    # "fit" encoder to the training column
    enc.fit(X)
    # encode some data (same data in this case)
    print enc.transform(X)
