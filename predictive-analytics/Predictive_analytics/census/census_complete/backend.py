"""
This file contains classes needed for data preprocessing
"""

from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.svm import SVC
import cPickle as pc
from onehot import OneHotEncoder

class PreprocessCensusData():

    def __init__(self, add_noise = False, do_norm = True):
        self.column_noise = add_noise
        self.normalize = do_norm

    def encode_X(self,X):

        Rx = []
        for x, e in zip(X.T, self.enc_X):
            Rx.append(x if e is None else e.transform(x))

        if self.column_noise:
            Rx.append( np.random.rand( X.shape[0], 100 ) )

        Rx = np.column_stack(Rx)
        return Rx


    def normalize_X(self,X):
        return self.scaler.transform(X)

    def predict(self, X):
        """

        :param X: numpy matrix with contents of csv with values for predictions
        :return:
        """

        X = self.encode_X(X)
        X = self.normalize_X(X)
        Yp = self.model.predict(X)

        R = []

        for y in Yp:
            if y > 0.0:
                R.append(">50K")
            else:
                R.append("<=50K")

        return R


    def process(self, XY):
        """
        This creates all the data needed for preprocessing of the census dataset

        :param XY: Contents of CSV file as a numpy array
        :return: None
        """

        # create one hot enecoding
        self.enc_X = []


        for i in range(XY.shape[1]-1):
            # columns for which no encoding is required
            if i in [0,2,4, 10, 11, 12]:
                self.enc_X.append(None)
            else:
                enc = OneHotEncoder()
                enc.fit(XY[:,i])

                self.enc_X.append(enc)


        X = self.encode_X(XY[:,:-1])

        # encode target value
        Y = (XY[:,-1] == " >50K")*2.0 - 1

        # separate datasaet into training, testing, validation parts:
        # [ training: 60%               | validtn.: 20% | testing: 20% ]
        tr_ix = int(len(X) * 0.6)
        vl_ix = tr_ix + int(len(X) * 0.2)

        X, Xv, Xt = X[:tr_ix], X[tr_ix:vl_ix], X[vl_ix:]
        Y, Yv, Yt = Y[:tr_ix], Y[tr_ix:vl_ix], Y[vl_ix:]

        if self.normalize:
            sc = StandardScaler()
            sc.fit(X)
            self.scaler = sc

            X = self.normalize_X(X)
            Xv = self.normalize_X(Xv)
            Xt = self.normalize_X(Xt)

        scores = []

        """
        X = X[:1000]
        Y = Y[:1000]
        """

        # create and train predictive model
        for C in [0.01, 0.1, 1.0, 10.0, 100.0]:#
            for G in [0.001, 0.005, 0.01, 0.1]:#
                clsf = SVC(C=C, gamma=G)
                clsf.fit(X,Y)
                Yp = clsf.predict(Xv)
                acc = np.mean( Yp == Yv )
                scores.append({'acc':acc, 'C':C, 'G':G})
                print {'acc':acc, 'C':C, 'G':G}

        # find the best one
        params = max(scores, key = lambda p: p['acc'])

        model = SVC(C=params['C'], gamma=params['G'])
        XXv = np.concatenate([X, Xv])
        YYv = np.concatenate([Y, Yv])
        model.fit(XXv, YYv)
        Yp = model.predict(Xt)
        score = np.mean( Yp == Yt )
        print "test accuracy:", score

        self.test_accuracy = score
        self.model = model
