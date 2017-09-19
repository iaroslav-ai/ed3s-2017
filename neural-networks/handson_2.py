import pickle as pc
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin

from keras.models import Model
from keras.layers import Dense, Input, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam

from misc import make_keras_picklable

# necessary for pickling to work
make_keras_picklable()

class DNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neurons=256, epochs=1, dropout=False,
                 batch_size=256, learning_rate=1e-3, beta_1=0.9,
                 beta_2=0.999):

        self.n_neurons = n_neurons
        self.epochs = epochs
        self.dropout = dropout
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def fit(self, X, y):
        # define network architecture
        x = Input(shape=X[0].shape)
        h = x
        h = Dense(self.n_neurons)(h)
        h = LeakyReLU()(h)
        if self.dropout:
            h = Dropout(0.5)(h)
        h = Dense(10, activation='softmax')(h)

        self.model = Model(inputs=x, outputs=h)

        # compile computational graph of NN
        self.model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=Adam(
                          lr=self.learning_rate,
                          beta_1=self.beta_1,
                          beta_2=self.beta_2,
                      ),
                      metrics=['accuracy'])

        # train NN
        self.model.fit(X, y,
                        batch_size=self.batch_size,
                        epochs=self.epochs,
                        verbose=1)

    def predict(self, X):
        return np.argmax(self.model.predict(X), axis=-1)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

X, y = pc.load(open('mnist.pc', 'rb'))

# reshape images to vector
X = np.reshape(X, (len(X), -1))

# split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75)

model = Pipeline([
    ('scale', StandardScaler()),
    ('model', DNNClassifier(epochs=32, n_neurons=1024, dropout=True))
])

model.fit(X_train, y_train)
pc.dump(model, open('model.bin', 'wb'))

model = pc.load(open('model.bin', 'rb'))
print(model.score(X_test, y_test))