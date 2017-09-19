import pickle as pc
import numpy as np

from sklearn.model_selection import train_test_split

from keras.models import Model
from keras.layers import Dense, Input
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam

X, y = pc.load(open('mnist.pc', 'rb'))

# reshape images to vector
X = np.reshape(X, (len(X), -1))

# split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75)

# define network architecture
x = Input(shape=X_train[0].shape)
h = x
h = Dense(256)(h)
h = LeakyReLU()(h)
h = Dense(10, activation='softmax')(h)
y = h

model = Model(inputs=x, outputs=y)

# compile computational graph of NN
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

# train NN
model.fit(X_train, y_train,
                    batch_size=256,
                    epochs=10,
                    verbose=1,)

# evaluate your NN
score = model.evaluate(X_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])