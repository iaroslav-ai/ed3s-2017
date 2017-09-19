import numpy as np
import pickle as pc
import matplotlib.pyplot as plt
from scipy.misc import imresize

X, y = pc.load(open('m.bin', 'rb'))

X = np.array([imresize(x, 0.5) for x in X])

pc.dump((X, y), open('mnist.pc', 'wb'))