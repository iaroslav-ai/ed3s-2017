import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

# generate example dataset
# random ground truth
N = 5
M = np.random.rand(N)*10
S = np.random.rand(N) + 1

def ensemble(X, S, M):
    return np.exp( -(np.outer(X, S) - M)**2 )

X = np.random.rand(200)*5
Y = np.sum( ensemble(X, S, M) , axis = 1)

plt.scatter(X,Y, label = 'data')
plt.axis([0,5,0,1.6])
plt.grid()
plt.legend()
plt.show()

M = None
S = None

# accumulated function values
Fp = Y*0

# perform model search
for i in range(12):
    
    best_error = 1e10
    # one way to do anything is to do it randomly
    for t in range(1000):
        
        Mp = np.random.rand(1)*10
        Sp = np.random.rand(1)+1
        
        # get outputs of new model
        F = ensemble(X, Sp, Mp)
        
        # calculate weight for candidate model
        w, r, rank, s = np.linalg.lstsq(F, Y-Fp)
        
        # check if it is better then everything seen so far
        if r[0]< best_error:
            Mn = Mp
            Sn = Sp
            Fn = F
            wb = w
            best_error = r[0]
    
    M = Mn
    S = Sn
    Fn  = Fn[:,0] * wb[0]
    Yp = Fp + Fn
    Ya = Fn
    plt.scatter(X,Y, label = 'data')
    plt.scatter(X,Yp, c='r', label = 'model')
    plt.scatter(X,Ya, c='g', label = 'added function')
    plt.axis([0,5,-0.5,1.6])
    plt.grid()
    plt.legend()
    plt.show()
    Fp = Yp