import requests
import random
from joblib import Parallel, delayed

def register():
    r = requests.post('http://localhost:8888/process', json={
        'email': ''.join([random.choice('abcdefghijklmn') for i in range(10)])
    })

    return r.json()

n = 5
r = Parallel(n_jobs=n)(delayed(register)() for i in range(n))

print(r)