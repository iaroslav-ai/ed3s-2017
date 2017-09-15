import requests
import random
from joblib import Parallel, delayed

def register():
    r = requests.post('http://52.59.7.147/process', json={
        'email': ''.join([random.choice('abcdefghijklmn') for i in range(10)])
    })

    return r.json()

n = 35
r = Parallel(n_jobs=n)(delayed(register)() for i in range(n))

vals = []

for v in r:
    if 'ip' in v:
        vip = v['ip']

        if vip in vals:
            print("!!!!!!!! repeated IP")

        vals.append(vip)


print(r)