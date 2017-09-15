import pandas as ps
import cPickle as pc
from backend import PreprocessCensusData

data = ps.read_csv("census.csv", sep = ',')
XY = data.as_matrix()

obj = PreprocessCensusData()
obj.process(XY)

pc.dump(obj, open("model.bin", "w"))

# check what happens if you add noise
obj = PreprocessCensusData(add_noise=True)
obj.process(XY)

# check what happens if you do not normalize
obj = PreprocessCensusData(do_norm=False)
obj.process(XY)

