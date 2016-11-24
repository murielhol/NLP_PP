from graph_construction import *
import numpy as np 

w = get_weight_matrix('europarl-v7.pt-en.en')
print "first row:"+str(w[0])
print "dimensions: "+str(np.shape(w))