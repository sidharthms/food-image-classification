import sys  
import os  
from pylearn2.utils import serial  
from pylearn2.config import yaml_parse  

try:  
    model_path = sys.argv[1]  
    test_path = sys.argv[2]  
    out_path = sys.argv[3]  
except IndexError:  
    print "Usage: predict.py <model file> <test file> <output file>"  
    quit()  
  
try:  
    model = serial.load( model_path )  
except Exception, e:  
    print model_path + "doesn't seem to be a valid model path, I got this error when trying to load it: "  
    print e  

# dataset = ....


batch_size = 100  
model.set_batch_size(batch_size)  


X = model.get_input_space().make_batch_theano()  
Y = model.fprop(X)


from theano import tensor as T  
  
Y = T.argmax(Y, axis=1)  
  
from theano import function  
  
f = function([X], Y)
print("loading data and predicting...")

y = []
for i in xrange(dataset.X.shape[0] / batch_size):
	x = dataset.X[i*batch_size:(i+1)*batch_size,:]
	y.append(f(x))
	
print("writing predictions...")
assert y.ndim == 1  
assert y.shape[0] == dataset.X.shape[0] 

out = open(out_path, 'w')  
for i in xrange(y.shape[0]): 
    out.write( '%f\n' % (y[i]))  
out.close() 

