## Easy unpickler

import cPickle

def unpickle(filename):
    f = open(filename, "rb")
    d = cPickle.load(f)
    f.close()
    return d

# d = unpickle("filename_to_link.p")
# for k, v in d.items(): 
# 	print k, v