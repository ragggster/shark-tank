## Easy unpickler

from cPickle import load
import cPickle

def unpickle(filename):
    f = open(filename, "rb")
    print dir(cPickle)
    d = load(f)
    f.close()
    return d

if __name__ == '__main__':
	print "hi"
# d = unpickle("filename_to_link.p")
# for k, v in d.items(): 
# 	print k, v