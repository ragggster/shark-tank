import dill as pickle 

with open('season4-labelled.p', 'rb') as handle: 
	b = pickle.load(handle) 

for k, v in b.iteritems(): 
	print k 

# wav_map = pickle.load(open('season4-labelled.p', 'rb'))
# for k,v in wav_map.iteritems(): 
# 	print k