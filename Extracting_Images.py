import cPickle

def unpickle(file):
	with open(file, 'rb') as fo:
		dict = cPickle.load(fo)
	return dict