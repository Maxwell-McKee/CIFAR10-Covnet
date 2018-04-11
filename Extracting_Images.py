import _pickle as pickle

def unpickle(file):
	with open(file, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
	return dict


data_dict1 = unpickle("/home/azenoni/CPSC 410 Neural Nets/cifar-10-batches-py/data_batch_1")
data_dict2 = unpickle("/home/azenoni/CPSC 410 Neural Nets/cifar-10-batches-py/data_batch_2")
data_dict3 = unpickle("/home/azenoni/CPSC 410 Neural Nets/cifar-10-batches-py/data_batch_3")
data_dict4 = unpickle("/home/azenoni/CPSC 410 Neural Nets/cifar-10-batches-py/data_batch_4")
data_dict5 = unpickle("/home/azenoni/CPSC 410 Neural Nets/cifar-10-batches-py/data_batch_5")
data_dictTest = unpickle("/home/azenoni/CPSC 410 Neural Nets/cifar-10-batches-py/test_batch")

print (unpickle("/home/azenoni/CPSC 410 Neural Nets/cifar-10-batches-py/batches.meta"))
