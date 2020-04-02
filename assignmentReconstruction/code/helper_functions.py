import numpy as np 

def get_rrmse(A,B):
	rrmse = np.sqrt(np.sum((A-B)**2)/np.sum(A**2))
	return rrmse

def my_filter(img, filter_name=None):

	if filter_name == 'ram-lak':
		pass
	