import numpy as np 

def likelihood_function(x,y):
	function_value = np.sum(np.abs(x-y)**2)
	grad_value = 2*(x-y)
	return function_value, grad_value


def quadratic_function(u):
	function_value = np.sum(np.abs(u)**2)
	grad_value = 2*u
	return function_value, grad_value
 

def discontinuity_adaptive_function(gamma):
	pass

def discontinuity_adaptive_huber_function(gamma):
	pass