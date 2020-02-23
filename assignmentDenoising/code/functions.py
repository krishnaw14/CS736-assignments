import numpy as np 

def rrmse(A,B):
	rrmse = np.sqrt(np.sum((A-B)**2)/np.sum(A**2))
	return rrmse

def likelihood_function(x,y):
	function_value = np.sum(np.abs(x-y)**2)
	grad_value = 2*(x-y)
	return function_value, grad_value


def quadratic_function(u, gamma):
	function_value = np.sum(np.abs(u)**2)
	grad_value = 2*u
	return function_value, grad_value
 

def discontinuity_adaptive_function(u, gamma):
	function_value = np.sum(gamma*np.abs(u) - (gamma**2)*np.log(1+np.abs(u)/gamma))
	grad_value = gamma*u/(gamma+np.abs(u))

	return function_value, grad_value


def discontinuity_adaptive_huber_function(u, gamma):
	function_value = np.sum(0.5*(np.abs(u)**2)*(np.abs(u) <= gamma) + (gamma*np.abs(u) - 0.5*gamma**2)*(np.abs(u) > gamma))
	grad_value = u*(np.abs(u) <= gamma) + gamma*np.sign(u)*(np.abs(u) > gamma)

	return function_value, grad_value