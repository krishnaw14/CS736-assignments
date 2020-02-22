import numpy as np 
import os
import matplotlib.pyplot as plt
from functions import *

def rrmse(A,B):
	rrmse = np.sqrt(np.sum((A-B)**2)/np.sum(A**2))
	return rrmse

# def posterior_function(x,y,prior_function, beta):
# 	likelihood = likelihood_function(x,y)

# 	prior = prior_function(x, np.roll(x, 1, axis=0))
# 	prior += prior_function(x, np.roll(x, -1, axis=0))
# 	prior += prior_function(x, np.roll(x, 1, axis=1))
# 	prior += prior_function(x, np.roll(x, -1, axis=1))

# 	posterior = (1-beta)*likelihood + beta*prior
# 	return posterior

# def posterior_derivative_function(x,y,prior_function, beta):
# 	likelihood_derivative = likelihood_derivative_function(x,y)
	
# 	prior_derivative = prior_function(x, np.roll(x, 1, axis=0))
# 	prior_derivative += prior_function(x, np.roll(x, -1, axis=0))
# 	prior_derivative += prior_function(x, np.roll(x, 1, axis=1))
# 	prior_derivative += prior_function(x, np.roll(x, -1, axis=1))

# 	posterior_derivative = (1-beta)*likelihood_derivative + beta*prior_derivative

# 	return posterior_derivative

def calculate_prior(x):
	prior_1, grad_1 = quadratic_function(x - np.roll(x, 1, axis=0))
	prior_2, grad_2 = quadratic_function(x - np.roll(x, -1, axis=0))
	prior_3, grad_3 = quadratic_function(x - np.roll(x, 1, axis=1))
	prior_4, grad_4 = quadratic_function(x - np.roll(x, -1, axis=1))

	prior = prior_1 + prior_2 + prior_3 + prior_4
	grad = grad_1 + grad_2 + grad_3 + grad_4
	return prior, grad


def denoise(noisy_img, denoised_img, beta=0.6, optimize_mode=False, prior='quadratic'):

	if not optimize_mode:
		print('Initial RRMSE between Noisy and Denoised Image:', rrmse(denoised_img, noisy_img))
	m, n = denoised_img.shape

	# Create Grid indices for parallel update
	m_ind_1 = np.arange(0,m,2)
	m_ind_2 = np.arange(1, m, 2)
	n_ind_1 = np.arange(0, n, 2)
	n_ind_2 = np.arange(1, n, 2)

	# Using Notation used in class
	x = 1.0*noisy_img.copy() # Initial Estimate
	y = 1.0*noisy_img.copy() # Observed Data

	alpha = 1e-2

	likelihood, likelihood_grad = likelihood_function(x,y)
	prior, prior_grad = calculate_prior(x)

	initial_post = beta*prior + (1-beta)*likelihood
	
	counter = 0	
	post_values = [initial_post]
	while counter < 100 and alpha > 1e-8:
		

		likelihood, likelihood_grad = likelihood_function(x,y)
		prior, prior_grad = calculate_prior(x)
		post = beta*prior + (1-beta)*likelihood
		post_grad = beta*prior_grad + (1-beta)*likelihood_grad
		x[0:m:2, 0:n:2] = x[0:m:2, 0:n:2]-alpha*post_grad[0:m:2, 0:n:2]

		likelihood, likelihood_grad = likelihood_function(x,y)
		prior, prior_grad = calculate_prior(x)
		post = beta*prior + (1-beta)*likelihood
		post_grad = beta*prior_grad + (1-beta)*likelihood_grad
		x[0:m:2, 1:n:2] = x[0:m:2, 1:n:2]-alpha*post_grad[0:m:2, 1:n:2]

		likelihood, likelihood_grad = likelihood_function(x,y)
		prior, prior_grad = calculate_prior(x)
		post = beta*prior + (1-beta)*likelihood
		post_grad = beta*prior_grad + (1-beta)*likelihood_grad
		x[1:m:2, 0:n:2] = x[1:m:2, 0:n:2]-alpha*post_grad[1:m:2, 0:n:2]

		likelihood, likelihood_grad = likelihood_function(x,y)
		prior, prior_grad = calculate_prior(x)
		post = beta*prior + (1-beta)*likelihood
		post_grad = beta*prior_grad + (1-beta)*likelihood_grad
		x[1:m:2, 1:n:2] = x[1:m:2, 1:n:2]-alpha*post_grad[1:m:2, 1:n:2]

		likelihood, likelihood_grad = likelihood_function(x,y)
		prior, prior_grad = calculate_prior(x)
		post = beta*prior + (1-beta)*likelihood

		if post/initial_post < 1:
			alpha *= 1.1
		else:
			alpha *= 0.5

		initial_post = post.copy()
		counter += 1
		post_values.append(post)

	if optimize_mode:
		return rrmse(denoised_img, x)
	else:
		print('After Optimization: RRMSE =', rrmse(denoised_img, x))
		plt.plot(post_values); plt.show()


	
	# import pdb; pdb.set_trace()
	




