import numpy as np 
import os
import matplotlib.pyplot as plt
from functions import *

def calculate_posterior(x,y, beta, gamma=None, prior_func=quadratic_function):

	def calculate_prior(x):
		prior_1, grad_1 = prior_func(x - np.roll(x, 1, axis=0))
		prior_2, grad_2 = prior_func(x - np.roll(x, -1, axis=0))
		prior_3, grad_3 = prior_func(x - np.roll(x, 1, axis=1))
		prior_4, grad_4 = prior_func(x - np.roll(x, -1, axis=1))

		prior = prior_1 + prior_2 + prior_3 + prior_4
		grad = grad_1 + grad_2 + grad_3 + grad_4
		return prior, grad

	likelihood, likelihood_grad = likelihood_function(x,y)
	prior, prior_grad = calculate_prior(x)
	posterior = beta*prior + (1-beta)*likelihood
	posterior_grad = beta*prior_grad + (1-beta)*likelihood_grad

	return posterior, posterior_grad


def denoise(noisy_img, denoised_img, beta=0.6, gamma=0.5, optimize_mode=False, prior='quadratic', 
	save_results_dir='../results/'):
	
	save_results_dir = os.path.join(save_results_dir, prior)
	os.makedirs(save_results_dir, exist_ok=True)
		
	m, n = denoised_img.shape

	# Using Notation used in class
	x = noisy_img.copy() # Initial Estimate
	y = noisy_img.copy() # Observed Data

	alpha = 1e-2 # Initial Learning Rate

	initial_log_posterior, _ = calculate_posterior(x,y, beta)
	
	counter = 0	
	post_values = [initial_log_posterior]
	while counter < 300 and alpha > 1e-8:

		# Monitor Objective Function. Partitioning the grid into sets such that a set contains no neighbours.
		
		log_posterior, log_posterior_grad = calculate_posterior(x,y, beta)
		x[0:m:2, 0:n:2] = x[0:m:2, 0:n:2]-alpha*log_posterior_grad[0:m:2, 0:n:2]

		log_posterior, log_posterior_grad = calculate_posterior(x,y, beta)
		x[0:m:2, 1:n:2] = x[0:m:2, 1:n:2]-alpha*log_posterior_grad[0:m:2, 1:n:2]

		log_posterior, log_posterior_grad = calculate_posterior(x,y, beta)
		x[1:m:2, 0:n:2] = x[1:m:2, 0:n:2]-alpha*log_posterior_grad[1:m:2, 0:n:2]

		log_posterior, log_posterior_grad = calculate_posterior(x,y, beta)
		x[1:m:2, 1:n:2] = x[1:m:2, 1:n:2]-alpha*log_posterior_grad[1:m:2, 1:n:2]

		log_posterior, _ = calculate_posterior(x,y, beta)

		# Dynamic Learning Rate
		if log_posterior/initial_log_posterior < 1:
			alpha *= 1.1
		else:
			alpha *= 0.5

		initial_log_posterior = log_posterior.copy()
		counter += 1
		post_values.append(log_posterior)

	if optimize_mode:
		return rrmse(denoised_img, x)
	else:
		print('Initial RRMSE between Noisy and Denoised Image:', rrmse(denoised_img, noisy_img))
		print('Denoising done in {} iterations. RRMSE after denoising = {}'.format(counter, rrmse(denoised_img, x)))

		plt.clf()
		plt.plot(np.arange(counter+1), post_values)
		plt.xlabel('Number of iterations')
		plt.ylabel('Objective Function (log)')
		plt.title('{} prior'.format(prior))
		plt.savefig(os.path.join(save_results_dir, 'objective_function.png'))
		plt.clf()

		cmap = 'gray' if 'q1' in save_results_dir else None

		plt.imshow(noisy_img, cmap=cmap)
		plt.savefig(os.path.join(save_results_dir, 'noisy_img.png'))

		plt.imshow(denoised_img, cmap=cmap)
		plt.savefig(os.path.join(save_results_dir, 'ground_turth_denoised_img.png'))

		plt.imshow(x, cmap=cmap)
		plt.savefig(os.path.join(save_results_dir, '{}_prior_denoised_img.png'.format(prior)))




	




