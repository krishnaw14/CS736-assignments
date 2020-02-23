import matplotlib.pyplot as plt
import numpy as np
from denoise import denoise
import sys 

gamma_upper_bound = 1
gamma_lower_bound = 0.00001
beta_upper_bound = 1
beta_lower_bound = 0.00001
image_type = sys.argv[1]
function_type = sys.argv[2]

denoised_img = plt.imread('../data/mri_image_noiseless.png')
low = plt.imread('../data/mri_image_noise_level_low.png')
med = plt.imread('../data/mri_image_noise_level_medium.png')
high = plt.imread('../data/mri_image_noise_level_high.png')

if image_type == "low":
	image = low
if image_type == "med":
	image = med 
if image_type == "high":
	image = high

rrmse_final_min = 100
beta_opt = 0
gamma_opt = 0
for i in range(10):
	for gamma in np.linspace(gamma_lower_bound,gamma_upper_bound,10):
		for beta in np.linspace(beta_lower_bound,beta_upper_bound,10):

			rrmse_final = denoise(image, denoised_img, beta=beta,gamma = gamma, optimize_mode=True, prior=function_type)
			if rrmse_final < rrmse_final_min:
				rrmse_final_min = rrmse_final
				beta_opt = beta
				gamma_opt = gamma
	gamma_lower_bound = max(gamma_opt - gamma_opt/2, 0.00001) 
	gamma_upper_bound = min(gamma_opt + gamma_opt/2, 1.0)
	
	beta_upper_bound = min(beta_opt + beta_opt/2, 1.0)
	beta_lower_bound = max(beta_opt - beta_opt/2, 0.00001)

print (beta_opt)
print (gamma_opt)
print (image_type)
print (function_type)
import pdb; pdb.set_trace()