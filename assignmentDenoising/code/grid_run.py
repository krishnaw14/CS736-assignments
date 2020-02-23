import matplotlib.pyplot as plt
import numpy as np
from denoise import denoise
from functions import *

denoised_img = plt.imread('../data/mri_image_noiseless.png')
low_noise_img = plt.imread('../data/mri_image_noise_level_low.png')
med_noise_img = plt.imread('../data/mri_image_noise_level_medium.png')
high_noise_img = plt.imread('../data/mri_image_noise_level_high.png')

rrmse_final_min = 100
beta_opt = 0
for beta in np.linspace(0.5,0.9,30):
	for gamma in np.linspace(0.001,0.05,30):
		rrmse_final = denoise(low_noise_img, denoised_img, alpha=beta, gamma=gamma, optimize_mode=True, prior='discontinuity_adaptive')
		if rrmse_final < rrmse_final_min:
			rrmse_final_min = rrmse_final
			beta_opt = beta
			gamma_opt = gamma

print('rrmse_min = {}, beta_opt = {}, gamma_opt = {}'.format(rrmse_final_min, beta_opt, gamma_opt))
import pdb; pdb.set_trace()