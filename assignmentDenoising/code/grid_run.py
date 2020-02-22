import matplotlib.pyplot as plt
import numpy as np
from denoise import denoise

denoised_img = plt.imread('../data/mri_image_noiseless.png')
low_noise_img = plt.imread('../data/mri_image_noise_level_low.png')
med_noise_img = plt.imread('../data/mri_image_noise_level_medium.png')
high_noise_img = plt.imread('../data/mri_image_noise_level_high.png')

rrmse_final_min = 100
beta_opt = 0
for beta in np.linspace(0.22,0.27,100):
	# optimize_mode=True if beta ==0.22 else False
	rrmse_final = denoise(high_noise_img, denoised_img, beta=beta,optimize_mode=True, prior='quadratic')
	if rrmse_final < rrmse_final_min:
		rrmse_final_min = rrmse_final
		beta_opt = beta

import pdb; pdb.set_trace()