import matplotlib.pyplot as plt
import numpy as np
import cv2
from denoise import denoise
from functions import *

# denoised_img = plt.imread('../data/mri_image_noiseless.png')
# low_noise_img = plt.imread('../data/mri_image_noise_level_low.png')
# med_noise_img = plt.imread('../data/mri_image_noise_level_medium.png')
# high_noise_img = plt.imread('../data/mri_image_noise_level_high.png')

denoised_img = cv2.cvtColor(plt.imread('../data/histology_noiseless.png'), cv2.COLOR_BGR2HSV)
noisy_img = cv2.cvtColor(plt.imread('../data/histology_noisy.png'), cv2.COLOR_BGR2HSV)

# rrmse_final_min = 100
# beta_opt = 0
# for beta in np.linspace(0,1,40):
# 	for gamma in np.linspace(0.001,1,40):
# 		rrmse_final = denoise(noisy_img[:,:,2], denoised_img[:,:,2], alpha=beta, gamma=gamma, optimize_mode=True, prior='discontinuity_adaptive_huber')
# 		if rrmse_final < rrmse_final_min:
# 			rrmse_final_min = rrmse_final
# 			beta_opt = beta
# 			gamma_opt = gamma

# rrmse_init_channel = rrmse(denoised_img[:,:,2], noisy_img[:,:,2])
# rrmse_init_total = rrmse(denoised_img, noisy_img)

# print('rrmse_min = {}, beta_opt = {}, gamma_opt = {}'.format(rrmse_final_min, beta_opt, gamma_opt))

channel_0 = denoise(noisy_img[:,:,0], denoised_img[:,:,0], alpha=1, gamma=1, optimize_mode=False, prior='discontinuity_adaptive_huber', 
	color_mode=True)

channel_1 = denoise(noisy_img[:,:,1], denoised_img[:,:,1], alpha=0.8974, gamma=0.02662, optimize_mode=False, prior='discontinuity_adaptive_huber', 
	color_mode=True)

channel_2 = denoise(noisy_img[:,:,2], denoised_img[:,:,2], alpha=0.8718, gamma=0.02662, optimize_mode=False, prior='discontinuity_adaptive_huber', 
	color_mode=True)

sample = noisy_img.copy()
sample[:,:,0] = channel_0
sample[:,:,1] = channel_1
sample[:,:,2] = channel_2

plt.imshow( noisy_img); plt.savefig('noisy.png'); plt.clf()
plt.imshow( denoised_img); plt.savefig('noiseless.png'); plt.clf()
plt.imshow( sample); plt.savefig('sample.png'); plt.clf()

import pdb; pdb.set_trace()