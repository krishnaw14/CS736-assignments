import argparse
import os
import cv2
import matplotlib.pyplot as plt
from denoise import denoise

# Solve For question 1: Brain MRI images

save_results_dir = '../results/q1'

denoised_img = plt.imread('../data/mri_image_noiseless.png')
low_noise_img = plt.imread('../data/mri_image_noise_level_low.png')
med_noise_img = plt.imread('../data/mri_image_noise_level_medium.png')
high_noise_img = plt.imread('../data/mri_image_noise_level_high.png')

# Quadratic Prior
print('\n---------------------------------------------------------------------------------')
print('Denoising Low Noise Level Image...')
denoise(low_noise_img, denoised_img, beta=0.0875, optimize_mode=False, prior='quadratic', 
	save_results_dir=os.path.join(save_results_dir, 'quadratic', 'low_noise_level'))
print('RRMSE at 1.2 times optimum beta=', round(denoise(low_noise_img, denoised_img, beta=0.0875*1.2, optimize_mode=True, prior='quadratic', 
	save_results_dir=os.path.join(save_results_dir, 'quadratic', 'low_noise_level')), 5))
print('RRMSE at 0.8 times optimum beta=', round(denoise(low_noise_img, denoised_img, beta=0.0875*0.8, optimize_mode=True, prior='quadratic', 
	save_results_dir=os.path.join(save_results_dir, 'quadratic', 'low_noise_level')), 5))
print('---------------------------------------------------------------------------------\n')

print('\n---------------------------------------------------------------------------------')
print('Denoising Medium Noise Level Image...')
denoise(med_noise_img, denoised_img, beta=0.1773, optimize_mode=False, prior='quadratic', 
	save_results_dir=os.path.join(save_results_dir, 'quadratic', 'med_noise_level'))
print('RRMSE at 1.2 times optimum beta=', round(denoise(med_noise_img, denoised_img, beta=0.1773*1.2, optimize_mode=True, prior='quadratic', 
	save_results_dir=os.path.join(save_results_dir, 'quadratic', 'low_noise_level')), 5))
print('RRMSE at 0.8 times optimum beta=', round(denoise(med_noise_img, denoised_img, beta=0.1773*0.8, optimize_mode=True, prior='quadratic', 
	save_results_dir=os.path.join(save_results_dir, 'quadratic', 'low_noise_level')), 5))
print('---------------------------------------------------------------------------------\n')

print('\n---------------------------------------------------------------------------------')
print('Denoising High Noise Level Image...')
denoise(high_noise_img, denoised_img, beta=0.2407, optimize_mode=False, prior='quadratic', 
	save_results_dir=os.path.join(save_results_dir, 'quadratic', 'high_noise_level'))
print('RRMSE at 1.2 times optimum beta=', round(denoise(high_noise_img, denoised_img, beta=0.2407*1.2, optimize_mode=True, prior='quadratic', 
	save_results_dir=os.path.join(save_results_dir, 'quadratic', 'low_noise_level')), 5))
print('RRMSE at 0.8 times optimum beta=', round(denoise(high_noise_img, denoised_img, beta=0.2407*0.8, optimize_mode=True, prior='quadratic', 
	save_results_dir=os.path.join(save_results_dir, 'quadratic', 'low_noise_level')), 5))
print('---------------------------------------------------------------------------------\n')




# Discontinuity Adaptive Huber Prior
# print('\n---------------------------------------------------------------------------------')
# print('Denoising Low Noise Level Image...')
# denoise(low_noise_img, denoised_img, beta=0.0875, optimize_mode=False, prior='discontinuity_adaptive_huber', 
# 	save_results_dir=os.path.join(save_results_dir, 'discontinuity_adaptive_huber', 'low_noise_level'))
# print('---------------------------------------------------------------------------------\n')

# print('\n---------------------------------------------------------------------------------')
# print('Denoising Medium Noise Level Image...')
# denoise(med_noise_img, denoised_img, beta=0.1773, optimize_mode=False, prior='discontinuity_adaptive_huber', 
# 	save_results_dir=os.path.join(save_results_dir, 'discontinuity_adaptive_huber', 'med_noise_level'))
# print('---------------------------------------------------------------------------------\n')

# print('\n---------------------------------------------------------------------------------')
# print('Denoising High Noise Level Image...')
# denoise(high_noise_img, denoised_img, beta=0.2407, optimize_mode=False, prior='discontinuity_adaptive_huber', 
# 	save_results_dir=os.path.join(save_results_dir, 'discontinuity_adaptive_huber', 'high_noise_level'))
# print('---------------------------------------------------------------------------------\n')

print('\n---------------------------------------------------------------------------------')
print('Denoising High Noise Level Image...')
denoise(high_noise_img, denoised_img, beta=0.9449, gamma=0.003, optimize_mode=False, prior='discontinuity_adaptive', 
	save_results_dir=os.path.join(save_results_dir, 'discontinuity_adaptive_huber', 'high_noise_level'))
print('RRMSE at 1.2 times optimum beta=', round(denoise(high_noise_img, denoised_img, beta=0.9449*1.2, gamma=0.003, optimize_mode=True, prior='discontinuity_adaptive', 
	save_results_dir=os.path.join(save_results_dir, 'discontinuity_adaptive', 'low_noise_level')), 5))
print('RRMSE at 0.8 times optimum beta=', round(denoise(high_noise_img, denoised_img, beta=0.9449*0.8, gamma=0.003, optimize_mode=True, prior='discontinuity_adaptive', 
	save_results_dir=os.path.join(save_results_dir, 'discontinuity_adaptive', 'low_noise_level')), 5))
print('RRMSE at 1.2 times optimum gamma=', round(denoise(high_noise_img, denoised_img, beta=0.9449, gamma=0.003*1.3, optimize_mode=True, prior='discontinuity_adaptive', 
	save_results_dir=os.path.join(save_results_dir, 'discontinuity_adaptive', 'low_noise_level')), 5))
print('RRMSE at 0.8 times optimum gamma=', round(denoise(high_noise_img, denoised_img, beta=0.9449, gamma=0.003*0.8, optimize_mode=True, prior='discontinuity_adaptive', 
	save_results_dir=os.path.join(save_results_dir, 'discontinuity_adaptive', 'low_noise_level')), 5))
print('---------------------------------------------------------------------------------\n')


# Solve For question 2: Histology Images

denoised_img = cv2.imread('../data/histology_noiseless.png')
denoised_img = cv2.normalize(denoised_img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
noisy_img = cv2.imread('../data/histology_noisy.png')
noisy_img = cv2.normalize(denoised_img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

# print('Solving for Histology Image')
# denoise(noisy_img, denoised_img)




