import argparse
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from denoise import denoise
from functions import *

# Solve For question 1: Brain MRI images

save_results_dir = '../results/q1'

denoised_img = plt.imread('../data/mri_image_noiseless.png')
low_noise_img = plt.imread('../data/mri_image_noise_level_low.png')
med_noise_img = plt.imread('../data/mri_image_noise_level_medium.png')
high_noise_img = plt.imread('../data/mri_image_noise_level_high.png')

print('QUADRATIC PRIOR')

print('\n---------------------------------------------------------------------------------')
print('Denoising Low Noise Level Image...')
denoise(low_noise_img, denoised_img, alpha=0.0875, optimize_mode=False, prior='quadratic', 
	save_results_dir=os.path.join(save_results_dir, 'quadratic', 'low_noise_level'))
print('RRMSE at 1.2 times optimum alpha=', round(denoise(low_noise_img, denoised_img, alpha=0.0875*1.2, optimize_mode=True, prior='quadratic', 
	save_results_dir=os.path.join(save_results_dir, 'quadratic', 'low_noise_level')), 5))
print('RRMSE at 0.8 times optimum alpha=', round(denoise(low_noise_img, denoised_img, alpha=0.0875*0.8, optimize_mode=True, prior='quadratic', 
	save_results_dir=os.path.join(save_results_dir, 'quadratic', 'low_noise_level')), 5))
print('---------------------------------------------------------------------------------\n')

print('\n---------------------------------------------------------------------------------')
print('Denoising Medium Noise Level Image...')
denoise(med_noise_img, denoised_img, alpha=0.1773, optimize_mode=False, prior='quadratic', 
	save_results_dir=os.path.join(save_results_dir, 'quadratic', 'med_noise_level'))
print('RRMSE at 1.2 times optimum alpha=', round(denoise(med_noise_img, denoised_img, alpha=0.1773*1.2, optimize_mode=True, prior='quadratic', 
	save_results_dir=os.path.join(save_results_dir, 'quadratic', 'low_noise_level')), 5))
print('RRMSE at 0.8 times optimum alpha=', round(denoise(med_noise_img, denoised_img, alpha=0.1773*0.8, optimize_mode=True, prior='quadratic', 
	save_results_dir=os.path.join(save_results_dir, 'quadratic', 'low_noise_level')), 5))
print('---------------------------------------------------------------------------------\n')

print('\n---------------------------------------------------------------------------------')
print('Denoising High Noise Level Image...')
denoise(high_noise_img, denoised_img, alpha=0.2407, optimize_mode=False, prior='quadratic', 
	save_results_dir=os.path.join(save_results_dir, 'quadratic', 'high_noise_level'))
print('RRMSE at 1.2 times optimum alpha=', round(denoise(high_noise_img, denoised_img, alpha=0.2407*1.2, optimize_mode=True, prior='quadratic', 
	save_results_dir=os.path.join(save_results_dir, 'quadratic', 'low_noise_level')), 5))
print('RRMSE at 0.8 times optimum alpha=', round(denoise(high_noise_img, denoised_img, alpha=0.2407*0.8, optimize_mode=True, prior='quadratic', 
	save_results_dir=os.path.join(save_results_dir, 'quadratic', 'low_noise_level')), 5))
print('---------------------------------------------------------------------------------\n')


print('DISCONTINUITY ADAPTIVE PRIOR')

print('\n---------------------------------------------------------------------------------')
print('Denoising Low Noise Level Image...')
denoise(low_noise_img, denoised_img, alpha=0.8748, gamma=0.0023, optimize_mode=False, prior='discontinuity_adaptive', 
	save_results_dir=os.path.join(save_results_dir, 'discontinuity_adaptive', 'low_noise_level'))
print('RRMSE at 1.2 times optimum alpha=', round(denoise(high_noise_img, denoised_img, alpha=0.8748*1.2, gamma=0.0023, optimize_mode=True, prior='discontinuity_adaptive', 
	save_results_dir=os.path.join(save_results_dir, 'discontinuity_adaptive', 'low_noise_level')), 5))
print('RRMSE at 0.8 times optimum alpha=', round(denoise(high_noise_img, denoised_img, alpha=0.8748*0.8, gamma=0.0023, optimize_mode=True, prior='discontinuity_adaptive', 
	save_results_dir=os.path.join(save_results_dir, 'discontinuity_adaptive', 'low_noise_level')), 5))
print('RRMSE at 1.2 times optimum gamma=', round(denoise(high_noise_img, denoised_img, alpha=0.8748, gamma=0.0023*1.2, optimize_mode=True, prior='discontinuity_adaptive', 
	save_results_dir=os.path.join(save_results_dir, 'discontinuity_adaptive', 'low_noise_level')), 5))
print('RRMSE at 0.8 times optimum gamma=', round(denoise(high_noise_img, denoised_img, alpha=0.8748, gamma=0.0023*0.8, optimize_mode=True, prior='discontinuity_adaptive', 
	save_results_dir=os.path.join(save_results_dir, 'discontinuity_adaptive', 'low_noise_level')), 5))
print('---------------------------------------------------------------------------------\n')

print('\n---------------------------------------------------------------------------------')
print('Denoising Medium Noise Level Image...')
denoise(med_noise_img, denoised_img, alpha=0.8862, gamma=0.00438, optimize_mode=False, prior='discontinuity_adaptive', 
	save_results_dir=os.path.join(save_results_dir, 'discontinuity_adaptive', 'med_noise_level'))
print('RRMSE at 1.2 times optimum alpha=', round(denoise(high_noise_img, denoised_img, alpha=0.8862*1.2, gamma=0.00438, optimize_mode=True, prior='discontinuity_adaptive', 
	save_results_dir=os.path.join(save_results_dir, 'discontinuity_adaptive', 'med_noise_level')), 5))
print('RRMSE at 0.8 times optimum alpha=', round(denoise(high_noise_img, denoised_img, alpha=0.8862*0.8, gamma=0.00438, optimize_mode=True, prior='discontinuity_adaptive', 
	save_results_dir=os.path.join(save_results_dir, 'discontinuity_adaptive', 'med_noise_level')), 5))
print('RRMSE at 1.2 times optimum gamma=', round(denoise(high_noise_img, denoised_img, alpha=0.8862, gamma=0.00438*1.2, optimize_mode=True, prior='discontinuity_adaptive', 
	save_results_dir=os.path.join(save_results_dir, 'discontinuity_adaptive', 'med_noise_level')), 5))
print('RRMSE at 0.8 times optimum gamma=', round(denoise(high_noise_img, denoised_img, alpha=0.8862, gamma=0.00438*0.8, optimize_mode=True, prior='discontinuity_adaptive', 
	save_results_dir=os.path.join(save_results_dir, 'discontinuity_adaptive', 'med_noise_level')), 5))
print('---------------------------------------------------------------------------------\n')

print('\n---------------------------------------------------------------------------------')
print('Denoising High Noise Level Image...')
denoise(high_noise_img, denoised_img, alpha=0.8738, gamma=0.008, optimize_mode=False, prior='discontinuity_adaptive', 
	save_results_dir=os.path.join(save_results_dir, 'discontinuity_adaptive', 'high_noise_level'))
print('RRMSE at 1.2 times optimum alpha=', round(denoise(high_noise_img, denoised_img, alpha=0.8738*1.2, gamma=0.008, optimize_mode=True, prior='discontinuity_adaptive', 
	save_results_dir=os.path.join(save_results_dir, 'discontinuity_adaptive', 'high_noise_level')), 5))
print('RRMSE at 0.8 times optimum alpha=', round(denoise(high_noise_img, denoised_img, alpha=0.8738*0.8, gamma=0.008, optimize_mode=True, prior='discontinuity_adaptive', 
	save_results_dir=os.path.join(save_results_dir, 'discontinuity_adaptive', 'high_noise_level')), 5))
print('RRMSE at 1.2 times optimum gamma=', round(denoise(high_noise_img, denoised_img, alpha=0.8738, gamma=0.008*1.2, optimize_mode=True, prior='discontinuity_adaptive', 
	save_results_dir=os.path.join(save_results_dir, 'discontinuity_adaptive', 'high_noise_level')), 5))
print('RRMSE at 0.8 times optimum gamma=', round(denoise(high_noise_img, denoised_img, alpha=0.8738, gamma=0.008*0.8, optimize_mode=True, prior='discontinuity_adaptive', 
	save_results_dir=os.path.join(save_results_dir, 'discontinuity_adaptive', 'high_noise_level')), 5))
print('---------------------------------------------------------------------------------\n')


print('DISCONTINUITY ADAPTIVE HUBER PRIOR')

print('\n---------------------------------------------------------------------------------')
print('Denoising Low Noise Level Image...')
denoise(low_noise_img, denoised_img, alpha=0.8748, gamma=0.00211, optimize_mode=False, prior='discontinuity_adaptive_huber', 
	save_results_dir=os.path.join(save_results_dir, 'discontinuity_adaptive_huber', 'low_noise_level'))
print('RRMSE at 1.2 times optimum alpha=', round(denoise(high_noise_img, denoised_img, alpha=0.8748*1.2, gamma=0.00211, optimize_mode=True, prior='discontinuity_adaptive_huber', 
	save_results_dir=os.path.join(save_results_dir, 'discontinuity_adaptive_huber', 'low_noise_level')), 5))
print('RRMSE at 0.8 times optimum alpha=', round(denoise(high_noise_img, denoised_img, alpha=0.8748*0.8, gamma=0.00211, optimize_mode=True, prior='discontinuity_adaptive_huber', 
	save_results_dir=os.path.join(save_results_dir, 'discontinuity_adaptive_huber', 'low_noise_level')), 5))
print('RRMSE at 1.2 times optimum gamma=', round(denoise(high_noise_img, denoised_img, alpha=0.8748, gamma=0.00211*1.2, optimize_mode=True, prior='discontinuity_adaptive_huber', 
	save_results_dir=os.path.join(save_results_dir, 'discontinuity_adaptive_huber', 'low_noise_level')), 5))
print('RRMSE at 0.8 times optimum gamma=', round(denoise(high_noise_img, denoised_img, alpha=0.8748, gamma=0.00211*0.8, optimize_mode=True, prior='discontinuity_adaptive_huber', 
	save_results_dir=os.path.join(save_results_dir, 'discontinuity_adaptive_huber', 'low_noise_level')), 5))
print('---------------------------------------------------------------------------------\n')

print('\n---------------------------------------------------------------------------------')
print('Denoising Medium Noise Level Image...')
denoise(med_noise_img, denoised_img, alpha=0.8663, gamma=0.0048, optimize_mode=False, prior='discontinuity_adaptive_huber', 
	save_results_dir=os.path.join(save_results_dir, 'discontinuity_adaptive_huber', 'med_noise_level'))
print('RRMSE at 1.2 times optimum alpha=', round(denoise(high_noise_img, denoised_img, alpha=0.8663*1.2, gamma=0.0048, optimize_mode=True, prior='discontinuity_adaptive_huber', 
	save_results_dir=os.path.join(save_results_dir, 'discontinuity_adaptive_huber', 'med_noise_level')), 5))
print('RRMSE at 0.8 times optimum alpha=', round(denoise(high_noise_img, denoised_img, alpha=0.8663*0.8, gamma=0.0048, optimize_mode=True, prior='discontinuity_adaptive_huber', 
	save_results_dir=os.path.join(save_results_dir, 'discontinuity_adaptive_huber', 'med_noise_level')), 5))
print('RRMSE at 1.2 times optimum gamma=', round(denoise(high_noise_img, denoised_img, alpha=0.8663, gamma=0.0048*1.2, optimize_mode=True, prior='discontinuity_adaptive_huber', 
	save_results_dir=os.path.join(save_results_dir, 'discontinuity_adaptive_huber', 'med_noise_level')), 5))
print('RRMSE at 0.8 times optimum gamma=', round(denoise(high_noise_img, denoised_img, alpha=0.8663, gamma=0.0048*0.8, optimize_mode=True, prior='discontinuity_adaptive_huber', 
	save_results_dir=os.path.join(save_results_dir, 'discontinuity_adaptive_huber', 'med_noise_level')), 5))
print('---------------------------------------------------------------------------------\n')


print('\n---------------------------------------------------------------------------------')
print('Denoising High Noise Level Image...')
denoise(high_noise_img, denoised_img, alpha=0.7729, gamma=0.01389, optimize_mode=False, prior='discontinuity_adaptive_huber', 
	save_results_dir=os.path.join(save_results_dir, 'discontinuity_adaptive_huber', 'high_noise_level'))
print('RRMSE at 1.2 times optimum alpha=', round(denoise(high_noise_img, denoised_img, alpha=0.7729*1.2, gamma=0.01389, optimize_mode=True, prior='discontinuity_adaptive_huber', 
	save_results_dir=os.path.join(save_results_dir, 'discontinuity_adaptive_huber', 'high_noise_level')), 5))
print('RRMSE at 0.8 times optimum alpha=', round(denoise(high_noise_img, denoised_img, alpha=0.7729*0.8, gamma=0.01389, optimize_mode=True, prior='discontinuity_adaptive_huber', 
	save_results_dir=os.path.join(save_results_dir, 'discontinuity_adaptive_huber', 'high_noise_level')), 5))
print('RRMSE at 1.2 times optimum gamma=', round(denoise(high_noise_img, denoised_img, alpha=0.7729, gamma=0.01389*1.2, optimize_mode=True, prior='discontinuity_adaptive_huber', 
	save_results_dir=os.path.join(save_results_dir, 'discontinuity_adaptive_huber', 'high_noise_level')), 5))
print('RRMSE at 0.8 times optimum gamma=', round(denoise(high_noise_img, denoised_img, alpha=0.7729, gamma=0.01389*0.8, optimize_mode=True, prior='discontinuity_adaptive_huber', 
	save_results_dir=os.path.join(save_results_dir, 'discontinuity_adaptive_huber', 'high_noise_level')), 5))
print('---------------------------------------------------------------------------------\n')



# Solve For question 2: Histology Images

noiseless_img = cv2.cvtColor(plt.imread('../data/histology_noiseless.png'), cv2.COLOR_BGR2HSV)
noisy_img = cv2.cvtColor(plt.imread('../data/histology_noisy.png'), cv2.COLOR_BGR2HSV)

print('\n Solving Question 2. Original RMSE:', rrmse(noiseless_img, noisy_img))

print('\nQuestion 2: QUADRATIC PRIOR')
save_results_dir = '../results/q2/'

print('\n---------------------------------------------------------------------------------')
print('Denoising H Channel...')
channel_0, post_values_0 = denoise(noisy_img[:,:,0], noiseless_img[:,:,0], alpha=0.9919, gamma=0, optimize_mode=False, prior='quadratic', 
	save_results_dir=os.path.join(save_results_dir, 'quadratic', 'H'), color_mode=True)
print('RRMSE at 1.2 times optimum alpha=', round(denoise(noisy_img[:,:,0], noiseless_img[:,:,0], alpha=0.9919*1.2, gamma=1, optimize_mode=True, prior='quadratic', 
	color_mode=True), 5))
print('RRMSE at 0.8 times optimum alpha=', round(denoise(noisy_img[:,:,0], noiseless_img[:,:,0], alpha=0.9919*0.8, gamma=1, optimize_mode=True, prior='quadratic', 
	color_mode=True), 5))
print('---------------------------------------------------------------------------------\n')

print('\n---------------------------------------------------------------------------------')
print('Denoising S Channel...')
channel_1, post_values_1 = denoise(noisy_img[:,:,1], noiseless_img[:,:,1], alpha=0.75275, gamma=0, optimize_mode=False, prior='quadratic', 
	save_results_dir=os.path.join(save_results_dir, 'quadratic', 'S'), color_mode=True)
print('RRMSE at 1.2 times optimum alpha=', round(denoise(noisy_img[:,:,1], noiseless_img[:,:,1], alpha=0.75275*1.2, gamma=1, optimize_mode=True, prior='quadratic', 
	color_mode=True), 5))
print('RRMSE at 0.8 times optimum alpha=', round(denoise(noisy_img[:,:,1], noiseless_img[:,:,1], alpha=0.75275*0.8, gamma=1, optimize_mode=True, prior='quadratic', 
	color_mode=True), 5))
print('---------------------------------------------------------------------------------\n')

print('\n---------------------------------------------------------------------------------')
print('Denoising V Channel...')
channel_2, post_values_2 = denoise(noisy_img[:,:,2], noiseless_img[:,:,2], alpha=0.7137, gamma=0, optimize_mode=False, prior='quadratic', 
	save_results_dir=os.path.join(save_results_dir, 'quadratic', 'V'), color_mode=True)
print('RRMSE at 1.2 times optimum alpha=', round(denoise(noisy_img[:,:,2], noiseless_img[:,:,2], alpha=0.7137*1.2, gamma=1, optimize_mode=True, prior='quadratic', 
	color_mode=True), 5))
print('RRMSE at 0.8 times optimum alpha=', round(denoise(noisy_img[:,:,2], noiseless_img[:,:,2], alpha=0.7137*0.8, gamma=1, optimize_mode=True, prior='quadratic', 
	color_mode=True), 5))
print('---------------------------------------------------------------------------------\n')


denoised_img = np.zeros_like(noisy_img)
denoised_img[:,:,0] = channel_0 
denoised_img[:,:,1] = channel_1
denoised_img[:,:,2] = channel_2

plt.imshow(noisy_img)
plt.savefig(os.path.join(save_results_dir, 'noisy_img.png'))
plt.imshow(noiseless_img)
plt.savefig(os.path.join(save_results_dir, 'ground_turth_denoised_img.png'))

plt.imshow(denoised_img)
plt.savefig(os.path.join(save_results_dir, 'quadratic_denoised_img.png'))
plt.clf()
plt.plot(post_values_0, label='H Channel')
plt.plot(post_values_1, label='S Channel')
plt.plot(post_values_2, label='V Channel')
plt.legend()
plt.xlabel('Number of iterations')
plt.ylabel('Objective Function (log)')
plt.title('Quadratic prior')
plt.savefig(os.path.join(save_results_dir, 'quadratic_all_channel_objective_function.png'))

print('\nRMSE after quadratic prior denoising:', rrmse(noiseless_img, denoised_img))

print('\nQuestion 2: HUBER PRIOR')

print('\n---------------------------------------------------------------------------------')
print('Denoising H Channel...')
channel_0, post_values_0 = denoise(noisy_img[:,:,0], noiseless_img[:,:,0], alpha=1, gamma=1, optimize_mode=False, prior='discontinuity_adaptive_huber', 
	save_results_dir=os.path.join(save_results_dir, 'huber', 'H'), color_mode=True)
print('RRMSE at 1.2 times optimum alpha=', round(denoise(noisy_img[:,:,0], noiseless_img[:,:,0], alpha=1*1.2, gamma=1, optimize_mode=True, prior='discontinuity_adaptive_huber', 
	color_mode=True), 5))
print('RRMSE at 0.8 times optimum alpha=', round(denoise(noisy_img[:,:,0], noiseless_img[:,:,0], alpha=1*0.8, gamma=1, optimize_mode=True, prior='discontinuity_adaptive_huber', 
	color_mode=True), 5))
print('RRMSE at 1.2 times optimum gamma=', round(denoise(noisy_img[:,:,0], noiseless_img[:,:,0], alpha=1, gamma=1*1.2, optimize_mode=True, prior='discontinuity_adaptive_huber', 
	color_mode=True), 5))
print('RRMSE at 0.8 times optimum gamma=', round(denoise(noisy_img[:,:,0], noiseless_img[:,:,0], alpha=1, gamma=1*0.8, optimize_mode=True, prior='discontinuity_adaptive_huber', 
	color_mode=True), 5))
print('---------------------------------------------------------------------------------\n')

print('\n---------------------------------------------------------------------------------')
print('Denoising S Channel...')
channel_1, post_values_1 = denoise(noisy_img[:,:,1], noiseless_img[:,:,1], alpha=0.8974, gamma=0.02662, optimize_mode=False, prior='discontinuity_adaptive_huber', 
	save_results_dir=os.path.join(save_results_dir, 'huber', 'S'), color_mode=True)
print('RRMSE at 1.2 times optimum alpha=', round(denoise(noisy_img[:,:,1], noiseless_img[:,:,1], alpha=0.8974*1.2, gamma=0.02662, optimize_mode=True, prior='discontinuity_adaptive_huber', 
	color_mode=True), 5))
print('RRMSE at 0.8 times optimum alpha=', round(denoise(noisy_img[:,:,1], noiseless_img[:,:,1], alpha=0.8974*0.8, gamma=0.02662, optimize_mode=True, prior='discontinuity_adaptive_huber', 
	color_mode=True), 5))
print('RRMSE at 1.2 times optimum gamma=', round(denoise(noisy_img[:,:,1], noiseless_img[:,:,1], alpha=0.8974, gamma=0.02662*1.2, optimize_mode=True, prior='discontinuity_adaptive_huber', 
	color_mode=True), 5))
print('RRMSE at 0.8 times optimum gamma=', round(denoise(noisy_img[:,:,1], noiseless_img[:,:,1], alpha=0.8974, gamma=0.02662*0.8, optimize_mode=True, prior='discontinuity_adaptive_huber', 
	color_mode=True), 5))
print('---------------------------------------------------------------------------------\n')

print('\n---------------------------------------------------------------------------------')
print('Denoising V Channel...')
channel_2, post_values_2 = denoise(noisy_img[:,:,2], noiseless_img[:,:,2], alpha=0.8718, gamma=0.02662, optimize_mode=False, prior='discontinuity_adaptive_huber', 
	save_results_dir=os.path.join(save_results_dir, 'huber', 'V'), color_mode=True)
print('RRMSE at 1.2 times optimum alpha=', round(denoise(noisy_img[:,:,2], noiseless_img[:,:,2], alpha=0.8718*1.2, gamma=0.02662, optimize_mode=True, prior='discontinuity_adaptive_huber', 
	color_mode=True), 5))
print('RRMSE at 0.8 times optimum alpha=', round(denoise(noisy_img[:,:,2], noiseless_img[:,:,2], alpha=0.8718*0.8, gamma=0.02662, optimize_mode=True, prior='discontinuity_adaptive_huber', 
	color_mode=True), 5))
print('RRMSE at 1.2 times optimum gamma=', round(denoise(noisy_img[:,:,2], noiseless_img[:,:,2], alpha=0.8718, gamma=0.02662*1.2, optimize_mode=True, prior='discontinuity_adaptive_huber', 
	color_mode=True), 5))
print('RRMSE at 0.8 times optimum gamma=', round(denoise(noisy_img[:,:,2], noiseless_img[:,:,2], alpha=0.8718, gamma=0.02662*0.8, optimize_mode=True, prior='discontinuity_adaptive_huber', 
	color_mode=True), 5))
print('---------------------------------------------------------------------------------\n')


denoised_img = np.zeros_like(noisy_img)
denoised_img[:,:,0] = channel_0 
denoised_img[:,:,1] = channel_1
denoised_img[:,:,2] = channel_2

plt.imshow(denoised_img)
plt.savefig(os.path.join(save_results_dir, 'huber_denoised_img.png'))
plt.clf()
plt.plot(post_values_0, label='H Channel')
plt.plot(post_values_1, label='S Channel')
plt.plot(post_values_2, label='V Channel')
plt.legend()
plt.xlabel('Number of iterations')
plt.ylabel('Objective Function (log)')
plt.title('Discontinuity Adaptive Huber prior')
plt.savefig(os.path.join(save_results_dir, 'huber_all_channel_objective_function.png'))

print('\nRMSE after huber prior denoising:', rrmse(noiseless_img, denoised_img))


print('Question 2: DISCONTINUITY ADAPTIVE PRIOR')

print('\n---------------------------------------------------------------------------------')
print('Denoising H Channel...')
channel_0, post_values_0 = denoise(noisy_img[:,:,0], noiseless_img[:,:,0], alpha=1, gamma=1, optimize_mode=False, prior='discontinuity_adaptive', 
	save_results_dir=os.path.join(save_results_dir, 'adaptive', 'H'), color_mode=True)
print('RRMSE at 1.2 times optimum alpha=', round(denoise(noisy_img[:,:,0], noiseless_img[:,:,0], alpha=1*1.2, gamma=1, optimize_mode=True, prior='discontinuity_adaptive', 
	color_mode=True), 5))
print('RRMSE at 0.8 times optimum alpha=', round(denoise(noisy_img[:,:,0], noiseless_img[:,:,0], alpha=1*0.8, gamma=1, optimize_mode=True, prior='discontinuity_adaptive', 
	color_mode=True), 5))
print('RRMSE at 1.2 times optimum gamma=', round(denoise(noisy_img[:,:,0], noiseless_img[:,:,0], alpha=1, gamma=1*1.2, optimize_mode=True, prior='discontinuity_adaptive', 
	color_mode=True), 5))
print('RRMSE at 0.8 times optimum gamma=', round(denoise(noisy_img[:,:,0], noiseless_img[:,:,0], alpha=1, gamma=1*0.8, optimize_mode=True, prior='discontinuity_adaptive', 
	color_mode=True), 5))
print('---------------------------------------------------------------------------------\n')

print('\n---------------------------------------------------------------------------------')
print('Denoising S Channel...')
channel_1, post_values_1 = denoise(noisy_img[:,:,1], noiseless_img[:,:,1], alpha=0.93878, gamma=0.02139, optimize_mode=False, prior='discontinuity_adaptive', 
	save_results_dir=os.path.join(save_results_dir, 'adaptive', 'S'), color_mode=True)
print('RRMSE at 1.2 times optimum alpha=', round(denoise(noisy_img[:,:,1], noiseless_img[:,:,1], alpha=0.93878*1.2, gamma=0.02139, optimize_mode=True, prior='discontinuity_adaptive', 
	color_mode=True), 5))
print('RRMSE at 0.8 times optimum alpha=', round(denoise(noisy_img[:,:,1], noiseless_img[:,:,1], alpha=0.93878*0.8, gamma=0.02139, optimize_mode=True, prior='discontinuity_adaptive', 
	color_mode=True), 5))
print('RRMSE at 1.2 times optimum gamma=', round(denoise(noisy_img[:,:,1], noiseless_img[:,:,1], alpha=0.93878, gamma=0.02139*1.2, optimize_mode=True, prior='discontinuity_adaptive', 
	color_mode=True), 5))
print('RRMSE at 0.8 times optimum gamma=', round(denoise(noisy_img[:,:,1], noiseless_img[:,:,1], alpha=0.93878, gamma=0.02139*0.8, optimize_mode=True, prior='discontinuity_adaptive', 
	color_mode=True), 5))
print('---------------------------------------------------------------------------------\n')

print('\n---------------------------------------------------------------------------------')
print('Denoising V Channel...')
channel_2, post_values_2 = denoise(noisy_img[:,:,2], noiseless_img[:,:,2], alpha=0.9184, gamma=0.02138, optimize_mode=False, prior='discontinuity_adaptive', 
	save_results_dir=os.path.join(save_results_dir, 'adaptive', 'V'), color_mode=True)
print('RRMSE at 1.2 times optimum alpha=', round(denoise(noisy_img[:,:,2], noiseless_img[:,:,2], alpha=0.9184*1.2, gamma=0.02138, optimize_mode=True, prior='discontinuity_adaptive', 
	color_mode=True), 5))
print('RRMSE at 0.8 times optimum alpha=', round(denoise(noisy_img[:,:,2], noiseless_img[:,:,2], alpha=0.9184*0.8, gamma=0.02138, optimize_mode=True, prior='discontinuity_adaptive', 
	color_mode=True), 5))
print('RRMSE at 1.2 times optimum gamma=', round(denoise(noisy_img[:,:,2], noiseless_img[:,:,2], alpha=0.9184, gamma=0.02138*1.2, optimize_mode=True, prior='discontinuity_adaptive', 
	color_mode=True), 5))
print('RRMSE at 0.8 times optimum gamma=', round(denoise(noisy_img[:,:,2], noiseless_img[:,:,2], alpha=0.9184, gamma=0.02138*0.8, optimize_mode=True, prior='discontinuity_adaptive', 
	color_mode=True), 5))
print('---------------------------------------------------------------------------------\n')


denoised_img = np.zeros_like(noisy_img)
denoised_img[:,:,0] = channel_0 
denoised_img[:,:,1] = channel_1
denoised_img[:,:,2] = channel_2

plt.imshow(denoised_img)
plt.savefig(os.path.join(save_results_dir, 'discontinuity_adaptive_denoised_img.png'))
plt.clf()
plt.plot(post_values_0, label='H Channel')
plt.plot(post_values_1, label='S Channel')
plt.plot(post_values_2, label='V Channel')
plt.legend()
plt.xlabel('Number of iterations')
plt.ylabel('Objective Function (log)')
plt.title('Discontinuity_Adaptive prior')
plt.savefig(os.path.join(save_results_dir, 'adaptive_all_channel_objective_function.png'))

print('\nRMSE after discontinuity adaptive prior denoising:', rrmse(noiseless_img, denoised_img))



