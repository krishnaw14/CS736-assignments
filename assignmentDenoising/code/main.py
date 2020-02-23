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

print('\n---------------------------------------------------------------------------------')
print('Denoising Low Noise Level Image...')
denoise(low_noise_img, denoised_img, beta=0.0875, optimize_mode=False, prior='quad', save_results_dir=os.path.join(save_results_dir, 'low_noise_level'))
print('---------------------------------------------------------------------------------\n')

print('\n---------------------------------------------------------------------------------')
print('Denoising Low Noise Level Image...')
denoise(low_noise_img, denoised_img, beta=0.8748, gamma = 0.0023, optimize_mode=False, prior='adapt', save_results_dir=os.path.join(save_results_dir, 'low_noise_level'))
print('---------------------------------------------------------------------------------\n')

print('\n---------------------------------------------------------------------------------')
print('Denoising Low Noise Level Image...')
denoise(low_noise_img, denoised_img, beta=0.8748, gamma = 0.00211, optimize_mode=False, prior='huber', save_results_dir=os.path.join(save_results_dir, 'low_noise_level'))
print('---------------------------------------------------------------------------------\n')

print('\n---------------------------------------------------------------------------------')
print('Denoising Medium Noise Level Image...')
denoise(med_noise_img, denoised_img, beta=0.1773, optimize_mode=False, prior='quad', save_results_dir=os.path.join(save_results_dir, 'med_noise_level'))
print('---------------------------------------------------------------------------------\n')

print('\n---------------------------------------------------------------------------------')
print('Denoising Medium Noise Level Image...')
denoise(med_noise_img, denoised_img, beta=0.874, gamma = 0.0051, optimize_mode=False, prior='adapt', save_results_dir=os.path.join(save_results_dir, 'med_noise_level'))
print('---------------------------------------------------------------------------------\n')

print('\n---------------------------------------------------------------------------------')
print('Denoising Medium Noise Level Image...')
denoise(med_noise_img, denoised_img, beta=0.8663, gamma = 0.0048, optimize_mode=False, prior='huber', save_results_dir=os.path.join(save_results_dir, 'med_noise_level'))
print('---------------------------------------------------------------------------------\n')

print('\n---------------------------------------------------------------------------------')
print('Denoising High Noise Level Image...')
denoise(high_noise_img, denoised_img, beta=0.2407, optimize_mode=False, prior='quad', save_results_dir=os.path.join(save_results_dir, 'high_noise_level'))
print('---------------------------------------------------------------------------------\n')

print('\n---------------------------------------------------------------------------------')
print('Denoising High Noise Level Image...')
denoise(high_noise_img, denoised_img, beta=0.8738, gamma = 0.008, optimize_mode=False, prior='adapt',save_results_dir=os.path.join(save_results_dir, 'high_noise_level'))
print('---------------------------------------------------------------------------------\n')

print('\n---------------------------------------------------------------------------------')
print('Denoising High Noise Level Image...')
denoise(high_noise_img, denoised_img, beta=0.7729, gamma = 0.01389, optimize_mode=False, prior='huber',save_results_dir=os.path.join(save_results_dir, 'high_noise_level'))
print('---------------------------------------------------------------------------------\n')

# Solve For question 2: Histology Images

# denoised_img = cv2.imread('../data/histology_noiseless.png')
# noisy_img = cv2.imread('../data/histology_noisy.png')

# print('Solving for Histology Image')
# denoise(noisy_img, denoised_img)