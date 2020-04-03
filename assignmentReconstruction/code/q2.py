#Author: Vipul Ramtekkkar | Krishna Wadhwani
#Rollno: 16D110013
#Assignment 3: Image Reconstruction
#References: https://scikit-image.org/docs/dev/auto_examples/transform/plot_radon_transform.html
import os 
import numpy as np 
import cv2 
import matplotlib.pyplot as plt
from skimage.transform import radon, iradon
from helper_functions import get_rrmse
from functions import *
from denoise import denoise
from filtered_backprojection import filtered_backprojection

save_results_dir = '../results/q2/'

ChestPhantom = plt.imread("../data/ChestPhantom.png")
print (ChestPhantom.shape)
theta = np.linspace(start = 0, stop = 180,num = 180,endpoint = False, dtype = int)
sinogram = radon(ChestPhantom, theta, circle = True)
# Solving for A (High computational complexity)



#A = np.linalg.lstsq(ChestPhantom.reshape(-1,1),sinogram.reshape(-1,1))

diff = np.max(sinogram) - np.min(sinogram)
noise = np.random.normal(0,diff*0.02,sinogram.shape)
sinogram_withnoise = sinogram + noise

img_recon = iradon(sinogram_withnoise, theta)

#part d Tikhonov 
from sklearn.linear_model import Ridge 

clf = Ridge(alpha = 1.0)
clf.fit(img_recon,ChestPhantom)
Ridge()


# Part e 

denoise(img_recon, ChestPhantom, alpha=0.0875, optimize_mode=False, prior='quadratic', 
	save_results_dir=os.path.join(save_results_dir, 'quadratic', 'low_noise_level'))
print('RRMSE at 1.2 times optimum alpha=', round(denoise(img_recon, ChestPhantom, alpha=0.0875*1.2, optimize_mode=True, prior='quadratic', 
	save_results_dir=os.path.join(save_results_dir, 'quadratic', 'low_noise_level')), 5))
print('RRMSE at 0.8 times optimum alpha=', round(denoise(img_recon, ChestPhantom, alpha=0.0875*0.8, optimize_mode=True, prior='quadratic', 
	save_results_dir=os.path.join(save_results_dir, 'quadratic', 'low_noise_level')), 5))
print('---------------------------------------------------------------------------------\n')


denoise(img_recon, ChestPhantom, alpha=0.8748, gamma=0.0023, optimize_mode=False, prior='discontinuity_adaptive', 
	save_results_dir=os.path.join(save_results_dir, 'discontinuity_adaptive', 'low_noise_level'))
print('RRMSE at 1.2 times optimum alpha=', round(denoise(img_recon, ChestPhantom, alpha=0.8748*1.2, gamma=0.0023, optimize_mode=True, prior='discontinuity_adaptive', 
	save_results_dir=os.path.join(save_results_dir, 'discontinuity_adaptive', 'low_noise_level')), 5))
print('RRMSE at 0.8 times optimum alpha=', round(denoise(img_recon, ChestPhantom, alpha=0.8748*0.8, gamma=0.0023, optimize_mode=True, prior='discontinuity_adaptive', 
	save_results_dir=os.path.join(save_results_dir, 'discontinuity_adaptive', 'low_noise_level')), 5))
print('RRMSE at 1.2 times optimum gamma=', round(denoise(img_recon, ChestPhantom, alpha=0.8748, gamma=0.0023*1.2, optimize_mode=True, prior='discontinuity_adaptive', 
	save_results_dir=os.path.join(save_results_dir, 'discontinuity_adaptive', 'low_noise_level')), 5))
print('RRMSE at 0.8 times optimum gamma=', round(denoise(img_recon, ChestPhantom, alpha=0.8748, gamma=0.0023*0.8, optimize_mode=True, prior='discontinuity_adaptive', 
	save_results_dir=os.path.join(save_results_dir, 'discontinuity_adaptive', 'low_noise_level')), 5))


denoise(img_recon, ChestPhantom, alpha=0.8748, gamma=0.00211, optimize_mode=False, prior='discontinuity_adaptive_huber', 
	save_results_dir=os.path.join(save_results_dir, 'discontinuity_adaptive_huber', 'low_noise_level'))
print('RRMSE at 1.2 times optimum alpha=', round(denoise(img_recon, ChestPhantom, alpha=0.8748*1.2, gamma=0.00211, optimize_mode=True, prior='discontinuity_adaptive_huber', 
	save_results_dir=os.path.join(save_results_dir, 'discontinuity_adaptive_huber', 'low_noise_level')), 5))
print('RRMSE at 0.8 times optimum alpha=', round(denoise(img_recon, ChestPhantom, alpha=0.8748*0.8, gamma=0.00211, optimize_mode=True, prior='discontinuity_adaptive_huber', 
	save_results_dir=os.path.join(save_results_dir, 'discontinuity_adaptive_huber', 'low_noise_level')), 5))
print('RRMSE at 1.2 times optimum gamma=', round(denoise(img_recon, ChestPhantom, alpha=0.8748, gamma=0.00211*1.2, optimize_mode=True, prior='discontinuity_adaptive_huber', 
	save_results_dir=os.path.join(save_results_dir, 'discontinuity_adaptive_huber', 'low_noise_level')), 5))
print('RRMSE at 0.8 times optimum gamma=', round(denoise(img_recon, ChestPhantom, alpha=0.8748, gamma=0.00211*0.8, optimize_mode=True, prior='discontinuity_adaptive_huber', 
	save_results_dir=os.path.join(save_results_dir, 'discontinuity_adaptive_huber', 'low_noise_level')), 5))
print('---------------------------------------------------------------------------------\n')
