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
print (sinogram.shape)
# Solving for A (High computational complexity)

A = []
size = 128*128
for i in range(size):
	a = np.zeros(size)
	a = np.expand_dims(a, axis = 1)
	a[i] = 1
	a = radon(a,theta)
	A.append(a)

A = np.vstack(A)
A = np.transpose(A)
sinogram_from_manual = np.dot(A,ChestPhantom.reshape(-1,1))

# a = np.eye(size)
# A = radon(a, theta)
# A = np.transpose(A)
# Finding via knowledge of both the sinogram and the image
#A = np.linalg.lstsq(ChestPhantom.reshape(-1,1),sinogram.reshape(-1,1))

diff = np.max(sinogram) - np.min(sinogram)
noise = np.random.normal(0,diff*0.02,sinogram.shape)
sinogram_withnoise = sinogram + noise

img_recon = iradon(sinogram_withnoise, theta)
print (rrmse(img_recon,ChestPhantom))

#part d Tikhonov 
from sklearn.linear_model import Ridge 
val = 100000
for alpha in range(0,100,1):
	alpha = alpha/100
	clf = Ridge(alpha = alpha)
	clf.fit(ChestPhantom,sinogram)
	y = clf.predict(ChestPhantom)
	img_recon = iradon(y,theta)
	a = rrmse(img_recon,ChestPhantom)
	print (a)
	if a < val:
		param = alpha
		val = a 
print (param) 
# Part e 

denoise(img_recon, ChestPhantom, alpha=0.0875, optimize_mode=False, prior='quadratic', 
	save_results_dir=os.path.join(save_results_dir, 'quadratic', 'img_recon'))
print('RRMSE at 1.2 times optimum alpha=', round(denoise(img_recon, ChestPhantom, alpha=0.0875*1.2, optimize_mode=True, prior='quadratic', 
	save_results_dir=os.path.join(save_results_dir, 'quadratic', 'img_recon')), 5))
print('RRMSE at 0.8 times optimum alpha=', round(denoise(img_recon, ChestPhantom, alpha=0.0875*0.8, optimize_mode=True, prior='quadratic', 
	save_results_dir=os.path.join(save_results_dir, 'quadratic', 'img_recon')), 5))
print('---------------------------------------------------------------------------------\n')


denoise(img_recon, ChestPhantom, alpha=0.8748, gamma=0.0023, optimize_mode=False, prior='discontinuity_adaptive', 
	save_results_dir=os.path.join(save_results_dir, 'discontinuity_adaptive', 'img_recon'))
print('RRMSE at 1.2 times optimum alpha=', round(denoise(img_recon, ChestPhantom, alpha=0.8748*1.2, gamma=0.0023, optimize_mode=True, prior='discontinuity_adaptive', 
	save_results_dir=os.path.join(save_results_dir, 'discontinuity_adaptive', 'img_recon')), 5))
print('RRMSE at 0.8 times optimum alpha=', round(denoise(img_recon, ChestPhantom, alpha=0.8748*0.8, gamma=0.0023, optimize_mode=True, prior='discontinuity_adaptive', 
	save_results_dir=os.path.join(save_results_dir, 'discontinuity_adaptive', 'img_recon')), 5))
print('RRMSE at 1.2 times optimum gamma=', round(denoise(img_recon, ChestPhantom, alpha=0.8748, gamma=0.0023*1.2, optimize_mode=True, prior='discontinuity_adaptive', 
	save_results_dir=os.path.join(save_results_dir, 'discontinuity_adaptive', 'img_recon')), 5))
print('RRMSE at 0.8 times optimum gamma=', round(denoise(img_recon, ChestPhantom, alpha=0.8748, gamma=0.0023*0.8, optimize_mode=True, prior='discontinuity_adaptive', 
	save_results_dir=os.path.join(save_results_dir, 'discontinuity_adaptive', 'img_recon')), 5))


denoise(img_recon, ChestPhantom, alpha=0.8748, gamma=0.00211, optimize_mode=False, prior='discontinuity_adaptive_huber', 
	save_results_dir=os.path.join(save_results_dir, 'discontinuity_adaptive_huber', 'img_recon'))
print('RRMSE at 1.2 times optimum alpha=', round(denoise(img_recon, ChestPhantom, alpha=0.8748*1.2, gamma=0.00211, optimize_mode=True, prior='discontinuity_adaptive_huber', 
	save_results_dir=os.path.join(save_results_dir, 'discontinuity_adaptive_huber', 'img_recon')), 5))
print('RRMSE at 0.8 times optimum alpha=', round(denoise(img_recon, ChestPhantom, alpha=0.8748*0.8, gamma=0.00211, optimize_mode=True, prior='discontinuity_adaptive_huber', 
	save_results_dir=os.path.join(save_results_dir, 'discontinuity_adaptive_huber', 'img_recon')), 5))
print('RRMSE at 1.2 times optimum gamma=', round(denoise(img_recon, ChestPhantom, alpha=0.8748, gamma=0.00211*1.2, optimize_mode=True, prior='discontinuity_adaptive_huber', 
	save_results_dir=os.path.join(save_results_dir, 'discontinuity_adaptive_huber', 'img_recon')), 5))
print('RRMSE at 0.8 times optimum gamma=', round(denoise(img_recon, ChestPhantom, alpha=0.8748, gamma=0.00211*0.8, optimize_mode=True, prior='discontinuity_adaptive_huber', 
	save_results_dir=os.path.join(save_results_dir, 'discontinuity_adaptive_huber', 'img_recon')), 5))
print('---------------------------------------------------------------------------------\n')
