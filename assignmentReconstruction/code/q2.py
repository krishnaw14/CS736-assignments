#Author: Vipul Ramtekkkar | Krishna Wadhwani
#Rollno: 16D110013
#Assignment 3: Image Reconstruction
#References: https://scikit-image.org/docs/dev/auto_examples/transform/plot_radon_transform.html

import numpy as np 
import cv2 
import matplotlib.pyplot as plt
from skimage.transform import radon
from helper_functions import get_rrmse
from filtered_backprojection import *


ChestPhantom = plt.imread("../data/ChestPhantom.png")
print (ChestPhantom.shape)
theta = np.linspace(start = 0, stop = 180,num = 180,endpoint = False, dtype = int)
signogram = radon(ChestPhantom, theta, circle = True)

A = np.linalg.solve(ChestPhantom.reshape(-1,1),signogram.reshape(-1,1))

diff = np.max(signogram) - np.min(signogram)
noise = np.random.normal(0,diff*0.02,signogram.shape)
signogram_withnoise = signogram + noise

filtered_backprojection(signogram_withnoise)

#part d Tikhonov 



