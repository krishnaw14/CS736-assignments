#Author: Vipul Ramtekkkar | Krishna Wadhwani
#Rollno: 16D110013
#Assignment 3: Image Reconstruction
#References: https://scikit-image.org/docs/dev/auto_examples/transform/plot_radon_transform.html

import numpy as np 
import cv2 
import matplotlib.pyplot as plt
from skimage.transform import radon 

ChestPhantom = plt.imread("../data/ChestPhantom.png")
theta = np.linspace(start = 0, stop = 180,num = 180,endpoint = False, dtype = int)
A = radon(ChestPhantom, theta, circle = True)
diff = np.max(A) - np.min(A)
noise = np.random.normal(0,diff*0.02,A.shape)

new_A = A + noise

'''
cv2.imshow("im", ChestPhantom)
cv2.imshow("final",new_A)
cv2.waitKey(0)
'''