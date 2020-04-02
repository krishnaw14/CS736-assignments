import cv2 
import numpy as np 
import matplotlib.pyplot as plt 
import skimage
from skimage import io
from skimage.transform import radon, rescale, iradon
from skimage.data import shepp_logan_phantom
import os 

from helper_functions import *

def filtered_backprojection(img_path = '../data/SheppLogan256.png')

	img = plt.imread(img_path)
	theta_values = np.arange(60)*3

	sinogram = radon(img, theta=theta_values, circle=True, preserve_range=True)
	reconstruction_fbp = iradon(sinogram, theta=theta_values, circle=True)

	import pdb; pdb.set_trace()

if __name__ == '__main__':
	filtered_backprojection()