import matplotlib.pyplot as plt
import numpy as np
from denoise import denoise
import cv2 
from functions import *

function_type = sys.argv[1]

denoised_img = cv2.cvtColor(plt.imread('../data/histology_noiseless.png'), cv2.COLOR_BGR2HSV)
noisy_img = cv2.cvtColor(plt.imread('../data/histology_noisy.png'), cv2.COLOR_BGR2HSV)

for j in range(3):
	gamma_upper_bound = 1
	gamma_lower_bound = 0.00001
	alpha_upper_bound = 1
	alpha_lower_bound = 0.00001
	rrmse_final_min = 100
	alpha_opt = 0
	gamma_opt = 0	
	for i in range(10):
		for gamma in np.linspace(gamma_lower_bound,gamma_upper_bound,10):
			for alpha in np.linspace(alpha_lower_bound,alpha_upper_bound,10):
				rrmse_final = denoise(noisy_img[:,:,j], denoised_img[:,:,j], alpha=alpha,gamma = gamma, optimize_mode=True, prior=function_type)
				if rrmse_final < rrmse_final_min:
					rrmse_final_min = rrmse_final
					alpha_opt = alpha
					gamma_opt = gamma
		gamma_lower_bound = max(gamma_opt - gamma_opt/2, 0.00001) 
		gamma_upper_bound = min(gamma_opt + gamma_opt/2, 1.0)
		
		alpha_upper_bound = min(alpha_opt + alpha_opt/2, 1.0)
		alpha_lower_bound = max(alpha_opt - alpha_opt/2, 0.00001)

	print (alpha_opt)
	print (gamma_opt)

import pdb; pdb.set_trace()