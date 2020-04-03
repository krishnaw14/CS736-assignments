import cv2 
import numpy as np 
import matplotlib.pyplot as plt 
import skimage
from skimage import io
from skimage.transform import radon, rescale, iradon
from skimage.data import shepp_logan_phantom
import os 

from helper_functions import *

def plot(img, title, save_path):
	plt.imshow(img, cmap='gray')
	plt.title(title)
	plt.savefig(save_path)
	plt.clf()

def filtered_backprojection(img, filter_name, L_name, L_value, save_dir):

	theta_values = np.arange(60)*3
	sinogram = radon(img, theta=theta_values, circle=False)

	sinogram_filtered = my_filter(sinogram, filter_name, L_value)

	# coeff = 1 if filter_name is None else 0.5
	img_recon = 0.5*iradon(sinogram_filtered, theta=theta_values, filter_name=None, circle=False)

	plot(img_recon, title='Reconstructed Image. Filter = {}, L = {}'.format(filter_name, L_name), 
				save_path=os.path.join(save_dir, 'recon_filter_{}_L_{}.png'.format(filter_name, L_name))
				)

	rrmse = get_rrmse(img, img_recon)
	return rrmse
			

def main(img_path='../data/SheppLogan256.png', _dir='../results/q1'):

	img = plt.imread(img_path)

	theta_values = np.arange(60)*3
	sinogram = radon(img, theta=theta_values, circle=False)
	img_recon = iradon(sinogram, theta=theta_values, filter_name=None, circle=False)

	plot(img, title='Original Image', save_path=os.path.join(_dir, 'original.png'))
	plot(img_recon, title='Reconstruced Image without filter', save_path=os.path.join(_dir, 'img_recon_no_filter.png'))


	N = img.shape[0]
	L_names = ['w_max', 'half w_max']
	L_values = [np.pi, 0.5*np.pi]
	filter_names = ['ram-lak']

	# Part A
	save_dir = os.path.join(_dir, 'part_a')
	os.makedirs(save_dir, exist_ok=True)
	for filter_name in filter_names:
		for L_name, L_value in zip(L_names, L_values):
			if filter_name == None:
				L_name, L_value = None, None
			rrmse = filtered_backprojection(img, filter_name, L_name, L_value, save_dir)
			print(rrmse)

	# Part B
	print('---------------------------Question 1: Part B---------------------------')
	save_dir = os.path.join(_dir, 'part_b')
	os.makedirs(save_dir, exist_ok=True)
	S0 = cv2.GaussianBlur(img, (5,5), 0)
	S1 = cv2.GaussianBlur(img, (5,5), 1)
	S5 = cv2.GaussianBlur(img, (5,5), 5)
	plot(S0, title='S0', save_path=os.path.join(save_dir, 's0.png'))
	plot(S1, title='S1', save_path=os.path.join(save_dir, 's1.png'))
	plot(S5, title='S5', save_path=os.path.join(save_dir, 's5.png'))

	theta_values = np.arange(60)*3

	sinogram = radon(S0, theta=theta_values, circle=False)
	recon_S0 = 0.5*iradon(sinogram, theta=theta_values, filter_name='ramp', circle=False)

	sinogram = radon(S1, theta=theta_values, circle=False)
	recon_S1 = 0.5*iradon(sinogram, theta=theta_values, filter_name='ramp', circle=False)

	sinogram = radon(S5, theta=theta_values, circle=False)
	recon_S5 = 0.5*iradon(sinogram, theta=theta_values, filter_name='ramp', circle=False)

	plot(recon_S0, title='Reconstructed S0', save_path=os.path.join(save_dir, 'recon_s0.png'))
	plot(recon_S1, title='Reconstructed S1', save_path=os.path.join(save_dir, 'recon_s1.png'))
	plot(recon_S5, title='Reconstructed S5', save_path=os.path.join(save_dir, 'recon_s5.png'))

	print('RRMSE for filtered S0:', get_rrmse(S0, recon_S0))
	print('RRMSE for filtered S1:', get_rrmse(S1, recon_S0))
	print('RRMSE for filtered S5:', get_rrmse(S5, recon_S0))


	# Part C
	save_dir = os.path.join(_dir, 'part_c')

if __name__ == '__main__':
	main()


