import numpy as np
import matplotlib.pyplot as plt 
from skimage.transform import radon, rescale, iradon
import os 

from helper_functions import get_rrmse

def compute_optimal_theta(img_path, save_dir, data_name):

	os.makedirs(save_dir, exist_ok=True)

	img = plt.imread(img_path)

	angle_range = np.arange(1, 151)

	rrmse_values = []
	for theta in range(181):

		angle_range_theta = angle_range + theta
		sinogram = radon(img, theta=angle_range_theta, circle=False)
		img_recon = iradon(sinogram, theta=angle_range_theta, circle=False)

		rrmse = get_rrmse(img, img_recon)
		rrmse_values.append(rrmse)

	opti_rrmse = np.min(rrmse_values)
	opti_theta = np.argmin(rrmse_values)
	opti_sinogram = radon(img, theta=angle_range+opti_theta, circle=False)
	opti_recon = iradon(opti_sinogram, theta=angle_range+opti_theta, circle=False)

	print('-------------------{}-------------------'.format(data_name))
	print('Optimal Angle theta:', opti_theta)
	print('Optimal RRMSE:', opti_rrmse)

	plt.plot(rrmse_values)
	plt.title('{}'.format(data_name))
	plt.xlabel('theta in degrees')
	plt.ylabel('RRMSE')
	plt.savefig(os.path.join(save_dir, data_name+'rrmse_plot.png'), edgecolor='black')

	plt.clf()
	fig, (ax1, ax2) = plt.subplots(1, 2)
	fig.suptitle('{} Results'.format(data_name))
	ax1.imshow(img, cmap='gray')
	ax1.set_title('Original Image')
	ax2.imshow(opti_recon, cmap='gray')
	ax2.set_title('Optimal Reconstructed Image')
	fig.savefig(os.path.join(save_dir, data_name+'recon.png'), edgecolor='black')


if __name__ == '__main__':
	compute_optimal_theta('../data/SheppLogan256.png', save_dir='../results/q3', data_name='SheppLogan256')
	compute_optimal_theta('../data/ChestCT.png', save_dir='../results/q3', data_name='ChestCT')
