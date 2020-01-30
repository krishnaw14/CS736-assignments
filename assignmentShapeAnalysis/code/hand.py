# Question 2: Experiment on human hand dataset
import os
import numpy as np 
import matplotlib.pyplot as plt 
import h5py

from functions import *

data_path = '../data/hand/data.mat'
f = h5py.File(data_path)
for k, v in f.items():
	data = np.array(v)

for i in range(40):
	plt.plot(data[i,:,0], data[i,:,1])

plt.show()

# Shape of Data is 40x56x2. 40 images. 56 points per each image pointset. 2 dimensions of the points.

# Align the pointsets

# Pick two pointsets
i1, i2 = np.random.randint(low=0, high=40, size=2)
z1 = np.expand_dims(data[i1],0)
z2 = np.expand_dims(data[i2],0)

# plt.plot(z1[0,:,0], z1[0,:,1])
# plt.plot(z2[0,:,0], z2[0,:,1])
# plt.show()

z = np.concatenate((z1,z2), axis=0)
zprime = compute_preshape_space(z)

R = compute_optimal_rotation(zprime[0], zprime[1])

z1_ = np.matmul(R, zprime[0])
z2_ = np.matmul(R, zprime[1])

# plt.plot(z1_[:,0], z1_[:,1])
# plt.plot(z2_[:,0], z2_[:,1])
# plt.show()

mean, z = compute_mean(data)
plt.plot(mean[0,:,0], mean[0,:,1], '--')

for i in range(40):
	plt.plot(z[i,:,0], z[i,:,1])

plt.show()
import pdb; pdb.set_trace()

