import numpy as np 
import os 
import matplotlib.pyplot as plt 

def visualize_checkpoints(img, pointset):
	img = plt.imread(img)

	fig, ax = plt.subplots()
	ax.imshow(img)
	
	x_values = list(pointset[0,:,0]); y_values = list(pointset[0,:,1])

	x_values.append(x_values[0]); y_values.append(y_values[0])
	ax.plot(x_values, y_values)
	plt.show()

# parser = argparse.ArgumentParser()
# parser.add_argument('--data_path', '-d', default='sample.png' , required=False, help="Path to the image")
# parser.add_argument('--pointset', '-p', required=True, help='path to store the pointsets')
# parser.add_argument('--overwrite', '-o', action='store_true', help='if overwriting existing pointset file is allowed')
# args = parser.parse_args()

pointset = np.load('leaf.npy')

img = 'data/leaf/data/leaf_1.png'
visualize_checkpoints(img, pointset)