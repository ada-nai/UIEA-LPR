import cv2
import os
import argparse
import numpy as np
from linfer import Network

def preprocessing(input_image, height, width):
	'''
	Given an input image, height and width:
	- Resize to width and height
	- Transpose the final "channel" dimension to be first
	- Reshape the image to add a "batch" of 1 at the start 
	'''
	image = np.copy(input_image)
	#image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
	image = cv2.resize(image, (width, height))
	image = image.transpose((2,0,1))
	image = image.reshape(1, 3, height, width)

	return image

def get_args():
	'''
	Gets the arguments from the command line.
	'''

	parser = argparse.ArgumentParser("Basic Edge App with Inference Engine")
	# -- Create the descriptions for the commands
	i_desc = "The location of the input image"
	m_desc = "The location of the model XML file"

	c_desc = "CPU extension file location, if applicable"
	d_desc = "Device, if not CPU (GPU, FPGA, MYRIAD)"
	#t_desc = "The type of model: POSE, TEXT or CAR_META"

	# -- Add required and optional groups
	parser._action_groups.pop()
	required = parser.add_argument_group('required arguments')
	optional = parser.add_argument_group('optional arguments')

	# -- Create the arguments
	required.add_argument("-i", help=i_desc, required=True)
	required.add_argument("-m", help=m_desc, required=True)

	#required.add_argument("-t", help=t_desc, required=True)
	optional.add_argument("-c", help=c_desc, default= None)
	optional.add_argument("-d", help=d_desc, default="CPU")
	args = parser.parse_args()

	return args

def perform_inference(args):
	'''
	Performs inference on an input image, given a model.
	'''
	# Create a Network for using the Inference Engine
	inference_network = Network()
	# Load the model in the network, and obtain its input shape
	n = 1
	c = 3
	h = 24
	w = 94

	eval = inference_network.load_model(args.m)
	print('eval:', eval)

	# Read the input image
	image = cv2.imread(args.i)

	### TODO: Preprocess the input image
	
	preprocessed_image = preprocessing(image, h, w)

	seq_ind = np.ones(88)
	seq_ind[0] = 0

	# Perform synchronous inference on the image
	inference_network.sync_inference(preprocessed_image)

	# Obtain the output of the inference request
	output = inference_network.extract_output()
	# print((output['decode']))
	# print(output['decode'].shape)

	### TODO: Handle the output of the network, based on args.t
	### Note: This will require using `handle_output` to get the correct
	###       function, and then feeding the output to that function.
	# output_func = handle_output(args.t)
	# processed_output = output_func(output, image.shape)

	# Create an output image based on network
	
	#output_image = create_output_image(args.t, image, processed_output)

	# Save down the resulting image
	#cv2.imwrite("lp_output.png".format(args.t), output_image)

def main():
	args = get_args()
	perform_inference(args)

if __name__ == "__main__":
	main()

