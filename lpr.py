import cv2
import os
import argparse
import numpy as np
from linfer import Network
from encoding import get_encoding

def preprocessing(input_image, height, width):
	'''
	Given an input image, height and width:
	- Resize to width and height
	- Transpose the final "channel" dimension to be first
	- Reshape the image to add a "batch" of 1 at the start 
	'''
	image = np.copy(input_image)
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

	# c_desc = "CPU extension file location, if applicable"
	# d_desc = "Device, if not CPU (GPU, FPGA, MYRIAD)"
	

	# -- Add required and optional groups
	parser._action_groups.pop()
	required = parser.add_argument_group('required arguments')
	optional = parser.add_argument_group('optional arguments')

	# -- Create the arguments
	required.add_argument("-i", help=i_desc, required=True)
	required.add_argument("-m", help=m_desc, required=True)

	#required.add_argument("-t", help=t_desc, required=True)
	# optional.add_argument("-c", help=c_desc, default= None)
	# optional.add_argument("-d", help=d_desc, default="CPU")
	args = parser.parse_args()

	return args

def decode_output(out, encoding):
	'''
	Converts inference output from shape (1, 88, 1, 1) to (88, 1)
	Also, decodes output to meaningful plate number 
	'''
	 
	fin_output = np.squeeze(out).tolist()
	plate_number = [encoding[code] for code in fin_output if code!= -1]
	return plate_number

def perform_inference(args):
	'''
	Performs inference on an input image, given a model.
	'''

	# Create a Network for using the Inference Engine
	inference_network = Network()

	# Load the model in the network, and obtain its input shape
	# n = 1
	# c = 3
	h = 24
	w = 94

	eval = inference_network.load_model(args.m)
	print('input_shapes:', eval)

	# Read the input image
	image = cv2.imread(args.i)

	# Preprocess the input image
	preprocessed_image = preprocessing(image, h, w)
	print('done with preprocessing...')

	# Create the second input for the lp_recognition model
	# of the form [0,1,...,1] of shape (88, 1)
	seq_ind = np.ones(88)
	seq_ind[0] = 0
	seq_ind = seq_ind.reshape(88, 1)
	
	# Perform synchronous inference on the image
	inference_network.sync_inference(preprocessed_image, seq_ind)

	# Obtain the output of the inference request
	gen_output = inference_network.extract_output()
	output = gen_output['decode']
	#print(gen_output)
	#print(type(output['decode']))
	
	# Obtain the decoded output
	lp_codes = get_encoding()
	numbers_list = decode_output(output, lp_codes)

	print('Recognized License plate number is :')
	for item in numbers_list:
		print(str(item), end='')

	print()
	


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

