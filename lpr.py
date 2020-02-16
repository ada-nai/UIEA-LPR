import cv2
import os
import argparse
import numpy as np
from linfer import Network
from encoding import get_encoding

def decode_output(out, encoding):

	'''
	Converts inference output from shape (1, 88, 1, 1) to (88, 1)
	Also, decodes output to meaningful plate number 
	'''
	fin_output = np.squeeze(out).tolist()
	plate_number = [encoding[code] for code in fin_output if code!= -1]

	return plate_number

def perform_inference(plate_image,args):

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
	eval = inference_network.load_model(args.m2)
	print('input_shapes:', eval)
	# Read the input image
	image = plate_image.copy()
	# Preprocess the input image
	# preprocessed_image = preprocessing(image, h, w)
	print('done with preprocessing...')
	# Create the second input for the lp_recognition model
	# of the form [0,1,...,1] of shape (88, 1)
	seq_ind = np.ones(88)
	seq_ind[0] = 0
	seq_ind = seq_ind.reshape(88, 1)
	# Perform synchronous inference on the image
	inference_network.sync_inference(plate_image, seq_ind)
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

	


