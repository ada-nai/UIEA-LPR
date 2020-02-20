import argparse
import cv2
import numpy as np
import lpr
 
from vinfer import Network
 

def preprocessing(input_image, height, width):
    
    '''
	Given an input image, height and width:
	- Resize to width and height
	- Transpose the final "channel" dimension to be first
	- Reshape the image to add a "batch" of 1 at the start 
	'''
    image = input_image.copy()
    image = cv2.resize(image, (width, height))
    image = image.transpose((2, 0, 1))
    image = image.reshape(1, 3, width, height)
 
    return image
 
 
def perform_inference(args):
   
    '''
	Performs inference on an input image, given a model.
	'''

    # Load model
    inference_network = Network()
    n, c, h, w = inference_network.load_model(args.m1) 
    image = cv2.imread(args.i)

    #Preprocess input image
    preprocessed_image = preprocessing(image, h, w)

    # Perform inference
    inference_network.sync_inference(preprocessed_image)
    output = inference_network.extract_output()
    
    # Extract most probable output
    out = output['DetectionOutput_']
    
    # Extract bounding box for result that predicts license plate with highest confidence level
    resh_out = np.reshape(out, (200, 7)) #reshaping
    plate_out = resh_out[resh_out[:, 1] == 2] #choosing output that predict label == `license_plate`
    max_conf_index = np.argmax(plate_out[:, 2]) #choosing license plate bounding box with highest confidence
    max_conf_out = plate_out[max_conf_index] 
    print('Plate Detection output:', max_conf_out)

    # Scale the obtained output such that it bounds the plate from original image
    unX1 = int(image.shape[1]*max_conf_out[3])
    unY1 = int(image.shape[0]*max_conf_out[4])
    unX2 = int(image.shape[1]*max_conf_out[5])
    unY2 = int(image.shape[0]*max_conf_out[6])

    # Create bounding boxes for newly scaled co-ordinates
    cv2.rectangle(image, (unX1, unY1), (unX2, unY2), color = (0, 255, 0))

    # extract license plate from the original image
    extracted = image[unY1:unY2, unX1:unX2]
    print('extracted shape:', extracted.shape)
    
    cv2.imwrite('extracted_lp.jpg', extracted)
    #mod = cv2.resize(extracted, (300, 300))
    print('done with detection and extraction')
    return extracted

