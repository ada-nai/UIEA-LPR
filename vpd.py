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
    out = [k for i in out for j in i for k in j if(k[2] > 0.5 and k[1] != 1)]
    out = np.asarray(out).flatten()
    print('Plate Detection output:', out)
   

    frame = image.copy()
    frame = cv2.resize(frame, (300, 300))
    print('frame shape:', frame.shape)

    # Scale the obtained output such that it bounds the plate from original image
    xmin = int(out[3] * frame.shape[0])
    ymin = int(out[4] * frame.shape[1])
    xmax = int(out[5] * frame.shape[0])
    ymax = int(out[6] * frame.shape[1])
    
    pX1 = xmin/300
    pY1 = ymin/300
    pX2 = xmax/300
    pY2 = ymax/300
    
    unX1 = int(image.shape[1]*pX1)
    unY1 = int(image.shape[0]*pY1)
    unX2 = int(image.shape[1]*pX2)
    unY2 = int(image.shape[0]*pY2)

    # Create bounding boxes for newly scaled co-ordinates
    cv2.rectangle(image, (unX1, unY1), (unX2, unY2), color = (0, 255, 0))

    # extract license plate from the original image
    extracted = image[unY1:unY2, unX1:unX2]
    print('extracted shape:', extracted.shape)
    
    cv2.imwrite('extracted_lp.jpg', extracted)
    #mod = cv2.resize(extracted, (300, 300))
    print('done with detection and extraction')
    return extracted

