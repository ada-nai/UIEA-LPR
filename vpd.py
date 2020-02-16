import argparse
import cv2
import numpy as np
import lpr
 
from vinfer import Network
 

def preprocessing(input_image, height, width):
   
    image = np.copy(input_image)
    image = cv2.resize(image, (width, height))
    image = image.transpose((2,0,1))
    image = image.reshape(1, 3, width, height)
 
    return image
 
 
def perform_inference(args):
   
    inference_network = Network()
    n, c, h, w = inference_network.load_model(args.m1) #, args.d, args.c
    image = cv2.imread(args.i)
    preprocessed_image = preprocessing(image, h, w)
    inference_network.sync_inference(preprocessed_image)
    output = inference_network.extract_output()
    
    out = output['DetectionOutput_']
    out = [k for i in out for j in i for k in j if(k[2]>0.5 and k[1]!=1)]
    out = np.asarray(out).flatten()
   

    frame = preprocessed_image

    xmin = int(out[3] * frame.shape[2])
    ymin = int(out[4] * frame.shape[3])
    xmax = int(out[5] * frame.shape[2])
    ymax = int(out[6] * frame.shape[3])
    

    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color=(0, 255, 0))

    cv2.imwrite("CAR-PLATE-output.jpg", frame)
    clone = image.copy()
    clone = cv2.resize(clone, (300, 300))
    extracted = frame[ymin:ymax, xmin:xmax]
    cv2.imwrite('extracted_lp.jpg', extracted)
    mod = cv2.resize(extracted,(300, 300))

    return mod

