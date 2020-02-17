import argparse
import cv2
import numpy as np
import lpr
 
from vinfer import Network
 

def preprocessing(input_image, height, width):
   
    image = input_image.copy()
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
    print(out)
   

    frame = image.copy()
    frame = cv2.resize(frame, (300, 300))
    print('frame shape:', frame.shape)

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


    cv2.rectangle(image, (unX1, unY1), (unX2, unY2), color=(0, 255, 0))

    # cv2.imwrite("CAR-PLATE-output.jpg", frame)
    # clone = image.copy()
    # clone = cv2.resize(clone, (300, 300))

    extracted = image[unY1:unY2, unX1:unX2]
    print('extracted shape:', extracted.shape)
    
    cv2.imwrite('extracted_lp.jpg', extracted)
    #mod = cv2.resize(extracted,(300, 300))
    print('done with detection and extraction')
    return extracted

