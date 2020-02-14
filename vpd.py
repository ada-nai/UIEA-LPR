import argparse
import cv2
import numpy as np
 
from vinfer import Network
 
def get_args():
   
    parser = argparse.ArgumentParser("Vehicle License Plate Detection")
   
    c_desc = "CPU extension file location, if applicable"
    d_desc = "Device, if not CPU (GPU, FPGA, MYRIAD)"
    i_desc = "The location of the input image"
    m_desc = "The location of the model XML file"
 
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')
 
    required.add_argument("-i", help=i_desc, required=True)
    required.add_argument("-m", help=m_desc, required=True)
    optional.add_argument("-c", help=c_desc, default=None)
    optional.add_argument("-d", help=d_desc, default="CPU")
    args = parser.parse_args()
 
    return args
 
 
def preprocessing(input_image, height, width):
   
    image = np.copy(input_image)
    image = cv2.resize(image, (width, height))
    image = image.transpose((2,0,1))
    image = image.reshape(1, 3, width, height)
 
    return image
 
# def get_plate(output, input_shape):
   
#     plate_class = output['DetectionOutput_']
#     print(plate_class.shape)
#     out_plate = np.empty([plate_class.shape[1], input_shape[0], input_shape[1]])
#     for t in range(len(plate_class[0])):
#         out_plate[t] = cv2.resize(plate_class[0][t], input_shape[0:2][::-1])  
#     print(out_plate)
 
#     return out_plate
   
 
 
# def get_mask(processed_output):
   
#     empty = np.zeros(processed_output.shape)
#     mask = np.dstack((empty, processed_output, empty))
 
#     return mask
 
 
# def create_output_image(image, output):
   
#     output = np.where(output[0]>0.5, 255, 0)
#     text_mask = get_mask(output)
#     image = image + text_mask
   
#     return image
 
def perform_inference(args):
   
    inference_network = Network()
    n, c, h, w = inference_network.load_model(args.m) #, args.d, args.c
    image = cv2.imread(args.i)
    # cv2.imshow('vpd', image)
    # print(image.shape)
    preprocessed_image = preprocessing(image, h, w)
    inference_network.sync_inference(preprocessed_image)
    output = inference_network.extract_output()
    
    out = output['DetectionOutput_']
    out = [k for i in out for j in i for k in j if(k[2]>0.5 and k[1]!=1)]
    out = np.asarray(out).flatten()
    print(out)

    frame = preprocessed_image
    print(frame.shape)

    #for detection in out.reshape(-1, 7):
    xmin = int(out[3] * frame.shape[2])
    ymin = int(out[4] * frame.shape[3])
    xmax = int(out[5] * frame.shape[2])
    ymax = int(out[6] * frame.shape[3])
    print(frame.shape[2])

    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color=(0, 255, 0))


    # processed_output = get_plate(output, image.shape)
   
    # output_image = create_output_image(image, processed_output)
    cv2.imwrite("CAR-PLATE-output.jpg", frame)
 
def main():
    args = get_args()
    perform_inference(args)
 
if __name__ == "__main__":
    main()