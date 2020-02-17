import argparse
import vpd
import lpr
import cv2

def get_args():
   
    parser = argparse.ArgumentParser("Vehicle License Plate Detection")

    # define required parameters
    i_desc = "The location of the input image"
    m1_desc = "The location of the model XML file - vpd"
    m2_desc = "The location of the model XML file - lpr"
 
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')
 
    required.add_argument("-i", help = i_desc, required = True)
    required.add_argument("-m1", help = m1_desc, required = True)
    required.add_argument("-m2", help = m2_desc, required = True)
    args = parser.parse_args()
 
    return args

def main():
    # Obtain arguments
    args = get_args()

    # Obtain the detected vehicle plate frame
    mod = vpd.perform_inference(args) # inference - > vpd
    print('mod obtained')

    # Obtain recognized license plate number 
    fin_out = lpr.perform_inference(mod, args.m2) # inference - > lpr
    print('fin_out', fin_out)

    # Write the output to image and save to disk
    # Output is saved to `Recognized_plate.jpg`
    input_image = cv2.imread(args.i)
    cv2.putText(input_image, fin_out, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX, 
        0.7, (0, 255, 0), 2)
    cv2.imwrite('Recognized_plate.jpg', input_image)
 
if __name__ == "__main__":
    main()

# Run the file -
# python3 main.py -i PATH_TO_IMAGE/car_1.bmp -m1 'PATH_TO_VEHICLE-DETECTION-MODEL/FP16/vehicle-license-plate-detection-barrier-0106.xml' -m2 'PATH_TO_VEHICLE-RECOGNITION-MODEL/FP16/license-plate-recognition-barrier-0001.xml' 


