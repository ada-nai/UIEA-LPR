import argparse
import vpd
import lpr

def get_args():
   
    parser = argparse.ArgumentParser("Vehicle License Plate Detection")
   
    # c_desc = "CPU extension file location, if applicable"
    # d_desc = "Device, if not CPU (GPU, FPGA, MYRIAD)"
    i_desc = "The location of the input image"
    m1_desc = "The location of the model XML file - vpd"
    m2_desc = "The location of the model XML file - lpr"
 
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')
 
    required.add_argument("-i", help=i_desc, required=True)
    required.add_argument("-m1", help=m1_desc, required=True)
    required.add_argument("-m2",help=m2_desc, required=True)
    # optional.add_argument("-c", help=c_desc, default=None)
    # optional.add_argument("-d", help=d_desc, default="CPU")
    args = parser.parse_args()
 
    return args


# Run the file -
#python3 main.py -i "images/car_1.bmp" -m1 "vpd_model/FP16/vehicle-license-plate-detection-barrier-0106.xml" -m2 "/opt/intel/UIEA-LPR/lpr_model/FP16/license-plate-recognition-barrier-0001.xml"

def main():
    args = get_args()
    print(args)
    mod = vpd.perform_inference(args) # inference -> vpd
    print('mod obtained')
    lpr.perform_inference(mod,args.m2) # inference -> lpr
 
if __name__ == "__main__":
    main()


