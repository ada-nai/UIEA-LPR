# UIEA-LPR

License Plate recognition from images of Chinese vehicles using [pre-trained models](https://software.intel.com/en-us/openvino-toolkit/documentation/pretrained-models) provided under the [Intel® Distribution of OpenVINO™ Toolkit](https://software.intel.com/en-us/openvino-toolkit)  



![](https://github.com/ada-nai/UIEA-LPR/blob/master/car_1.bmp) | 
|:--:| 
|*Input - Car image*|

![](https://github.com/ada-nai/UIEA-LPR/blob/master/Recognized_plate.jpg) | 
|:--:| 
|*Output - Recognized License Plate Number*|



Following pre-trained models were used for the application:

  1) [vehicle-license-plate-detection-barrier-0106](https://docs.openvinotoolkit.org/2019_R1/_vehicle_license_plate_detection_barrier_0106_description_vehicle_license_plate_detection_barrier_0106.html)
  2) [license-plate-recognition-barrier-0001 ](https://docs.openvinotoolkit.org/2019_R1/_license_plate_recognition_barrier_0001_description_license_plate_recognition_barrier_0001.html)

```
main.py     -> driver program 
            
vpd.py      -> vehicle/plate detection app
vinfer.py   -> inference helper file for vpd.py
            
lpr.py      -> license plate recognition app
linfer.py   -> inference helper file for lpr.py
encoding.py -> program for proper decoding of model output to license plate symbol
```

## Flow of the program

![](https://github.com/ada-nai/UIEA-LPR/tree/master/images/Flowdiag/FlowDiagram.png) | 
|:--:| 
|*Project Flow Diagram*|


1. `main.py` accepts arguments for inputs. Arguments to be provided are:
  i. Input image (i)
  ii. Path for vehicle/plate detection model (m1)
  iii. Path for license plate recognition model (m2)
2. Inference is performed on car image using the vehicle/plate detection model with `vpd.py` and `vinfer.py`. The output obtained is the frame of the detected license plate.
3. The output from step 2 is fed as input to the license plate recognition model. Inference is performed on the input using the license plate recognition model with `lpr.py` and `linfer.py`. The output obatined is an encoded array of the symbols in the license plate.
4. Finally, `encoding.py` is used to decode the encoded output obtained in step 3. 
5. The decoded output is written on the input image and saved to disk, by the name `Recognized_plate.jpg`

## Run the program (Linux)
To run the program on your local machine, follow the steps mentioned below:
1. Download the [The Intel® Distribution of OpenVINO™](https://software.intel.com/en-us/openvino-toolkit/choose-download/free-download-linux). Follow the [installation guide if necessary](https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_linux.html)
2. Set the environment variables by pasting the command as mentioned in the `vinovars.txt` file to the terminal window. ~~If it is not added explicitly to your profile~~.
3. Run the program by executing the following command:
`python3 main.py -i PATH_TO_IMAGE/car_1.bmp -m1 'PATH_TO_VEHICLE-DETECTION-MODEL/FP16/vehicle-license-plate-detection-barrier-0106.xml' -m2 'PATH_TO_VEHICLE-RECOGNITION-MODEL/FP16/license-plate-recognition-barrier-0001.xml'`

NB: replace `PATH` by the appropriate path to the image/model in your system.


## Limitations
The project uses two pre-trained models, namely - license plate detection model and license plate recognition model. The models have been trained by feeding 'n' images. If we give an input image to our model which doesn't match the characteristics of images from which the model has been trained, the model fails to detect the specific characteristic that we are trying to detect, the vehicle license plate in our case. To solve this issue, we need to re-train the model by feeding more quality images. By doing this, the success rate of our detection will increase considerably.


## Applications
The project after some improvements can be used in areas of high-security concern. The license plate number can be recorded for every vehicle entering the desired premise. This will help different organizations to keep a track of the vehicles entering their premises and will reduce the risk of any unwanted issues. The vehicles which are not registered, will not be allowed to enter the area until the vehicle owner registers his/her vehicle in the system.


## Future scope
The project works only on images. It can be further improved by implementing video inputs. By doing so, the application can be deployed at the edge and the license plates can be detected in real-time. Such systems are currently being used in places where we require high security - in banks, different organizations, etc. The project can be further improved by making a full-fledged application - A web application where in the users are expected to drag and drop an image of their choice in a field, inference will be performed and the license plate number is printed out. An application can be built for android/iOS users which could detect and recognize the license plates in real-time.
